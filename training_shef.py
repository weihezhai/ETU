import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset
from tqdm.auto import tqdm

############## Custom Progress Callback ###############
class CustomProgressCallback(TrainerCallback):
    '''
    A custom TrainerCallback that uses tqdm progress bars to monitor training.
    It shows perepoch and perstep progress. Additionally, on each step it displays
    current GPU memory usage (if available).
    '''
    def on_train_begin(self, args, state, control, **kwargs):
        print("Training has begun!")
        if state.max_steps > 0:
            self.steps_per_epoch = int(state.max_steps // args.num_train_epochs)
        else:
            # Fallback: use 0 if we cannot compute (in some cases state.max_steps is 0)
            self.steps_per_epoch = 0
        self.epoch_bar = tqdm(total=args.num_train_epochs, desc="Epoch", dynamic_ncols=True)
        self.step_bar = None
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Reset the perepoch step progress bar.
        if self.step_bar is not None:
            self.step_bar.close()
        # Use computed steps per epoch or fallback to 100 if undefined.
        total_steps = self.steps_per_epoch if self.steps_per_epoch > 0 else 100
        self.step_bar = tqdm(total=total_steps, desc=f"Epoch {state.epoch:.2f} Steps", dynamic_ncols=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_bar is not None:
            self.step_bar.update(1)
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                self.step_bar.set_postfix({"GPU Mem (GB)": f"{mem:.2f}"})
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.step_bar is not None:
            self.step_bar.close()
            self.step_bar = None
        self.epoch_bar.update(1)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.epoch_bar.close()
        if self.step_bar is not None:
            self.step_bar.close()
        print("Training finished!")
        return control

############## End Custom Progress Callback ###############

def preprocess_function(examples, tokenizer):
    """
    For each example, build a text prompt as follows:

    "Question: <question>
     Trajectory: <node1, node2, ...>
     Relation: "

    Append the target relation to form the full text.
    The tokenization is done for the full text as well as for the prompt alone,
    and tokens corresponding to the prompt are masked (set to  minus 100) so that only the
    target relation tokens contribute to the loss.
    """
    full_texts = []
    prompt_texts = []
    for question, trajectory, target_relation in zip(
        examples["question"], examples["input_trajectory"], examples["target_relation"]
    ):
        prompt = f"Question: {question}\nTrajectory: {', '.join(trajectory)}\nRelation: "
        full_text = prompt + target_relation
        full_texts.append(full_text)
        prompt_texts.append(prompt)
    
    tokenized_full = tokenizer(
        full_texts, truncation=True, max_length=512, padding="max_length"
    )
    tokenized_prompts = tokenizer(
        prompt_texts, truncation=True, max_length=512, padding="max_length"
    )
    
    labels = []
    for full_ids, prompt_ids in zip(tokenized_full["input_ids"], tokenized_prompts["input_ids"]):
        # Determine effective prompt length (ignoring pad tokens)
        prompt_len = sum(1 for t in prompt_ids if t != tokenizer.pad_token_id)
        label_ids = full_ids.copy()
        label_ids[:prompt_len] = [-100] * prompt_len
        labels.append(label_ids)
    
    tokenized_full["labels"] = labels
    return tokenized_full

def main(args):
    # Use the provided model name
    MODEL_NAME = args.model_name

    # Load the entire dataset from the JSONL file.
    # The dataset loads as one split ("train"), so we'll split it into train, develop, and test.
    dataset = load_dataset("json", data_files="relation_prediction_dataset.jsonl")

    # Split off 20% for test, then equally split those into development and test.
    raw_train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
    dev_test_split = raw_train_test["test"].train_test_split(test_size=0.5, seed=42)

    train_dataset = raw_train_test["train"]       # 80% of the data
    develop_dataset = dev_test_split["train"]       # 10% of the data
    test_dataset = dev_test_split["test"]           # 10% of the data

    # Load the tokenizer corresponding to the model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize each dataset split.
    tokenized_train = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_dev = develop_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=develop_dataset.column_names
    )
    tokenized_test = test_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )

    # Load model in 16bit precision using automatic device mapping.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Set up training arguments.
    # NOTE: max_grad_norm is set to 0.0 to disable gradient clipping. This is a workaround for
    # the "Attempting to unscale FP16 gradients" error.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="epoch",  # Evaluate after each epoch on the dev set.
        save_strategy="epoch",        # Save checkpoint at the end of each epoch.
        weight_decay=0.01,
        save_total_limit=3,
        report_to=["none"],
        disable_tqdm=True,  # Disable default tqdm; our custom progress callback will handle progress.
    )

    # Create a data collator for causal language modeling.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize the Trainer; include our custom progress callback.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        callbacks=[CustomProgressCallback()]
    )

    print("Starting training...")
    trainer.train()  # Trainer will evaluate and save checkpoints at each epoch.
    
    # Save the final model.
    trainer.save_model(args.output_dir)
    print("Training completed and final model saved.")

    # Evaluate on the development set.
    dev_results = trainer.evaluate(eval_dataset=tokenized_dev)
    print("Development Set Evaluation Results:")
    print(dev_results)

    # Evaluate on the test set.
    test_results = trainer.evaluate(eval_dataset=tokenized_test)
    print("Test Set Evaluation Results:")
    print(test_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5 7B on the relation prediction dataset."
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/scratch/mpx602/ETU/qwen2.5/qwen2.5-7b-finetuned-relation",
        help="Directory to save checkpoints and model."
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B", help="Name of the pretrained model.")
    args = parser.parse_args()

    main(args)