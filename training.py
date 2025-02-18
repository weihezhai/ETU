import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# Specify the pretrained model name for Qwen2.5 7B.
# (Replace with the correct repo id if needed.)
MODEL_NAME = "QwenInc/qwen2.5-7b-hf"

# Load the entire dataset from the JSONL file.
# By default the dataset loads as one split ("train"), so we'll split it into train, develop, and test.
dataset = load_dataset("json", data_files="relation_prediction_dataset.jsonl")

# First, split off 20% as a temporary test split.
raw_train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
# Then, split that temporary test split equally into develop and test sets.
dev_test_split = raw_train_test["test"].train_test_split(test_size=0.5, seed=42)

train_dataset = raw_train_test["train"]       # 80% of the data
develop_dataset = dev_test_split["train"]       # 10% of the data
test_dataset = dev_test_split["test"]           # 10% of the data

# Load the tokenizer. This tokenizer should match the model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# It is a good idea to set the pad token to eos_token if not already set.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    """
    For each example, we build a text prompt by concatenating the question,
    the input trajectory (joined by commas), and the fixed string "Relation:".
    The target (the relation to predict) is appended after the prompt.
    
    We then tokenize the full text. In order to train the model in a causal LM
    fashion where only the target relation contributes to the loss, we also text‐tokenize
    the prompt and set all label tokens corresponding to the prompt portion to -100.
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
    
    # Tokenize the full text with truncation and padding.
    tokenized_full = tokenizer(
        full_texts, truncation=True, max_length=512, padding="max_length"
    )
    
    # Tokenize only the prompt to determine its length.
    tokenized_prompts = tokenizer(
        prompt_texts, truncation=True, max_length=512, padding="max_length"
    )
    
    # Create labels by copying the input_ids and setting the tokens corresponding to the prompt to -100.
    labels = []
    for full_ids, prompt_ids in zip(
        tokenized_full["input_ids"], tokenized_prompts["input_ids"]
    ):
        # Count the effective length of the prompt (stop at pad token)
        prompt_len = sum(1 for t in prompt_ids if t != tokenizer.pad_token_id)
        label_ids = full_ids.copy()
        label_ids[:prompt_len] = [-100] * prompt_len
        labels.append(label_ids)
    
    tokenized_full["labels"] = labels
    return tokenized_full

# Tokenize each split separately.
tokenized_train = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names
)
tokenized_dev = develop_dataset.map(
    preprocess_function, batched=True, remove_columns=develop_dataset.column_names
)
tokenized_test = test_dataset.map(
    preprocess_function, batched=True, remove_columns=test_dataset.column_names
)

# Load model in 16-bit precision and let device_map be automatic
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
)

# Set up training arguments.
# Using evaluation and saving strategy "epoch" ensures that the model is evaluated on the
# development set and a checkpoint is saved at the end of each epoch.
training_args = TrainingArguments(
    output_dir="/data/scratch/mpx602/ETU/qwen2.5/qwen2.5-7b-finetuned-relation",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluate after each epoch on the dev set.
    save_strategy="epoch",        # Save checkpoint at the end of each epoch.
    weight_decay=0.01,
    save_total_limit=3,
    report_to=["none"],
)

# Create a data collator for language modeling (mlm=False because this is a causal LM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    data_collator=data_collator,
)

def main():
    # Start training. The total training time will depend on your dataset size,
    # number of epochs, and overall effective batch size.
    print("Starting training...")
    trainer.train()  # Trainer will evaluate on the dev set and save checkpoints at each epoch.

    # Save the final model.
    trainer.save_model("/data/scratch/mpx602/ETU/qwen2.5/qwen2.5-7b-finetuned-relation")
    print("Training completed and final model saved.")

    # Evaluate the final checkpoint on the development set.
    dev_results = trainer.evaluate(eval_dataset=tokenized_dev)
    print("Development Set Evaluation Results:")
    print(dev_results)

    # Evaluate the final checkpoint on the test set.
    test_results = trainer.evaluate(eval_dataset=tokenized_test)
    print("Test Set Evaluation Results:")
    print(test_results)

if __name__ == "__main__":
    main() 