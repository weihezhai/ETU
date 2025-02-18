import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import os

def load_dataset(file_path):
    """Load and preprocess the relation prediction dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            # Format the input text
            input_text = (
                f"Question: {example['question']}\n"
                f"Current path: {' -> '.join(example['input_trajectory'])}\n"
                f"Target node: {example['target_node']}\n"
                f"Predict the relation that connects the last node to the target node:"
            )
            # The target is just the relation
            target_text = f"{example['target_relation']}"
            
            data.append({
                "input": input_text,
                "output": target_text
            })
    
    return Dataset.from_list(data)

def prepare_training_data(dataset, tokenizer):
    """Prepare the dataset for training by tokenizing inputs and outputs."""
    def tokenize_function(examples):
        # Combine input and output with a separator
        prompts = examples["input"]
        completions = examples["output"]
        
        # Format as instruction format
        texts = [
            f"{prompt}\n{completion}</s>"
            for prompt, completion in zip(prompts, completions)
        ]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset

def main():
    # Configuration
    model_name = "Qwen/Qwen1.5-7B"
    dataset_path = "relation_prediction_dataset_with_target.jsonl"
    output_dir = "relation_prediction_model"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        save_steps=500,
        warmup_steps=100,
        save_total_limit=2,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Load and prepare dataset
    dataset = load_dataset(dataset_path)
    train_dataset = prepare_training_data(dataset, tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train
    trainer.train()
    
    # Save the final model
    trainer.save_model()

if __name__ == "__main__":
    main() 