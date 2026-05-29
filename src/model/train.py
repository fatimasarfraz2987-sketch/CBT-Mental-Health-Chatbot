"""
Fine-tuning script for dialogue models.
Trains DialoGPT or Flan-T5 on CBT therapy dialogue pairs.
"""
import torch
import json
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    """Custom dataset for dialogue pairs."""
    
    def __init__(self, json_path, tokenizer, max_length=512):
        """
        Initialize dataset from JSON file.
        
        Args:
            json_path: Path to cbt_pairs.json
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
        """
        with open(json_path, 'r') as f:
            self.pairs = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        question = pair['question']
        answer = pair['answer']
        
        # Combine question and answer with separator
        text = f"{question} {self.tokenizer.eos_token} {answer}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def train_model(
    model_name="microsoft/DialoGPT-small",
    data_path="data/processed/cbt_pairs.json",
    output_dir="models/fine_tuned",
    num_epochs=3,
    batch_size=8
):
    """
    Train dialogue model on CBT pairs.
    
    Args:
        model_name: Base model from HuggingFace
        data_path: Path to training data
        output_dir: Where to save checkpoint
        num_epochs: Number of training epochs
        batch_size: Training batch size
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create dataset
    dataset = DialogueDataset(data_path, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    final_path = os.path.join(output_dir, "checkpoint-final")
    trainer.save_model(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    train_model()
