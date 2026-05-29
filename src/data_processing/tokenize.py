"""
WEEK 1: Data & Foundation (Part 2)
Goal: Tokenize cbt_pairs.json and prepare for model training.

Author Comment:
    Tokenize cbt_pairs.json using DialoGPT tokenizer.
    Apply padding and truncation to max_length=512.
    Split into train (80%), validation (10%), test (10%) sets.
    Save as HuggingFace Dataset to disk at data/processed/tokenized/
    
    Let Copilot generate the implementation — review each suggested block before accepting.
"""
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


class DialogueTokenizer:
    """Tokenize dialogue pairs for training."""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium", max_length=512):
        """
        Initialize tokenizer with specified model.
        
        Args:
            model_name: HuggingFace model identifier (DialoGPT-medium for training)
            max_length: Maximum sequence length (512 is standard)
        """
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # DialoGPT uses EOS as padding
        self.max_length = max_length
        print(f"✓ Tokenizer loaded (vocab size: {len(self.tokenizer)})")
    
    def tokenize_pair(self, input_text, response_text):
        """
        Tokenize a question-answer pair for DialoGPT.
        
        Args:
            input_text: User input/question
            response_text: Therapist response
            
        Returns:
            Dictionary with tokenized inputs
        """
        # DialoGPT format: concatenate with EOS token
        combined_text = input_text + self.tokenizer.eos_token + response_text
        
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def tokenize_dataset(self, input_json, output_dir):
        """
        Tokenize entire dataset and split into train/val/test.
        
        Args:
            input_json: Path to cbt_pairs.json
            output_dir: Directory to save tokenized dataset
        """
        print("\n" + "=" * 60)
        print("WEEK 1 TOKENIZATION PIPELINE")
        print("=" * 60)
        print(f"\n[Step 1/4] Loading CBT pairs from {input_json}...")
        
        # Load JSON
        with open(input_json, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        
        print(f"✓ Loaded {len(pairs)} pairs")
        
        # Tokenize
        print("\n[Step 2/4] Tokenizing {len(pairs)} pairs...")
        tokenized_data = []
        
        for idx, pair in enumerate(pairs):
            if (idx + 1) % 500 == 0:
                print(f"  Tokenized {idx + 1}/{len(pairs)}...")
            
            try:
                tokens = self.tokenize_pair(pair['input'], pair['response'])
                tokenized_data.append(tokens)
            except Exception as e:
                print(f"  ⚠ Warning: Failed to tokenize pair {idx}: {e}")
                continue
        
        print(f"✓ Successfully tokenized {len(tokenized_data)} pairs")
        
        # Split into train (80%) / temp (20%)
        print("\n[Step 3/4] Splitting: train(80%) / val(10%) / test(10%)...")
        train_data, temp_data = train_test_split(
            tokenized_data, test_size=0.2, random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42
        )
        
        print(f"  Train: {len(train_data)} samples")
        print(f"  Valid: {len(val_data)} samples")
        print(f"  Test:  {len(test_data)} samples")
        
        # Create HuggingFace Datasets
        print("\n[Step 4/4] Creating HuggingFace Datasets...")
        train_dataset = Dataset.from_dict({
            'input_ids': [d['input_ids'] for d in train_data],
            'attention_mask': [d['attention_mask'] for d in train_data]
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': [d['input_ids'] for d in val_data],
            'attention_mask': [d['attention_mask'] for d in val_data]
        })
        
        test_dataset = Dataset.from_dict({
            'input_ids': [d['input_ids'] for d in test_data],
            'attention_mask': [d['attention_mask'] for d in test_data]
        })
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        # Save to disk
        os.makedirs(output_dir, exist_ok=True)
        dataset_dict.save_to_disk(output_dir)
        
        print(f"✓ Saved to {output_dir}")
        print("\n" + "=" * 60)
        print("WEEK 1 TOKENIZATION COMPLETE!")
        print(f"Next: Use notebook/training_colab.ipynb for fine-tuning")
        print("=" * 60)
        
        return dataset_dict


if __name__ == "__main__":
    tokenizer = DialogueTokenizer()
    tokenizer.tokenize_dataset(
        "data/processed/cbt_pairs.json",
        "data/processed/tokenized"
    )
