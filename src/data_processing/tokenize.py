"""
WEEK 1: Data & Foundation (Part 2)
Goal: Tokenize cbt_pairs.json and prepare for model training.

Author Comment:
    Tokenize cbt_pairs.json using DialoGPT tokenizer.
    Apply padding and truncation to max_length=512.
    Split into train (80%), validation (10%), test (10%) sets.
    Save as HuggingFace Dataset to disk at data/processed/tokenized/
"""

import json
import os
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


class DialogueTokenizer:
    """Tokenize dialogue pairs for training."""

    def __init__(self, model_name="microsoft/DialoGPT-medium", max_length=512):
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        print(f"✓ Tokenizer loaded (vocab size: {len(self.tokenizer)})")

    def tokenize_pair(self, input_text, response_text):
        combined_text = f"{input_text}{self.tokenizer.eos_token}{response_text}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["input_ids"],
        }

    def tokenize_dataset(self, input_json, output_dir):
        print("\n" + "=" * 60)
        print("WEEK 1 TOKENIZATION PIPELINE")
        print("=" * 60)
        print(f"\n[Step 1/4] Loading CBT pairs from {input_json}...")

        with open(input_json, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        print(f"✓ Loaded {len(pairs)} pairs")
        print(f"\n[Step 2/4] Tokenizing {len(pairs)} pairs...")

        tokenized_data = []
        for idx, pair in enumerate(pairs):
            if (idx + 1) % 500 == 0:
                print(f"  Tokenized {idx + 1}/{len(pairs)}...")
            try:
                tokens = self.tokenize_pair(pair["input"], pair["response"])
                tokens["emotion"] = pair.get("emotion", "neutral")
                tokenized_data.append(tokens)
            except Exception as e:
                print(f"  ⚠ Warning: Failed to tokenize pair {idx}: {e}")

        print(f"✓ Successfully tokenized {len(tokenized_data)} pairs")

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

        print("\n[Step 4/4] Creating HuggingFace Datasets...")
        def build_dataset(data):
            return Dataset.from_dict({
                "input_ids": [item["input_ids"] for item in data],
                "attention_mask": [item["attention_mask"] for item in data],
                "labels": [item["labels"] for item in data],
                "emotion": [item["emotion"] for item in data],
            })

        dataset_dict = DatasetDict({
            "train": build_dataset(train_data),
            "validation": build_dataset(val_data),
            "test": build_dataset(test_data),
        })

        os.makedirs(output_dir, exist_ok=True)
        dataset_dict.save_to_disk(output_dir)

        print(f"✓ Saved tokenized dataset to {output_dir}")
        print("\n" + "=" * 60)
        print("WEEK 1 TOKENIZATION COMPLETE!")
        print("=" * 60)

        return dataset_dict


if __name__ == "__main__":
    tokenizer = DialogueTokenizer()
    tokenizer.tokenize_dataset(
        "data/processed/cbt_pairs.json",
        "data/processed/tokenized"
    )
