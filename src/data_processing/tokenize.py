"""
Tokenization pipeline for dialogue datasets.
Converts raw dialogues into tokenized sequences ready for model training.
"""
import json
import os
from pathlib import Path
from transformers import AutoTokenizer


class DialogueTokenizer:
    """Tokenize dialogue pairs for training."""
    
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        """
        Initialize tokenizer with specified model.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def tokenize_pair(self, question, answer):
        """
        Tokenize a question-answer pair.
        
        Args:
            question: User question text
            answer: Therapist response text
            
        Returns:
            Tokenized inputs ready for model
        """
        encoding = self.tokenizer(
            question,
            answer,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return encoding
    
    def tokenize_dataset(self, input_json, output_dir):
        """
        Tokenize entire dataset JSON file.
        
        Args:
            input_json: Path to cbt_pairs.json
            output_dir: Directory to save tokenized output
        """
        os.makedirs(output_dir, exist_ok=True)
        
        with open(input_json, 'r') as f:
            pairs = json.load(f)
        
        tokenized = []
        for pair in pairs:
            tokens = self.tokenize_pair(pair['question'], pair['answer'])
            tokenized.append({
                'input_ids': tokens['input_ids'][0].tolist(),
                'attention_mask': tokens['attention_mask'][0].tolist(),
                'token_type_ids': tokens['token_type_ids'][0].tolist() if 'token_type_ids' in tokens else []
            })
        
        output_path = os.path.join(output_dir, 'tokenized_pairs.json')
        with open(output_path, 'w') as f:
            json.dump(tokenized, f, indent=2)
        
        print(f"Tokenized {len(tokenized)} pairs. Saved to {output_path}")
        return tokenized


if __name__ == "__main__":
    tokenizer = DialogueTokenizer()
    tokenizer.tokenize_dataset(
        "data/processed/cbt_pairs.json",
        "data/processed"
    )
