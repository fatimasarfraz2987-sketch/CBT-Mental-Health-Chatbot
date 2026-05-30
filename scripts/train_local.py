"""
Local training script based on notebooks/training_colab.ipynb cells.
Runs a short smoke-training pass (1 epoch) on a small subset.

Usage:
  python scripts/train_local.py --data_dir data/processed/tokenized --output_dir models/fine_tuned --epochs 1 --subset 128
"""
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load tokenized dataset
    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Tokenized dataset not found at {args.data_dir}. Run src/data_processing/tokenize.py first.")

    dataset = load_from_disk(args.data_dir)
    print(f"Loaded dataset: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")

    # Optionally subset for a smoke run
    if args.subset and args.subset > 0:
        train_ds = dataset['train'].select(range(min(args.subset, len(dataset['train']))))
        val_ds = dataset['validation'].select(range(min(int(args.subset/10)+1, len(dataset['validation']))))
    else:
        train_ds = dataset['train']
        val_ds = dataset['validation']

    # Load model & tokenizer
    model_name = args.model_name
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Apply LoRA (PEFT)
    print("Applying LoRA adapters (PEFT)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Training args
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print("Starting training (smoke test)...")
    train_result = trainer.train()

    print("Saving final checkpoint...")
    trainer.save_model(os.path.join(args.output_dir, 'checkpoint-final'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'checkpoint-final'))

    print("Done. Training stats:")
    print(train_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed/tokenized')
    parser.add_argument('--output_dir', type=str, default='models/fine_tuned')
    parser.add_argument('--model_name', type=str, default='microsoft/DialoGPT-medium')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--subset', type=int, default=128, help='number of train samples for smoke run')

    args = parser.parse_args()
    main(args)
