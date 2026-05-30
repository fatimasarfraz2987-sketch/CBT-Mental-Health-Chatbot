"""
WEEK 1: Data & Foundation
Goal: Download datasets, clean them, and produce cbt_pairs.json ready for training.

Author Comment:
    Load and merge Counsel Chat and EmpatheticDialogues into CBT-format pairs.
    Each pair should have: {"input": "user message", "response": "therapist response", "emotion": "label"}
    Filter out short responses (< 50 chars) and rows with missing data.
    Save final dataset to data/processed/cbt_pairs.json with count printed at the end.
"""

import json
import os
import re
from html import unescape

import pandas as pd
from datasets import load_dataset, load_from_disk

RAW_COUNSEL_PATH = os.path.join("data", "raw", "counsel_chat", "counselchat-data.csv")
EMPATHETIC_RAW_DIR = os.path.join("data", "raw", "empathetic_dialogues")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "cbt_pairs.json")

os.makedirs(PROCESSED_DIR, exist_ok=True)

EMPATHETIC_SOURCE = "empathetic_dialogues"
COUNSEL_SOURCE = "counselchat"
MIN_RESPONSE_LENGTH = 50


def strip_html(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def load_empathetic_dataset():
    print("\n[Step 1/3] Loading EmpatheticDialogues dataset...")
    try:
        dataset = load_dataset("empathetic_dialogues")
        print("✓ Loaded EmpatheticDialogues from HuggingFace")
        return dataset
    except Exception as exc:
        print(f"⚠ HuggingFace load failed: {exc}")
        if os.path.isdir(EMPATHETIC_RAW_DIR):
            print(f"✓ Loading EmpatheticDialogues from local cache: {EMPATHETIC_RAW_DIR}")
            return load_from_disk(EMPATHETIC_RAW_DIR)
        raise


def build_empathetic_pairs(dataset):
    pairs = []
    for split_name in dataset.keys():
        for example in dataset[split_name]:
            context = example.get("context", "")
            if isinstance(context, list):
                input_text = " ".join(str(turn).strip() for turn in context if turn)
            else:
                input_text = str(context or "").strip()

            response = str(example.get("utterance", "")).strip()
            emotion = str(example.get("prompt", "neutral")).strip().lower() or "neutral"

            if not input_text or not response:
                continue
            if len(response) < MIN_RESPONSE_LENGTH:
                continue

            pairs.append(
                {
                    "input": input_text,
                    "response": response,
                    "emotion": emotion,
                    "source": EMPATHETIC_SOURCE,
                    "split": split_name,
                }
            )

    print(f"✓ Built {len(pairs)} EmpatheticDialogues CBT pairs")
    return pairs


def build_counsel_pairs(path=RAW_COUNSEL_PATH):
    pairs = []
    print("\n[Step 2/3] Loading CounselChat dataset...")
    if not os.path.exists(path):
        print(f"⚠ CounselChat file not found at {path}")
        return pairs

    counsel_df = pd.read_csv(path)
    print(f"✓ Loaded {len(counsel_df)} CounselChat records")

    for _, row in counsel_df.iterrows():
        question = str(row.get("questionText", "") or "").strip()
        answer = strip_html(str(row.get("answerText", "") or "")).strip()
        topic = str(row.get("topics", "") or "").strip()

        if not question or not answer:
            continue
        if len(answer) < MIN_RESPONSE_LENGTH:
            continue

        pairs.append(
            {
                "input": question,
                "response": answer,
                "emotion": "neutral",
                "source": COUNSEL_SOURCE,
                "topic": topic or "general",
            }
        )

    print(f"✓ Built {len(pairs)} CounselChat CBT pairs")
    return pairs


def save_cbt_pairs(pairs):
    print(f"\n[Step 3/3] Saving {len(pairs)} CBT pairs to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(pairs)} CBT pairs")


if __name__ == "__main__":
    cbt_pairs = []

    try:
        empathetic_dataset = load_empathetic_dataset()
        cbt_pairs.extend(build_empathetic_pairs(empathetic_dataset))
    except Exception as exc:
        print(f"⚠ Could not load EmpatheticDialogues: {exc}")

    cbt_pairs.extend(build_counsel_pairs())

    if not cbt_pairs:
        raise RuntimeError("No CBT pairs available. Please check raw dataset files.")

    save_cbt_pairs(cbt_pairs)

    print("\n" + "=" * 60)
    print(f"WEEK 1 COMPLETE: {len(cbt_pairs)} CBT pairs ready for training")
    print("Next: Run src/data_processing/tokenize.py to prepare model inputs")
    print("=" * 60)
