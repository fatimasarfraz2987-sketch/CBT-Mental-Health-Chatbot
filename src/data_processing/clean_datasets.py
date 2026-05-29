"""
WEEK 1: Data & Foundation
Goal: Download datasets, clean them, and produce cbt_pairs.json ready for training.

Author Comment:
    Load and merge Counsel Chat and EmpatheticDialogues into CBT-format pairs.
    Each pair should have: {"input": "user message", "response": "therapist response", "emotion": "label"}
    Filter out short responses (< 50 chars) and rows with missing data.
    Save final dataset to data/processed/cbt_pairs.json with count printed at the end.
    
    Let Copilot generate the implementation — review each suggested block before accepting.
"""
import os
import tarfile
import tempfile
import urllib.request
import csv
import pandas as pd
import json
from datasets import Dataset, DatasetDict, load_dataset

output_dir = "data/raw/empathetic_dialogues/"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("WEEK 1: Data Processing & Foundation")
print("=" * 60)
print("\n[Step 1/3] Loading EmpatheticDialogues dataset...")
try:
    dataset = load_dataset("empathetic_dialogues")
except Exception as e:
    print(f"Direct load_dataset('empathetic_dialogues') failed: {e}")
    print("Falling back to the original ParlAI archive URL.")
    archive_url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "empatheticdialogues.tar.gz")
        print(f"Downloading archive from {archive_url}...")
        urllib.request.urlretrieve(archive_url, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            split_files = {
                "train": "empatheticdialogues/train.csv",
                "valid": "empatheticdialogues/valid.csv",
                "test": "empatheticdialogues/test.csv",
            }
            dataset = DatasetDict()
            for split, path in split_files.items():
                with tar.extractfile(path) as fh:
                    reader = csv.DictReader(line.decode("utf-8") for line in fh)
                    rows = []
                    for row in reader:
                        rows.append(
                            {
                                "utterance": row["utterance"],
                                "utterance_idx": int(row["utterance_idx"]),
                                "context": row["context"],
                                "prompt": row["prompt"],
                                "speaker_idx": int(row["speaker_idx"]),
                                "conv_id": row["conv_id"],
                                "selfeval": row.get("selfeval", "") or "",
                                "tags": row.get("tags", "") or "",
                            }
                        )
                dataset[split] = Dataset.from_pandas(pd.DataFrame(rows))

    print("Download and extraction complete.")

dataset.save_to_disk(output_dir)
print("✓ EmpatheticDialogues saved to", output_dir)

# ============================================================================
# [Step 2/3] Load Counsel Chat and convert to CBT format
# ============================================================================
print("\n[Step 2/3] Processing CounselChat dataset...")
counsel_path = os.path.join("data", "raw", "counsel_chat", "counselchat-data.csv")
processed_dir = os.path.join("data", "processed")
os.makedirs(processed_dir, exist_ok=True)

cbt_pairs = []

try:
    counsel_df = pd.read_csv(counsel_path)
    print(f"✓ Loaded {len(counsel_df)} records from CounselChat")
    
    for idx, row in counsel_df.iterrows():
        question = str(row.get("questionText", "")).strip()
        answer = str(row.get("answerText", "")).strip()
        topic = str(row.get("topics", "")).strip()
        
        # Filter: ensure both question and answer exist
        if not question or not answer:
            continue
        
        # Filter: answer must be at least 50 characters (meaningful response)
        if len(answer) < 50:
            continue
        
        # Create CBT-format pair
        cbt_pair = {
            "input": question,
            "response": answer,
            "emotion": "neutral",  # Will be labeled later by sentiment module
            "topic": topic if topic else "general",
            "source": "counselchat"
        }
        cbt_pairs.append(cbt_pair)
    
    print(f"✓ After filtering: {len(cbt_pairs)} valid CBT pairs")

except FileNotFoundError:
    print(f"⚠ CounselChat file not found at {counsel_path}")
    print("  Make sure counselchat-data.csv is in data/raw/counsel_chat/")

# ============================================================================
# [Step 3/3] Save CBT pairs to JSON
# ============================================================================
print("\n[Step 3/3] Saving cleaned CBT pairs...")

output_path = os.path.join(processed_dir, "cbt_pairs.json")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(cbt_pairs, f, indent=2, ensure_ascii=False)

print(f"✓ Saved {len(cbt_pairs)} CBT pairs to {output_path}")
print("\n" + "=" * 60)
print(f"WEEK 1 COMPLETE: {len(cbt_pairs)} training pairs ready!")
print(f"Next: Run tokenize.py to prepare for model training")
print("=" * 60)