import os
import tarfile
import tempfile
import urllib.request
import csv
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

output_dir = "data/raw/empathetic_dialogues/"
os.makedirs(output_dir, exist_ok=True)

print("Loading EmpatheticDialogues via datasets library...")
try:
    dataset = load_dataset("empathetic_dialogues")
except Exception as e:
    print("Direct load_dataset('empathetic_dialogues') failed:", e)
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
print("EmpatheticDialogues saved. Splits:", dataset.keys())

# Load counsel_chat CSV and extract question-answer pairs as CBT dialogue
# Filter only rows where the therapist's response is longer than 50 characters
# Save the cleaned pairs to data/processed/cbt_pairs.json
counsel_path = os.path.join("data", "raw", "counsel_chat", "counselchat-data.csv")
processed_dir = os.path.join("data", "processed")
os.makedirs(processed_dir, exist_ok=True)

try:
    counsel_df = pd.read_csv(counsel_path)
except FileNotFoundError:
    print(f"CounselChat file not found at {counsel_path}, skipping CBT pair extraction.")
else:
    cbt_pairs = []
    for _, row in counsel_df.iterrows():
        question = str(row.get("questionText", "")).strip()
        answer = str(row.get("answerText", "")).strip()
        if question and len(answer) > 50:
            cbt_pairs.append({"question": question, "answer": answer})

    output_path = os.path.join(processed_dir, "cbt_pairs.json")
    pd.DataFrame(cbt_pairs).to_json(output_path, orient="records", indent=2, force_ascii=False)
    print(f"Saved {len(cbt_pairs)} CBT pairs to {output_path}")