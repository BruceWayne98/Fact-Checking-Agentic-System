import json
import sys
from pathlib import Path
from collections import Counter

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from datasets import load_dataset
from config import LIAR_DIR

# Original 6-class labels in the LIAR dataset
LIAR_6_CLASSES = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true"
]

# Mapping from 6 LIAR fine-grained categories to 3 TRUST Agent categories:
# - True: mostly-true, true
# - False: pants-fire, false, barely-true
# - Uncertain: half-true (borderline / split assertion)
MAPPING_6_TO_3 = {
    "true": "True",
    "mostly-true": "True",
    "half-true": "Uncertain",
    "barely-true": "False",
    "false": "False",
    "pants-fire": "False"
}

def map_label_6_to_3(label_name: str) -> str:
    """Map a 6-class LIAR label string into 3-class (True, False, Uncertain)."""
    return MAPPING_6_TO_3.get(label_name.lower().strip(), "Uncertain")

def load_liar_dataset():
    """Download and return the LIAR dataset splits from Hugging Face."""
    print("Loading LIAR dataset from Hugging Face (tasksource/liar)...")
    dataset = load_dataset("tasksource/liar")
    return dataset

def inspect_dataset(dataset):
    """Print split counts, label distributions, and sample statements."""
    label_feature = dataset["train"].features["label"]
    int2str = label_feature.int2str

    print("\n=== Dataset Splits ===")
    for split_name in dataset.keys():
        print(f"  {split_name}: {len(dataset[split_name])} examples")

    print("\n=== Train Split 6-Class Label Distribution ===")
    train_labels = [int2str(item["label"]) for item in dataset["train"]]
    counts = Counter(train_labels)
    for label, count in counts.most_common():
        pct = (count / len(train_labels)) * 100
        print(f"  {label:<15}: {count:>5} ({pct:.1f}%)")

    print("\n=== Train Split Mapped 3-Class Distribution ===")
    mapped_labels = [map_label_6_to_3(lbl) for lbl in train_labels]
    counts_3 = Counter(mapped_labels)
    for label, count in counts_3.most_common():
        pct = (count / len(mapped_labels)) * 100
        print(f"  {label:<15}: {count:>5} ({pct:.1f}%)")

    print("\n=== Sample Statements (5 items) ===")
    for i in range(5):
        sample = dataset["train"][i]
        lbl_str = int2str(sample["label"])
        lbl_3 = map_label_6_to_3(lbl_str)
        print(f"\n[Sample #{i+1}]")
        print(f"  Speaker : {sample.get('speaker', 'N/A')}")
        print(f"  Context : {sample.get('context', 'N/A')}")
        print(f"  Statement: \"{sample['statement']}\"")
        print(f"  Original Label: {lbl_str} -> Mapped (3-class): {lbl_3}")

def save_sample_subset(dataset, n_samples: int = 100):
    """Save a sample subset of test statements to JSONL for rapid offline experimentation."""
    LIAR_DIR.mkdir(parents=True, exist_ok=True)
    sample_file = LIAR_DIR / "sample.jsonl"
    
    test_split = dataset["test"]
    label_feature = test_split.features["label"]
    int2str = label_feature.int2str

    samples = []
    for i in range(min(n_samples, len(test_split))):
        item = test_split[i]
        raw_label = int2str(item["label"])
        samples.append({
            "id": item.get("id", str(i)),
            "statement": item["statement"],
            "speaker": item.get("speaker", ""),
            "context": item.get("context", ""),
            "label_6": raw_label,
            "label_3": map_label_6_to_3(raw_label)
        })

    with open(sample_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n[OK] Saved {len(samples)} test samples to {sample_file}")

if __name__ == "__main__":
    ds = load_liar_dataset()
    inspect_dataset(ds)
    save_sample_subset(ds, 100)
