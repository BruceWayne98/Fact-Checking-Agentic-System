import io
import json
import sys
from pathlib import Path

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import EVAL_DIR

def print_evaluation_report():
    results_path = EVAL_DIR / "results.json"
    if not results_path.exists():
        print(f"Error: {results_path} not found. Please run evaluation/run_eval.py first.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pessimistic = data["pessimistic"]
    optimistic = data["optimistic"]
    abstention_rate = data["abstention_rate"]
    n_samples = data["num_samples"]

    print("\n" + "=" * 70)
    print("📈 TRUST AGENTS BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Dataset              : LIAR Test Split ({n_samples} statements)")
    print(f"System Abstention Rate: {abstention_rate * 100:.1f}%")
    print("-" * 70)
    print(f"{'Framework / Configuration':<36} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Explainable?'}")
    print("-" * 70)
    print(f"{'TRUST Baseline (Pessimistic)':<36} | {pessimistic['accuracy']*100:>6.1f}%   | {pessimistic['macro_f1']:>8.4f}   | Yes (Cited)")
    print(f"{'TRUST Baseline (Optimistic)':<36}  | {optimistic['accuracy']*100:>6.1f}%   | {optimistic['macro_f1']:>8.4f}   | Yes (Cited)")
    print(f"{'Fine-Tuned BERT (Paper Reported)':<36} | {'~27-30%':>6}   | {'~0.25-0.28':>8}   | No (Black-box)")
    print(f"{'Fine-Tuned RoBERTa (Paper)':<36}      | {'~30-33%':>6}   | {'~0.28-0.31':>8}   | No (Black-box)")
    print(f"{'Zero-Shot LLM (Paper Reported)':<36}   | {'~22-26%':>6}   | {'~0.20-0.24':>8}   | No (Hallucination)")
    print("-" * 70)
    print("\n💡 Key Insights from Evaluation:")
    print("1. Abstention Mechanism: Unlike black-box classifiers that guess blindly on ambiguous statements,")
    print(f"   TRUST Agents abstains with an 'Uncertain' verdict on {abstention_rate*100:.1f}% of statements when external")
    print("   retrieval lacks authoritative corroboration.")
    print("2. Interpretability Trade-Off: Every single verdict produced by TRUST Agents includes human-readable")
    print("   reasoning and explicit source citations from the retrieval corpus.")
    print("=" * 70)

if __name__ == "__main__":
    print_evaluation_report()
