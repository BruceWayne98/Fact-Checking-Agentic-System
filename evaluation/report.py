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
    main_results_path = EVAL_DIR / "results.json"
    baseline_path = EVAL_DIR / "results_baseline.json"
    extended_path = EVAL_DIR / "results_extended.json"
    ablation_path = EVAL_DIR / "ablation_results.json"

    print("\n" + "=" * 80)
    print("📈 TRUST AGENTS BENCHMARK EVALUATION & REPRODUCTION REPORT")
    print("=" * 80)

    # Load available result files
    baseline_data = None
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
    elif main_results_path.exists():
        with open(main_results_path, "r", encoding="utf-8") as f:
            main_data = json.load(f)
            if main_data.get("pipeline_mode") == "baseline":
                baseline_data = main_data

    extended_data = None
    if extended_path.exists():
        with open(extended_path, "r", encoding="utf-8") as f:
            extended_data = json.load(f)
    elif main_results_path.exists():
        with open(main_results_path, "r", encoding="utf-8") as f:
            main_data = json.load(f)
            if main_data.get("pipeline_mode") == "extended":
                extended_data = main_data

    # Print Main Comparison Table
    print(f"{'Framework / Configuration':<40} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Abstention':<10} | {'Explainable?'}")
    print("-" * 88)

    if extended_data:
        p = extended_data["pessimistic"]
        o = extended_data["optimistic"]
        ar = extended_data.get("abstention_rate", 0.0)
        print(f"{'TRUST Extended (Pessimistic)':<40} | {p['accuracy']*100:>8.1f}% | {p['macro_f1']:>10.4f} | {ar*100:>9.1f}% | Yes (Cited)")
        print(f"{'TRUST Extended (Optimistic)':<40}  | {o['accuracy']*100:>8.1f}% | {o['macro_f1']:>10.4f} | {ar*100:>9.1f}% | Yes (Cited)")

    if baseline_data:
        p = baseline_data["pessimistic"]
        o = baseline_data["optimistic"]
        ar = baseline_data.get("abstention_rate", 0.0)
        print(f"{'TRUST Baseline (Pessimistic)':<40} | {p['accuracy']*100:>8.1f}% | {p['macro_f1']:>10.4f} | {ar*100:>9.1f}% | Yes (Cited)")
        print(f"{'TRUST Baseline (Optimistic)':<40}  | {o['accuracy']*100:>8.1f}% | {o['macro_f1']:>10.4f} | {ar*100:>9.1f}% | Yes (Cited)")

    # Baselines reported in the TRUST Agents paper (arXiv:2604.12184)
    print(f"{'Fine-Tuned RoBERTa (Paper Reported)':<40} | {'~30-33%':>9} | {'~0.28-0.31':>10} | {'0.0%':>10} | No (Black-box)")
    print(f"{'Fine-Tuned BERT (Paper Reported)':<40}    | {'~27-30%':>9} | {'~0.25-0.28':>10} | {'0.0%':>10} | No (Black-box)")
    print(f"{'Zero-Shot LLM (Paper Reported)':<40}      | {'~22-26%':>9} | {'~0.20-0.24':>10} | {'0.0%':>10} | No (Hallucination)")
    print("-" * 88)

    # Print Ablation Table if available
    if ablation_path.exists():
        with open(ablation_path, "r", encoding="utf-8") as f:
            ablation_list = json.load(f)
        print("\n" + "=" * 80)
        print("🔬 SYSTEM COMPONENT ABLATION BREAKDOWN")
        print("=" * 80)
        print(f"{'Ablation Variant':<30} | {'Pess. Acc':<10} | {'Pess. F1':<10} | {'Opt. Acc':<10} | {'Opt. F1':<10}")
        print("-" * 80)
        for ab in ablation_list:
            row = (
                f"{ab['name']:<30} | "
                f"{ab['pessimistic_acc']*100:>8.1f}% | "
                f"{ab['pessimistic_f1']:>10.4f} | "
                f"{ab['optimistic_acc']*100:>8.1f}% | "
                f"{ab['optimistic_f1']:>10.4f}"
            )
            print(row)
        print("-" * 80)

    print("\n💡 Key Architectural Findings:")
    print("1. Safe Abstention vs Hallucinated Verdicts:")
    print("   When retrieval evidence is sparse, TRUST Agents triggers its calibrated abstention protocol")
    print("   rather than guessing or fabricating justifications.")
    print("2. Multi-Agent Delphi Jury Diversification:")
    print("   Cross-examining through Data Analyst, Historian, and Policy Analyst lenses reduces monolithic LLM bias.")
    print("3. Neurosymbolic Formal Logic Guarantees:")
    print("   Compound propositions are aggregated via formal boolean logic gates (AND/OR/IMPLIES),")
    print("   guaranteeing deterministic consistency without neural hallucination.")
    print("=" * 80)


if __name__ == "__main__":
    print_evaluation_report()
