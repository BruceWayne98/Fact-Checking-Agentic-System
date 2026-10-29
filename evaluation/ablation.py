import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 unbuffered output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import LIAR_DIR, EVAL_DIR
from graph.extended_graph import build_extended_graph
from evaluation.metrics import evaluate_predictions
from evaluation.run_eval import load_evaluation_samples


ABLATION_CONFIGS = [
    {
        "name": "Full TRUST Extended",
        "description": "Decomposer + Delphi Jury (3 Personas) + Logic Aggregator",
        "enable_decomposer": True,
        "enable_jury": True,
        "enable_logic_aggregator": True
    },
    {
        "name": "w/o Delphi Jury",
        "description": "Decomposer + Single Verifier + Logic Aggregator",
        "enable_decomposer": True,
        "enable_jury": False,
        "enable_logic_aggregator": True
    },
    {
        "name": "w/o Decomposer",
        "description": "No Claim Decomposition + Delphi Jury + Verifier Consensus",
        "enable_decomposer": False,
        "enable_jury": True,
        "enable_logic_aggregator": False
    },
    {
        "name": "w/o Logic Aggregator",
        "description": "Decomposer + Delphi Jury + Raw Majority Verifier (No Propositional Logic)",
        "enable_decomposer": True,
        "enable_jury": True,
        "enable_logic_aggregator": False
    },
    {
        "name": "TRUST Baseline (4 Agents)",
        "description": "Extractor -> Hybrid Retrieval -> Single Verifier -> Explainer",
        "enable_decomposer": False,
        "enable_jury": False,
        "enable_logic_aggregator": False
    }
]


def run_single_ablation(
    config: Dict[str, Any],
    samples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    name = config["name"]
    print(f"\n" + "-" * 60)
    print(f"🔬 Running Ablation Variant: {name}")
    print(f"   Description: {config['description']}")
    print("-" * 60)

    app = build_extended_graph(
        enable_decomposer=config["enable_decomposer"],
        enable_jury=config["enable_jury"],
        enable_logic_aggregator=config["enable_logic_aggregator"]
    )

    ground_truths = []
    predictions = []
    start_time = time.time()

    for idx, item in enumerate(samples, 1):
        statement = item["statement"]
        label_3 = item["label_3"]

        result = app.invoke({"article": statement})
        verifications = result.get("verifications", [])

        logic_results = [v for v in verifications if v.logic_trace]
        if logic_results:
            predicted_verdict = logic_results[0].verdict
        elif not verifications:
            predicted_verdict = "Uncertain"
        else:
            verdicts = [v.verdict for v in verifications]
            if any(v == "False" for v in verdicts):
                predicted_verdict = "False"
            elif all(v == "True" for v in verdicts):
                predicted_verdict = "True"
            else:
                predicted_verdict = "Uncertain"

        ground_truths.append(label_3)
        predictions.append(predicted_verdict)
        sys.stdout.write(f"\r  Progress: [{idx}/{len(samples)}] -> {predicted_verdict:<10}")
        sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/len(samples):.2f}s/stmt)")

    pessimistic_eval = evaluate_predictions(ground_truths, predictions, abstention_mode="pessimistic")
    optimistic_eval = evaluate_predictions(ground_truths, predictions, abstention_mode="optimistic")

    return {
        "name": name,
        "description": config["description"],
        "num_samples": len(samples),
        "elapsed_seconds": round(elapsed, 2),
        "latency_per_stmt": round(elapsed / len(samples), 3),
        "abstention_rate": pessimistic_eval["abstention_rate"],
        "pessimistic_acc": pessimistic_eval["accuracy"],
        "pessimistic_f1": pessimistic_eval["macro_f1"],
        "optimistic_acc": optimistic_eval["accuracy"],
        "optimistic_f1": optimistic_eval["macro_f1"]
    }


def run_ablation_study(limit: int = 20) -> List[Dict[str, Any]]:
    sample_path = LIAR_DIR / "sample.jsonl"
    samples = load_evaluation_samples(sample_path, limit=limit)
    if not samples:
        print("No evaluation samples found.")
        return []

    print("=" * 75)
    print(f"🧪 Starting TRUST Agents Component Ablation Study on {len(samples)} Statements")
    print("=" * 75)

    study_results = []
    for cfg in ABLATION_CONFIGS:
        res = run_single_ablation(cfg, samples)
        study_results.append(res)

    # Print summary Markdown table
    print("\n" + "=" * 85)
    print("📊 ABLATION STUDY COMPARATIVE RESULTS")
    print("=" * 85)
    header = f"{'Configuration':<26} | {'Pess. Acc':<10} | {'Pess. F1':<10} | {'Opt. Acc':<10} | {'Opt. F1':<10} | {'Abstention':<10}"
    print(header)
    print("-" * 85)
    for r in study_results:
        row = (
            f"{r['name']:<26} | "
            f"{r['pessimistic_acc']*100:>8.1f}% | "
            f"{r['pessimistic_f1']:>10.4f} | "
            f"{r['optimistic_acc']*100:>8.1f}% | "
            f"{r['optimistic_f1']:>10.4f} | "
            f"{r['abstention_rate']*100:>9.1f}%"
        )
        print(row)
    print("-" * 85)

    # Save to disk
    out_file = EVAL_DIR / "ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(study_results, f, indent=2)

    print(f"\nAblation study results saved to: {out_file}")
    return study_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run component ablation study on TRUST Agents.")
    parser.add_argument("--limit", type=int, default=20, help="Number of benchmark statements per ablation variant")
    args = parser.parse_args()

    run_ablation_study(limit=args.limit)
