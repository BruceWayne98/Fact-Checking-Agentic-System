import io
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import LIAR_DIR, EVAL_DIR
from graph.baseline_graph import build_baseline_graph
from evaluation.metrics import evaluate_predictions

def load_evaluation_samples(sample_file: Path, limit: int = 50) -> List[Dict[str, Any]]:
    """Loads benchmark test items from sample.jsonl."""
    samples = []
    if not sample_file.exists():
        print(f"Error: {sample_file} not found. Please run data/load_liar.py first.")
        return []

    with open(sample_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            if line.strip():
                samples.append(json.loads(line))
    return samples

def evaluate_baseline_on_benchmark(limit: int = 50):
    """
    Runs the 4-Agent Baseline pipeline over LIAR benchmark statements
    and computes Accuracy, Macro-F1, and Abstention Rate under both
    Pessimistic and Optimistic mappings.
    """
    sample_path = LIAR_DIR / "sample.jsonl"
    samples = load_evaluation_samples(sample_path, limit=limit)
    if not samples:
        return

    print("=" * 65)
    print(f"📊 Benchmarking TRUST Agents Baseline on {len(samples)} LIAR statements")
    print("=" * 65)

    app = build_baseline_graph()
    
    ground_truths = []
    predictions = []
    detailed_records = []

    start_time = time.time()

    for idx, item in enumerate(samples, 1):
        statement = item["statement"]
        label_3 = item["label_3"]      # "True", "False", or "Uncertain"
        label_6 = item["label_6"]

        print(f"\n[{idx}/{len(samples)}] Statement: \"{statement[:50]}...\" (GT: {label_3} / {label_6})")
        
        # Invoke LangGraph
        result = app.invoke({"article": statement})
        verifications = result.get("verifications", [])

        # Pipeline verdict aggregation for this statement:
        # If any sub-claim is False -> overall verdict is False
        # Else if all are True -> True
        # Else -> Uncertain
        if not verifications:
            predicted_verdict = "Uncertain"
            avg_conf = 0.0
        else:
            verdicts = [v.verdict for v in verifications]
            confidences = [v.confidence for v in verifications]
            avg_conf = round(sum(confidences) / len(confidences), 2)

            if any(v == "False" for v in verdicts):
                predicted_verdict = "False"
            elif all(v == "True" for v in verdicts):
                predicted_verdict = "True"
            else:
                predicted_verdict = "Uncertain"

        print(f"  -> Prediction: {predicted_verdict} (Avg Conf: {avg_conf * 100:.0f}%)")

        ground_truths.append(label_3)
        predictions.append(predicted_verdict)

        detailed_records.append({
            "id": item.get("id", str(idx)),
            "statement": statement,
            "speaker": item.get("speaker", ""),
            "label_6": label_6,
            "label_3": label_3,
            "predicted_verdict": predicted_verdict,
            "avg_confidence": avg_conf,
            "claims_extracted": [v.claim for v in verifications]
        })

    elapsed = time.time() - start_time
    print(f"\n[OK] Evaluated {len(samples)} statements in {elapsed:.1f}s ({elapsed/len(samples):.2f}s/stmt)")

    # Compute metrics
    # Note: Ground truth 'Uncertain' (half-true) can be matched or evaluated against binary projections
    # Filter ground truths for binary accuracy comparison (or keep all 3 classes)
    pessimistic_eval = evaluate_predictions(ground_truths, predictions, abstention_mode="pessimistic")
    optimistic_eval = evaluate_predictions(ground_truths, predictions, abstention_mode="optimistic")

    results_data = {
        "dataset": "LIAR Test Subset",
        "num_samples": len(samples),
        "elapsed_seconds": round(elapsed, 2),
        "abstention_rate": pessimistic_eval["abstention_rate"],
        "pessimistic": {
            "accuracy": pessimistic_eval["accuracy"],
            "macro_f1": pessimistic_eval["macro_f1"],
            "classification_report": pessimistic_eval["report"]
        },
        "optimistic": {
            "accuracy": optimistic_eval["accuracy"],
            "macro_f1": optimistic_eval["macro_f1"],
            "classification_report": optimistic_eval["report"]
        },
        "records": detailed_records
    }

    # Save to disk
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EVAL_DIR / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {out_file}")
    return results_data

if __name__ == "__main__":
    evaluate_baseline_on_benchmark(limit=50)
