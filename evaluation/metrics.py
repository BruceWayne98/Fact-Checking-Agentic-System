from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def apply_abstention_map(verdict: str, mode: str = "pessimistic") -> str:
    """
    Applies the paper's abstention mapping for 'Uncertain' verdicts:
    - Pessimistic: 'Uncertain' -> 'False' (assuming unverified/ambiguous claims fail verification)
    - Optimistic: 'Uncertain' -> 'True' (giving benefit of the doubt)
    """
    verdict = verdict.strip().capitalize()
    if verdict == "Uncertain":
        return "False" if mode.lower() == "pessimistic" else "True"
    return verdict

def compute_abstention_rate(predictions: List[str]) -> float:
    """Calculates the proportion of predictions that are 'Uncertain'."""
    if not predictions:
        return 0.0
    uncertain_count = sum(1 for p in predictions if p.strip().capitalize() == "Uncertain")
    return uncertain_count / len(predictions)

def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    abstention_mode: str = "pessimistic"
) -> Dict[str, Any]:
    """
    Computes Accuracy, Macro-F1, and abstention rate against ground truth binary/3-class labels.
    Handles 'Uncertain' predictions via pessimistic or optimistic mapping.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Lengths must match: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")

    raw_abstention_rate = compute_abstention_rate(y_pred)

    # Apply mapping for evaluation against ground truth
    mapped_preds = [apply_abstention_map(p, mode=abstention_mode) for p in y_pred]

    acc = accuracy_score(y_true, mapped_preds)
    macro_f1 = f1_score(y_true, mapped_preds, average="macro", zero_division=0)

    return {
        "mode": abstention_mode,
        "total_samples": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "abstention_rate": round(raw_abstention_rate, 4),
        "report": classification_report(y_true, mapped_preds, zero_division=0, output_dict=True)
    }

if __name__ == "__main__":
    # Sanity unit test
    ground_truth = ["True", "False", "True", "False", "True"]
    model_predictions = ["True", "False", "Uncertain", "True", "Uncertain"]

    print("--- Pessimistic Evaluation ---")
    pessimistic_res = evaluate_predictions(ground_truth, model_predictions, abstention_mode="pessimistic")
    print(f"Accuracy : {pessimistic_res['accuracy'] * 100:.1f}%")
    print(f"Macro-F1 : {pessimistic_res['macro_f1']:.4f}")
    print(f"Abstention Rate: {pessimistic_res['abstention_rate'] * 100:.1f}%")

    print("\n--- Optimistic Evaluation ---")
    optimistic_res = evaluate_predictions(ground_truth, model_predictions, abstention_mode="optimistic")
    print(f"Accuracy : {optimistic_res['accuracy'] * 100:.1f}%")
    print(f"Macro-F1 : {optimistic_res['macro_f1']:.4f}")
    print(f"Abstention Rate: {optimistic_res['abstention_rate'] * 100:.1f}%")
