import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from graph.state import VerificationResult


def detect_logical_structure(compound_claim: str) -> str:
    """
    Infers the formal propositional logic operator from discourse markers in a compound claim.
    Returns: "AND", "OR", or "IMPLIES".
    
    Default for complex factual news assertions is "AND" (conjunction of all bundled propositions).
    """
    claim_lower = compound_claim.lower().strip()

    # Disjunction markers (OR)
    if re.search(r"\b(either\b.*\bor\b|\bor\b|unless)\b", claim_lower):
        return "OR"

    # Implication / Causal markers (IMPLIES)
    if re.search(r"\bif\b.*?(?:then|,|\bleading to\b)|implies that|therefore|entails that", claim_lower):
        return "IMPLIES"

    # Conjunction markers (AND) - default for compound political and news claims
    return "AND"


def aggregate_verdicts(
    atomic_verdicts: List[Union[Tuple[str, str, float], VerificationResult, Dict[str, Any]]],
    operator: str = "AND",
    compound_claim: Optional[str] = None
) -> VerificationResult:
    """
    Deterministic symbolic logic aggregator for multi-agent atomic claim verifications.
    
    Implements formal propositional logic over three-valued verdicts (True, False, Uncertain):
    - AND (∧): Compound statement is True ONLY IF all atomic propositions are True.
               If ANY atomic proposition is False, the compound statement is False.
               If no proposition is False but some are Uncertain, the compound statement is Uncertain.
    - OR (∨): Compound statement is True IF ANY atomic proposition is True.
              If ALL atomic propositions are False, the compound statement is False.
              Otherwise Uncertain.
    - IMPLIES (→): Premise -> Conclusion. True if Premise is False or Conclusion is True.
                   False if Premise is True and Conclusion is False.
    
    Returns a unified VerificationResult with a deterministic logic trace and calibrated confidence.
    """
    if not atomic_verdicts:
        return VerificationResult(
            claim=compound_claim or "Empty Compound Claim",
            verdict="Uncertain",
            confidence=0.0,
            reasoning="No atomic sub-claims provided for symbolic aggregation.",
            logic_trace="No sub-claims to evaluate."
        )

    # Standardize input list into (claim_str, verdict_str, conf_float)
    normalized: List[Tuple[str, str, float, Any]] = []
    for item in atomic_verdicts:
        if isinstance(item, VerificationResult):
            normalized.append((item.claim, item.verdict.capitalize(), float(item.confidence), item))
        elif isinstance(item, dict):
            normalized.append((item.get("claim", ""), item.get("verdict", "Uncertain").capitalize(), float(item.get("confidence", 0.5)), item))
        elif isinstance(item, (tuple, list)):
            claim_text = item[0]
            verdict = str(item[1]).capitalize()
            conf = float(item[2])
            normalized.append((claim_text, verdict, conf, None))

    target_claim = compound_claim or " / ".join(c for c, _, _, _ in normalized)
    op = operator.upper().strip()

    # --- 1. AND OPERATOR (Conjunction: p1 ∧ p2 ∧ ... ∧ pn) ---
    if op == "AND":
        # Check for any False sub-claim (fails the entire conjunction)
        failing_subs = [(idx + 1, c, v, conf) for idx, (c, v, conf, _) in enumerate(normalized) if v == "False"]
        if failing_subs:
            # First failed sub-claim determines the primary failure
            first_fail_idx, first_fail_claim, _, fail_conf = failing_subs[0]
            logic_trace = f"Sub-claim {first_fail_idx} failed (AND operator)"
            reasoning = (
                f"Symbolic aggregation under AND operator: Compound assertion is refuted because "
                f"atomic sub-claim {first_fail_idx} (\"{first_fail_claim}\") was verified as False."
            )
            return VerificationResult(
                claim=target_claim,
                verdict="False",
                confidence=round(fail_conf, 4),
                reasoning=reasoning,
                logic_trace=logic_trace
            )

        # Check for any Uncertain sub-claim (prevents complete verification)
        uncertain_subs = [(idx + 1, c, v, conf) for idx, (c, v, conf, _) in enumerate(normalized) if v == "Uncertain"]
        if uncertain_subs:
            first_unc_idx, first_unc_claim, _, unc_conf = uncertain_subs[0]
            logic_trace = f"Sub-claim {first_unc_idx} is Uncertain (AND operator)"
            reasoning = (
                f"Symbolic aggregation under AND operator: Cannot confirm compound assertion because "
                f"atomic sub-claim {first_unc_idx} (\"{first_unc_claim}\") is Uncertain."
            )
            return VerificationResult(
                claim=target_claim,
                verdict="Uncertain",
                confidence=round(unc_conf, 4),
                reasoning=reasoning,
                logic_trace=logic_trace
            )

        # All sub-claims are True!
        min_conf = min(conf for _, _, conf, _ in normalized)
        logic_trace = f"All {len(normalized)} sub-claims verified True (AND operator)"
        reasoning = (
            f"Symbolic aggregation under AND operator: All {len(normalized)} atomic propositions "
            f"were independently verified True. Minimum constituent confidence: {int(min_conf * 100)}%."
        )
        return VerificationResult(
            claim=target_claim,
            verdict="True",
            confidence=round(min_conf, 4),
            reasoning=reasoning,
            logic_trace=logic_trace
        )

    # --- 2. OR OPERATOR (Disjunction: p1 ∨ p2 ∨ ... ∨ pn) ---
    elif op == "OR":
        # Check for any True sub-claim (satisfies the disjunction)
        true_subs = [(idx + 1, c, v, conf) for idx, (c, v, conf, _) in enumerate(normalized) if v == "True"]
        if true_subs:
            first_true_idx, first_true_claim, _, true_conf = true_subs[0]
            max_conf = max(conf for _, _, _, conf in true_subs)
            logic_trace = f"Sub-claim {first_true_idx} verified True (OR operator)"
            reasoning = (
                f"Symbolic aggregation under OR operator: Compound assertion is verified because "
                f"atomic sub-claim {first_true_idx} (\"{first_true_claim}\") is True."
            )
            return VerificationResult(
                claim=target_claim,
                verdict="True",
                confidence=round(max_conf, 4),
                reasoning=reasoning,
                logic_trace=logic_trace
            )

        # If all sub-claims are False
        if all(v == "False" for _, v, _, _ in normalized):
            avg_conf = sum(conf for _, _, conf, _ in normalized) / len(normalized)
            logic_trace = f"All {len(normalized)} sub-claims failed (OR operator)"
            reasoning = f"Symbolic aggregation under OR operator: All constituent options were refuted."
            return VerificationResult(
                claim=target_claim,
                verdict="False",
                confidence=round(avg_conf, 4),
                reasoning=reasoning,
                logic_trace=logic_trace
            )

        # Otherwise Uncertain
        logic_trace = "Disjunction inconclusive with uncertain constituent claims"
        return VerificationResult(
            claim=target_claim,
            verdict="Uncertain",
            confidence=0.40,
            reasoning="Constituent sub-claims yield inconclusive disjunction.",
            logic_trace=logic_trace
        )

    # --- 3. IMPLIES OPERATOR (p1 → p2) ---
    elif op == "IMPLIES" and len(normalized) >= 2:
        premise_claim, premise_v, premise_conf, _ = normalized[0]
        conclusion_claim, conclusion_v, conclusion_conf, _ = normalized[1]

        if premise_v == "True" and conclusion_v == "False":
            logic_trace = "Premise True but Conclusion False (IMPLIES violation)"
            return VerificationResult(
                claim=target_claim,
                verdict="False",
                confidence=round(min(premise_conf, conclusion_conf), 4),
                reasoning=f"Implication refuted: Premise \"{premise_claim}\" is True, but Conclusion \"{conclusion_claim}\" is False.",
                logic_trace=logic_trace
            )
        elif premise_v == "False":
            logic_trace = "Vacuously True (Premise is False)"
            return VerificationResult(
                claim=target_claim,
                verdict="True",
                confidence=round(premise_conf, 4),
                reasoning="Implication is vacuously True because the antecedent premise was refuted.",
                logic_trace=logic_trace
            )
        elif premise_v == "True" and conclusion_v == "True":
            logic_trace = "Both Premise and Conclusion verified True"
            return VerificationResult(
                claim=target_claim,
                verdict="True",
                confidence=round(min(premise_conf, conclusion_conf), 4),
                reasoning="Implication holds: both antecedent and consequent are verified True.",
                logic_trace=logic_trace
            )
        else:
            logic_trace = "Implication components are Uncertain"
            return VerificationResult(
                claim=target_claim,
                verdict="Uncertain",
                confidence=0.40,
                reasoning="Implication components are unconfirmed.",
                logic_trace=logic_trace
            )

    # Fallback to AND logic
    return aggregate_verdicts(atomic_verdicts, operator="AND", compound_claim=compound_claim)


if __name__ == "__main__":
    print("=== Testing Agent: Logic Aggregator (Phase 10) ===")

    # Checkpoint Test from ROADMAP.md
    checkpoint_atomic_verdicts = [
        ("Biden passed infrastructure bill in 2021", "True", 0.92),
        ("It created 2M jobs", "False", 0.71),
        ("It cut carbon by 15%", "Uncertain", 0.38)
    ]
    checkpoint_operator = "AND"

    result = aggregate_verdicts(checkpoint_atomic_verdicts, operator=checkpoint_operator)
    print("\nCheckpoint Evaluation:")
    print(f"  Verdict:     {result.verdict}")
    print(f"  Confidence:  {result.confidence}")
    print(f"  Logic Trace: {result.logic_trace}")
    print(f"  Reasoning:   {result.reasoning}")

    # Verify checkpoint conditions
    assert result.verdict == "False", f"Expected 'False', got {result.verdict}"
    assert result.confidence == 0.71, f"Expected 0.71, got {result.confidence}"
    assert "Sub-claim 2 failed (AND operator)" in result.logic_trace, f"Unexpected trace: {result.logic_trace}"
    print("\n[OK] Phase 10 Checkpoint passed successfully!")

    # Test Disjunction (OR)
    or_verdicts = [
        ("The unemployment rate fell to 3.5% in January 2024", "True", 0.85),
        ("Unemployment reached 42%", "False", 0.90)
    ]
    or_result = aggregate_verdicts(or_verdicts, operator="OR")
    print("\nOR Aggregation Test:")
    print(f"  Verdict:     {or_result.verdict}")
    print(f"  Confidence:  {or_result.confidence}")
    print(f"  Logic Trace: {or_result.logic_trace}")

    # Test Operator Detection
    print("\nOperator Detection Tests:")
    test_phrases = [
        ("Biden passed the bill in 2021 and created jobs", "AND"),
        ("Either taxes were raised or spending was cut", "OR"),
        ("If interest rates rise, inflation will fall", "IMPLIES")
    ]
    for phrase, expected in test_phrases:
        detected = detect_logical_structure(phrase)
        print(f"  \"{phrase}\" -> Detected: {detected} (Expected: {expected})")
