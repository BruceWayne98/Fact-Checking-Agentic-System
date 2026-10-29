import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import ollama
except ImportError:
    ollama = None

from config import PROMPTS_DIR, DEFAULT_MODEL, OLLAMA_BASE_URL, UNCERTAIN_CONFIDENCE_THRESHOLD
from graph.state import VerificationResult

def load_prompt_template() -> str:
    prompt_file = PROMPTS_DIR / "verifier.txt"
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Evaluate claim against evidence. Output JSON with verdict (True/False/Uncertain) and confidence."

def build_verifier_prompt(claim: str, evidence_passages: List[Dict[str, Any]]) -> str:
    """Formats the claim and numbered evidence passages for verification."""
    if not evidence_passages:
        evidence_str = "No external evidence passages found."
    else:
        passages = []
        for idx, ev in enumerate(evidence_passages, 1):
            doc_id = ev.get("id", f"doc_{idx}")
            src = ev.get("source", "Unknown")
            txt = ev.get("text", "")
            passages.append(f"[{doc_id}] (Source: {src}):\n\"{txt}\"")
        evidence_str = "\n\n".join(passages)

    user_prompt = f"""CLAIM TO VERIFY:
"{claim}"

RETRIEVED EVIDENCE PASSAGES:
{evidence_str}

Evaluate the claim and output the JSON result:"""
    return user_prompt

def rule_based_fallback_verifier(
    claim: str,
    evidence_passages: List[Dict[str, Any]]
) -> VerificationResult:
    """
    Deterministic lexical/heuristic verifier used when LLM is unavailable or for rapid unit testing.
    Checks for direct semantic keyword coverage, negation cues, or lack of evidence.
    """
    if not evidence_passages:
        return VerificationResult(
            claim=claim,
            verdict="Uncertain",
            confidence=0.20,
            reasoning="No relevant evidence was retrieved from the knowledge base to support or refute the claim.",
            evidence_refs=[]
        )

    best_ev = evidence_passages[0]
    hybrid_score = best_ev.get("hybrid_score", 0.0)
    ev_text = best_ev.get("text", "").lower()
    claim_lower = claim.lower()

    # Evidence match threshold
    if hybrid_score < 0.35:
        return VerificationResult(
            claim=claim,
            verdict="Uncertain",
            confidence=round(max(hybrid_score, 0.30), 2),
            reasoning=f"Weak evidence correlation ({hybrid_score:.2f}). Evidence is insufficient for definitive verdict.",
            evidence_refs=[best_ev]
        )

    # Check for direct negation or contradiction cues
    negation_words = ["not", "never", "no", "does not", "did not", "refutes", "false", "untrue"]
    has_claim_negation = any(neg in claim_lower for neg in negation_words)
    has_ev_negation = any(neg in ev_text for neg in negation_words)

    # Keyword overlap
    claim_words = set(re.findall(r"\w+", claim_lower)) - {"the", "a", "an", "is", "was", "in", "at", "by", "for", "to"}
    ev_words = set(re.findall(r"\w+", ev_text))
    overlap = len(claim_words & ev_words) / len(claim_words) if claim_words else 0.0

    if overlap > 0.65 and (has_claim_negation == has_ev_negation):
        confidence = min(round(0.60 + (overlap * 0.35), 2), 0.95)
        return VerificationResult(
            claim=claim,
            verdict="True",
            confidence=confidence,
            reasoning=f"Retrieved passage from {best_ev.get('source')} strongly corroborates the factual entities and assertions.",
            evidence_refs=[best_ev]
        )
    elif overlap > 0.40 and (has_claim_negation != has_ev_negation):
        confidence = min(round(0.60 + (overlap * 0.30), 2), 0.90)
        return VerificationResult(
            claim=claim,
            verdict="False",
            confidence=confidence,
            reasoning=f"Retrieved passage from {best_ev.get('source')} contradicts the claim polarity.",
            evidence_refs=[best_ev]
        )
    else:
        return VerificationResult(
            claim=claim,
            verdict="Uncertain",
            confidence=round(0.40 + (overlap * 0.15), 2),
            reasoning="Evidence overlaps partially with claim keywords but lacks definitive confirmation.",
            evidence_refs=[best_ev]
        )

def verify_claim(
    claim: str,
    evidence_passages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL
) -> VerificationResult:
    """
    Agent 3 entry point: Verifies a single claim against retrieved evidence passages.
    1. Prompts LLM for evidence-grounded verification.
    2. Calibrates confidence: if confidence < threshold -> forces 'Uncertain'.
    3. Falls back smoothly to rule-based verification if LLM service is offline.
    """
    system_prompt = load_prompt_template()
    user_prompt = build_verifier_prompt(claim, evidence_passages)

    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.0}
        )
        content = response["message"]["content"].strip()
        
        # Parse JSON from LLM response
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            verdict = str(data.get("verdict", "Uncertain")).strip().capitalize()
            confidence = float(data.get("confidence", 0.5))
            reasoning = str(data.get("reasoning", "")).strip()
            cited_ids = data.get("cited_doc_ids", [])

            # Filter cited evidence passages
            cited_refs = [ev for ev in evidence_passages if ev.get("id") in cited_ids]
            if not cited_refs and evidence_passages:
                cited_refs = [evidence_passages[0]]

            # Enforce calibration threshold
            if confidence < UNCERTAIN_CONFIDENCE_THRESHOLD:
                verdict = "Uncertain"
            elif verdict not in ("True", "False", "Uncertain"):
                verdict = "Uncertain"

            return VerificationResult(
                claim=claim,
                verdict=verdict,
                confidence=round(confidence, 2),
                reasoning=reasoning or "Evaluated against retrieved evidence.",
                evidence_refs=cited_refs
            )
    except Exception:
        # LLM offline or unresponsive -> graceful fallback to deterministic verifier
        pass

    return rule_based_fallback_verifier(claim, evidence_passages)

if __name__ == "__main__":
    # Test cases: supported claim, contradicted claim, and claim with weak/no evidence
    test_cases = [
        {
            "claim": "The US unemployment rate was 3.5% in January 2024",
            "evidence": [{
                "id": "doc_00000",
                "source": "Bureau of Labor Statistics Historical Archives",
                "text": "The United States Bureau of Labor Statistics reported that the national unemployment rate hovered between 3.4% and 3.9% in late 2023 and early 2024, reaching 3.5% in early 2024.",
                "hybrid_score": 1.0
            }]
        },
        {
            "claim": "The Affordable Care Act mandates free sex change surgeries",
            "evidence": [{
                "id": "doc_00004",
                "source": "Congressional Research Service (CRS)",
                "text": "The Affordable Care Act of 2010 mandates coverage of essential health benefits, preventive screenings, and prohibits discrimination based on pre-existing conditions. The statutory legislation does not mandate or establish free gender confirmation surgeries.",
                "hybrid_score": 0.85
            }]
        },
        {
            "claim": "Alien spacecraft landed in Central Park last Sunday",
            "evidence": []  # No evidence
        }
    ]

    print("=== Testing Agent 3: Verifier Agent ===\n")
    for idx, tc in enumerate(test_cases, 1):
        res = verify_claim(tc["claim"], tc["evidence"])
        print(f"--- Test #{idx} ---")
        print(f"Claim      : {res.claim}")
        print(f"Verdict    : {res.verdict} (Confidence: {res.confidence * 100:.0f}%)")
        print(f"Reasoning  : {res.reasoning}")
        print(f"Evidence   : {[ev.get('source') for ev in res.evidence_refs]}")
        print()
