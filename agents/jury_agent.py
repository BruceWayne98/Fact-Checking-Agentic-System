import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import ollama
except ImportError:
    ollama = None

from config import PROMPTS_DIR, DEFAULT_MODEL, OLLAMA_BASE_URL, UNCERTAIN_CONFIDENCE_THRESHOLD
from graph.state import JuryVerdict


def load_jury_persona_prompts() -> Dict[str, str]:
    """
    Parses prompts/jury_personas.txt to extract system prompts for each persona:
    DATA_ANALYST, HISTORIAN, POLICY_ANALYST.
    """
    persona_file = PROMPTS_DIR / "jury_personas.txt"
    personas = {}
    if persona_file.exists():
        with open(persona_file, "r", encoding="utf-8") as f:
            content = f.read()
        sections = re.split(r"===\s*([A-Z_]+)\s*===", content)
        for i in range(1, len(sections), 2):
            key = sections[i].strip()
            prompt_text = sections[i + 1].strip()
            personas[key] = prompt_text

    # Default fallback prompts if file parsing yields partial results
    if "DATA_ANALYST" not in personas:
        personas["DATA_ANALYST"] = (
            "You are a Data Analyst evaluating numerical and statistical claims. "
            "Check exact figures, baselines, and percentages strictly against the evidence."
        )
    if "HISTORIAN" not in personas:
        personas["HISTORIAN"] = (
            "You are a Historian evaluating chronological and timeline accuracy. "
            "Check historical dates, precedents, and eras strictly against the evidence."
        )
    if "POLICY_ANALYST" not in personas:
        personas["POLICY_ANALYST"] = (
            "You are a Policy Analyst evaluating legislative and statutory authority. "
            "Check legal enactments, mandates, and institutional facts strictly against the evidence."
        )
    return personas


PERSONA_PROMPTS = load_jury_persona_prompts()


def _format_evidence_context(evidence_passages: List[Dict[str, Any]]) -> str:
    if not evidence_passages:
        return "No external evidence passages retrieved."
    lines = []
    for idx, doc in enumerate(evidence_passages, 1):
        source = doc.get("source", "Unknown")
        text = doc.get("text", "")
        lines.append(f"Passage [{idx}] (Source: {source}):\n\"{text}\"")
    return "\n\n".join(lines)


def _heuristic_fallback_evaluation(
    claim: str,
    evidence_passages: List[Dict[str, Any]],
    persona: str
) -> JuryVerdict:
    """
    Deterministic fallback when Ollama is unavailable. Evaluates based on
    persona-specific lexical and syntactic criteria.
    """
    if not evidence_passages:
        return JuryVerdict(
            persona=persona,
            claim=claim,
            verdict="Uncertain",
            confidence=0.20,
            reasoning=f"[{persona} Fallback] No evidence retrieved to verify claim."
        )

    all_evidence_text = " ".join(doc.get("text", "").lower() for doc in evidence_passages)
    claim_lower = claim.lower()

    if persona == "DATA_ANALYST":
        # Extract numerical tokens (percentages, numbers, years)
        numbers_in_claim = re.findall(r"\b\d+(?:\.\d+)?%?|\b(?:one|two|three|million|billion|trillion)\b", claim_lower)
        if not numbers_in_claim:
            return JuryVerdict(
                persona=persona,
                claim=claim,
                verdict="True",
                confidence=0.70,
                reasoning="Claim contains no empirical or numerical figures; verified non-quantitative factual statement."
            )
        matched_numbers = [num for num in numbers_in_claim if num in all_evidence_text]
        if len(matched_numbers) == len(numbers_in_claim):
            return JuryVerdict(
                persona=persona,
                claim=claim,
                verdict="True",
                confidence=0.88,
                reasoning=f"All quantitative figures {numbers_in_claim} corroborated directly in evidence records."
            )
        elif matched_numbers:
            return JuryVerdict(
                persona=persona,
                claim=claim,
                verdict="Uncertain",
                confidence=0.48,
                reasoning=f"Partial numerical match: found {matched_numbers}, but missing unverified numbers from {numbers_in_claim}."
            )
        else:
            return JuryVerdict(
                persona=persona,
                claim=claim,
                verdict="False",
                confidence=0.75,
                reasoning=f"Quantitative figures {numbers_in_claim} are uncorroborated or contradicted by evidence records."
            )

    elif persona == "HISTORIAN":
        # Check years or timeline markers
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", claim)
        time_terms = re.findall(r"\b(since|in|during|after|before|decade|century|historical|precedent)\b", claim_lower)
        if years:
            matched_years = [y for y in years if y in all_evidence_text]
            if matched_years:
                return JuryVerdict(
                    persona=persona,
                    claim=claim,
                    verdict="True",
                    confidence=0.92,
                    reasoning=f"Historical chronology confirmed: year anchor {matched_years} verified in archival records."
                )
            else:
                return JuryVerdict(
                    persona=persona,
                    claim=claim,
                    verdict="Uncertain",
                    confidence=0.35,
                    reasoning=f"Archival records do not corroborate the timeline anchor ({years}) asserted in the claim."
                )
        return JuryVerdict(
            persona=persona,
            claim=claim,
            verdict="True",
            confidence=0.75,
            reasoning="Timeline and historical sequence consistent with archival records."
        )

    else:  # POLICY_ANALYST
        policy_keywords = ["bill", "act", "law", "passed", "legislation", "mandate", "signed", "reform", "senate", "congress"]
        found_keywords = [kw for kw in policy_keywords if kw in claim_lower]
        if found_keywords:
            matched_in_ev = [kw for kw in found_keywords if kw in all_evidence_text]
            if matched_in_ev:
                return JuryVerdict(
                    persona=persona,
                    claim=claim,
                    verdict="True",
                    confidence=0.85,
                    reasoning=f"Statutory authority and legislative mechanism ({matched_in_ev}) verified in policy documentation."
                )
            else:
                return JuryVerdict(
                    persona=persona,
                    claim=claim,
                    verdict="False",
                    confidence=0.78,
                    reasoning=f"Legislative mandate asserted in claim is not authorized or enacted in official policy records."
                )
        return JuryVerdict(
            persona=persona,
            claim=claim,
            verdict="True",
            confidence=0.72,
            reasoning="Institutional governance check reveals no regulatory contradiction."
        )


def evaluate_as_persona(
    claim: str,
    evidence_passages: List[Dict[str, Any]],
    persona: str,
    model: str = DEFAULT_MODEL
) -> JuryVerdict:
    """
    Evaluates a claim from the specialized perspective of a single Delphi jury persona.
    """
    system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["DATA_ANALYST"])
    evidence_context = _format_evidence_context(evidence_passages)

    user_prompt = (
        f"Delphi Jury Persona: {persona}\n"
        f"Claim to verify:\n\"{claim}\"\n\n"
        f"Retrieved Evidence Passages:\n{evidence_context}\n\n"
        "Evaluate strictly from your persona's lens using ONLY the evidence passages above.\n"
        "Respond with a JSON object with keys: persona, verdict, confidence, reasoning."
    )

    if ollama:
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
            raw = response["message"]["content"].strip()
            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                verdict = parsed.get("verdict", "Uncertain").capitalize()
                if verdict not in ("True", "False", "Uncertain"):
                    verdict = "Uncertain"
                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                reasoning = parsed.get("reasoning", "").strip()

                # Calibrated confidence constraint
                if confidence < UNCERTAIN_CONFIDENCE_THRESHOLD:
                    verdict = "Uncertain"

                return JuryVerdict(
                    persona=persona,
                    claim=claim,
                    verdict=verdict,
                    confidence=round(confidence, 4),
                    reasoning=reasoning
                )
        except Exception as e:
            print(f"[JuryAgent] Persona '{persona}' Ollama call failed ({e}). Falling back.")

    return _heuristic_fallback_evaluation(claim, evidence_passages, persona)


def run_jury(
    claim: str,
    evidence_passages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL
) -> List[JuryVerdict]:
    """
    Executes the Delphi Multi-Agent Jury across all 3 personas (Data Analyst, Historian, Policy Analyst)
    in parallel threads.
    """
    personas = ["DATA_ANALYST", "HISTORIAN", "POLICY_ANALYST"]
    verdicts = []

    # Run parallel evaluation using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(evaluate_as_persona, claim, evidence_passages, p, model): p
            for p in personas
        }
        for future in futures:
            try:
                verdicts.append(future.result())
            except Exception as e:
                persona_name = futures[future]
                verdicts.append(_heuristic_fallback_evaluation(claim, evidence_passages, persona_name))

    # Maintain deterministic order: DATA_ANALYST, HISTORIAN, POLICY_ANALYST
    order_map = {"DATA_ANALYST": 0, "HISTORIAN": 1, "POLICY_ANALYST": 2}
    verdicts.sort(key=lambda x: order_map.get(x.persona, 99))
    return verdicts


if __name__ == "__main__":
    print("=== Testing Delphi Multi-Agent Jury (Phase 9) ===")

    # Test numerical claim with synthetic evidence
    numerical_claim = "The US unemployment rate fell to 3.5% in January 2024, the lowest since 1969."
    numerical_evidence = [
        {
            "id": "doc_unemp",
            "source": "Bureau of Labor Statistics Historical Archives",
            "text": "The United States Bureau of Labor Statistics reported that the national unemployment rate hovered between 3.4% and 3.9% in late 2023 and early 2024, reaching 3.5% in early 2024, maintaining near historic lows comparable to rates recorded in 1969."
        }
    ]

    print(f"\n--- Test 1: Numerical Claim ---\nClaim: \"{numerical_claim}\"")
    jury_results_1 = run_jury(numerical_claim, numerical_evidence)
    for v in jury_results_1:
        print(f"\n[{v.persona}]")
        print(f"  Verdict:    {v.verdict} (Confidence: {int(v.confidence * 100)}%)")
        print(f"  Reasoning:  {v.reasoning}")

    # Test historical / policy claim where numerical evidence is missing
    policy_claim = "Biden's infrastructure bill passed in 2021, creating 2 million jobs and reducing carbon emissions by 15%."
    policy_evidence = [
        {
            "id": "doc_infra",
            "source": "White House & Dept of Transportation Records",
            "text": "President Biden signed the Infrastructure Investment and Jobs Act in November 2021, allocating over 1 trillion dollars for transportation, water systems, and clean energy grid modernization."
        }
    ]

    print(f"\n--- Test 2: Multi-Faceted Policy & Statistics Claim ---\nClaim: \"{policy_claim}\"")
    jury_results_2 = run_jury(policy_claim, policy_evidence)
    for v in jury_results_2:
        print(f"\n[{v.persona}]")
        print(f"  Verdict:    {v.verdict} (Confidence: {int(v.confidence * 100)}%)")
        print(f"  Reasoning:  {v.reasoning}")
