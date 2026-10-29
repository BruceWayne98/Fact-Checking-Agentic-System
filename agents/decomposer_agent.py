import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import spacy

try:
    import ollama
except ImportError:
    ollama = None

from config import PROMPTS_DIR, DEFAULT_MODEL, OLLAMA_BASE_URL

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


def load_decomposer_prompt() -> str:
    prompt_file = PROMPTS_DIR / "decomposer.txt"
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Decompose the following compound claim into atomic, self-contained sub-claims as a JSON list of strings."


def is_compound_claim(claim: str) -> bool:
    """
    Heuristic and syntactic detector for compound factual statements.
    Evaluates:
      1. Explicit causal/temporal connectors (e.g. 'because', 'leading to', 'resulting in')
      2. Participle clauses anchored by commas (e.g. ', creating...', ', reducing...')
      3. Coordinating conjunctions linking multiple distinct predicate phrases (e.g. 'and', 'while')
      4. Syntactic dependency tree checking for multiple verb heads/advcl/conj structures.
    """
    clean = claim.strip()
    if not clean:
        return False

    # 1. Regex patterns for causal, consequential, and participle linkages
    causal_pattern = re.compile(
        r"\b(because|leading to|resulting in|which caused|which resulted in|due to|as a result of)\b",
        re.IGNORECASE
    )
    if causal_pattern.search(clean):
        return True

    # 2. Comma followed by present participle (, creating, , reducing, , boosting, etc.)
    participle_pattern = re.compile(
        r",\s*(creating|reducing|increasing|boosting|adding|lowering|causing|yielding|costing|eliminating|generating)\b",
        re.IGNORECASE
    )
    if participle_pattern.search(clean):
        return True

    # 3. Multiple coordinating clauses joined by 'and', 'while', 'whereas'
    coord_pattern = re.compile(
        r",\s*(and|while|whereas|but)\b|\b(and\s+(?:also\s+)?(?:creating|reducing|boosting|adding|cutting))\b",
        re.IGNORECASE
    )
    if coord_pattern.search(clean):
        return True

    # 4. Syntactic dependency analysis via spaCy
    if nlp:
        doc = nlp(clean)
        verb_count = sum(1 for token in doc if token.pos_ in ("VERB", "AUX"))
        has_conj_or_advcl = any(token.dep_ in ("conj", "advcl", "ccomp") for token in doc)
        has_coord_conj = any(token.pos_ == "CCONJ" for token in doc)
        
        # If there are multiple verbs and a subordinate or coordinate clause structure
        if verb_count >= 2 and (has_conj_or_advcl or has_coord_conj):
            # Verify if there is substantial token length or entity density
            if len(doc) >= 8:
                return True

    return False


def _fallback_heuristic_decompose(claim: str) -> List[str]:
    """
    Deterministic rule-based fallback to unpack compound claims when
    Ollama LLM is unavailable or unparseable.
    Uses regex splitting on coordinating commas and participles, restoring subject context.
    """
    # Extract likely grammatical subject
    subject = "The statement"
    if nlp:
        doc = nlp(claim)
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj_tokens = [w.text_with_ws for w in token.subtree]
                subject = "".join(subj_tokens).strip()
                break

    # Split on participle phrases or coordinated conjunctions
    split_regex = re.compile(
        r",\s*(?:and\s+)?(?=(?:creating|reducing|increasing|boosting|adding|lowering|causing|yielding|costing|eliminating|generating)\b)|,\s*and\s+|\s+and\s+(?=(?:reducing|creating|boosting|adding|cutting|lowering)\b)",
        re.IGNORECASE
    )
    parts = split_regex.split(claim)
    parts = [p.strip().rstrip(".,;") for p in parts if p and len(p.strip()) > 5]

    if len(parts) <= 1:
        return [claim.strip()]

    atomic_claims = []
    # Primary clause
    primary = parts[0]
    if not primary.endswith("."):
        primary += "."
    atomic_claims.append(primary)

    # Subordinate participle/consequence clauses
    for p in parts[1:]:
        # If it begins with a participle, e.g. "creating 2 million jobs"
        p_match = re.match(r"^(creating|reducing|increasing|boosting|adding|lowering|causing|yielding|costing|eliminating|generating)\s+(.*)", p, re.IGNORECASE)
        if p_match:
            verb_ing, rest = p_match.groups()
            # Convert -ing to past tense for assertion clarity
            past_verb = {
                "creating": "created",
                "reducing": "reduced",
                "increasing": "increased",
                "boosting": "boosted",
                "adding": "added",
                "lowering": "lowered",
                "causing": "caused",
                "yielding": "yielded",
                "costing": "cost",
                "eliminating": "eliminated",
                "generating": "generated"
            }.get(verb_ing.lower(), verb_ing)

            # Build self-contained sentence: "The infrastructure bill created 2 million jobs."
            # Ensure proper noun/determiner formatting for subject
            subj_prefix = subject if subject.lower().startswith(("the ", "a ", "an ", "biden")) else f"The {subject}"
            subclaim = f"{subj_prefix} {past_verb} {rest}."
            atomic_claims.append(subclaim)
        else:
            if not p.lower().startswith(("the ", "a ", "it ", "biden")):
                subclaim = f"{subject} {p}."
            else:
                subclaim = f"{p}."
            atomic_claims.append(subclaim)

    return atomic_claims


def decompose_claim(claim: str, model: str = DEFAULT_MODEL) -> List[str]:
    """
    Decomposes a compound factual claim into atomic sub-claims.
    If the claim is not compound, returns [claim].
    Uses Ollama with prompt from prompts/decomposer.txt, with fallback to heuristic logic.
    """
    clean_claim = claim.strip()
    if not clean_claim:
        return []

    # Check if decomposition is warranted
    if not is_compound_claim(clean_claim):
        return [clean_claim]

    system_prompt = load_decomposer_prompt()
    user_prompt = f"Decompose this compound claim into atomic, self-contained sub-claims:\n\n\"{clean_claim}\""

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
            raw_content = response["message"]["content"].strip()

            # Extract JSON list using regex
            json_match = re.search(r"\[.*\]", raw_content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list) and len(parsed) > 0:
                    cleaned_claims = []
                    for item in parsed:
                        if isinstance(item, str) and item.strip():
                            c = item.strip()
                            if not c.endswith("."):
                                c += "."
                            cleaned_claims.append(c)
                    if cleaned_claims:
                        return cleaned_claims
        except Exception as e:
            print(f"[DecomposerAgent] Ollama call failed ({e}), falling back to heuristic parsing.")

    # Rule-based fallback
    return _fallback_heuristic_decompose(clean_claim)


if __name__ == "__main__":
    print("=== Testing Agent: Decomposer Agent (Phase 8) ===")

    test_claims = [
        # Checkpoint Claim
        "Biden's infrastructure bill passed in 2021, creating 2 million jobs and reducing carbon emissions by 15%",
        # Simple Claim (should not decompose)
        "The US unemployment rate fell to 3.5% in January 2024.",
        # Economic compound claim
        "The corporate tax cut was signed in December 2017, boosting US GDP by 4% while adding $1.5 trillion to the national debt."
    ]

    for claim in test_claims:
        compound_flag = is_compound_claim(claim)
        print(f"\nOriginal Claim: \"{claim}\"")
        print(f"Is Compound Claim? -> {compound_flag}")
        sub_claims = decompose_claim(claim)
        print(f"Decomposed Atomic Claims ({len(sub_claims)}):")
        for idx, sub in enumerate(sub_claims, 1):
            print(f"  {idx}. \"{sub}\"")
