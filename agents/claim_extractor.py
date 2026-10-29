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

def load_prompt_template() -> str:
    prompt_file = PROMPTS_DIR / "claim_extractor.txt"
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Extract verifiable claims as a JSON list of strings."

def extract_with_spacy_ner(text: str) -> List[Dict[str, Any]]:
    """
    Extracts named entities that anchor verifiable claims.
    Entities include PERSON, ORG, GPE, DATE, MONEY, PERCENT, CARDINAL, etc.
    """
    if not nlp:
        return []
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return entities

def extract_with_dependency_parser(text: str) -> List[str]:
    """
    Extracts Subject-Verb-Object (SVO) factual propositions using dependency parsing.
    Grammar anchors:
      - Subject: nsubj, nsubjpass
      - Verb: ROOT, relcl, advcl
      - Object: dobj, pobj, attr, ccomp, xcomp
    """
    if not nlp:
        return []
    doc = nlp(text)
    svo_triplets = []

    for token in doc:
        # Look for finite verbs
        if token.pos_ in ("VERB", "AUX"):
            subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            objects = [w for w in token.rights if w.dep_ in ("dobj", "attr", "prep", "acomp", "xcomp")]

            if subjects:
                for subj in subjects:
                    # Collect subtree or full phrase for context
                    subj_text = "".join(w.text_with_ws for w in subj.subtree).strip()
                    verb_text = token.lemma_ if token.lemma_ != "-PRON-" else token.text
                    
                    if objects:
                        for obj in objects:
                            obj_text = "".join(w.text_with_ws for w in obj.subtree).strip()
                            svo_triplets.append(f"{subj_text} {token.text} {obj_text}".strip())
                    else:
                        # Intransitive or passive verb proposition
                        right_text = "".join(w.text_with_ws for w in token.rights).strip()
                        if right_text:
                            svo_triplets.append(f"{subj_text} {token.text} {right_text}".strip())

    return list(dict.fromkeys(svo_triplets))  # remove duplicates

def refine_with_llm(
    text: str,
    ner_entities: List[Dict[str, Any]],
    svo_candidates: List[str],
    model: str = DEFAULT_MODEL
) -> List[str]:
    """
    Uses the Ollama LLM to synthesize raw text + NER + SVO candidates into clean,
    atomic, verifiable claim statements.
    Falls back gracefully to rule-based candidates if Ollama is not reachable.
    """
    system_prompt = load_prompt_template()
    
    entities_summary = ", ".join([f"{e['text']} ({e['label']})" for e in ner_entities]) if ner_entities else "None"
    svo_summary = "\n".join([f"- {s}" for s in svo_candidates]) if svo_candidates else "None"

    user_prompt = f"""Raw Text:
"{text}"

Identified Entities:
{entities_summary}

Identified SVO Candidates:
{svo_summary}

Please extract and return the distinct, independently verifiable factual claims:"""

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
        
        # Extract JSON array from markdown or plain text
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            claims = json.loads(match.group(0))
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return claims
    except Exception as e:
        # LLM not running or connection error -> fallback to deterministic heuristics
        pass

    # Fallback heuristic: use SVO candidates or clean sentence splits
    if svo_candidates:
        return svo_candidates
    return [text.strip()]

def extract_claims(text: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Main entry point for Agent 1 (Claim Extractor):
    1. Runs spaCy NER to identify key factual anchors.
    2. Runs Dependency Parsing to identify Subject-Verb-Object propositions.
    3. Prompts LLM (with fallback) to produce clean, isolated claim strings.
    """
    entities = extract_with_spacy_ner(text)
    svo_candidates = extract_with_dependency_parser(text)
    final_claims = refine_with_llm(text, entities, svo_candidates, model=model)

    return {
        "raw_text": text,
        "entities": entities,
        "svo_candidates": svo_candidates,
        "claims": final_claims
    }

if __name__ == "__main__":
    test_statements = [
        "The US unemployment rate fell to 3.5% in January 2024, the lowest since 1969.",
        "Says the Annies List political group supports third-trimester abortions on demand.",
        "Hillary Clinton agrees with John McCain by voting to give George Bush the benefit of the doubt on Iran.",
        "Under the new bill passed by Congress last Tuesday, small businesses will receive a 20 percent tax credit."
    ]

    print("=== Testing Agent 1: Claim Extractor ===\n")
    for i, stmt in enumerate(test_statements, 1):
        print(f"--- Example #{i} ---")
        print(f"Input: {stmt}")
        res = extract_claims(stmt)
        print("NER Entities:", [f"{e['text']} ({e['label']})" for e in res["entities"]])
        print("SVO Candidates:", res["svo_candidates"])
        print("Final Extracted Claims:")
        for c in res["claims"]:
            print(f"  • {c}")
        print()
