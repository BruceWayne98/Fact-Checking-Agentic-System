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

from config import PROMPTS_DIR, DEFAULT_MODEL, OLLAMA_BASE_URL
from graph.state import VerificationResult

def load_prompt_template() -> str:
    prompt_file = PROMPTS_DIR / "explainer.txt"
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Generate a structured, evidence-cited markdown verification report."

def build_explainer_prompt(verification_results: List[VerificationResult], context_text: Optional[str] = None) -> str:
    """Formats verification results into a structured prompt for the LLM explainer."""
    summary_items = []
    for idx, r in enumerate(verification_results, 1):
        citations = []
        for ev in r.evidence_refs:
            src = ev.get("source", "Unknown Source")
            txt = ev.get("text", "")
            citations.append(f"- Source: {src}\n  Excerpt: \"{txt}\"")
        citations_str = "\n".join(citations) if citations else "No explicit evidence cited."

        item = f"""Claim #{idx}:
- Assertion: "{r.claim}"
- Verdict: {r.verdict}
- Confidence: {int(r.confidence * 100)}%
- Reason: {r.reasoning}
- Evidence Citations:
{citations_str}"""
        summary_items.append(item)

    items_block = "\n\n".join(summary_items)
    user_prompt = f"""Generate a professional, human-auditable fact-checking report for the following evaluated claims:

{items_block}

Format as an audit report in GitHub-flavored Markdown:"""
    return user_prompt

def deterministic_report_generator(verification_results: List[VerificationResult]) -> str:
    """
    Produces a structured, formatted markdown report directly from VerificationResult objects.
    Serves as an instant, auditable template or fallback when LLM is offline.
    """
    total = len(verification_results)
    true_count = sum(1 for r in verification_results if r.verdict == "True")
    false_count = sum(1 for r in verification_results if r.verdict == "False")
    uncertain_count = sum(1 for r in verification_results if r.verdict == "Uncertain")

    lines = [
        "# 📋 TRUST Agents Verification Audit Report",
        "",
        "## Executive Summary",
        f"- **Total Claims Evaluated:** {total}",
        f"- **Verified True:** {true_count} ✅",
        f"- **Refuted / False:** {false_count} ❌",
        f"- **Uncertain / Abstentions:** {uncertain_count} ⚠️",
        "",
        "---",
        "",
        "## Detailed Claim-by-Claim Breakdown",
        ""
    ]

    for idx, r in enumerate(verification_results, 1):
        if r.verdict == "True":
            badge = "✅ **VERIFIED TRUE**"
        elif r.verdict == "False":
            badge = "❌ **FALSE / REFUTED**"
        else:
            badge = "⚠️ **UNCERTAIN / UNPROVEN**"

        lines.append(f"### Claim {idx}: *\"{r.claim}\"*")
        lines.append(f"- **Verdict:** {badge}")
        lines.append(f"- **Confidence:** {int(r.confidence * 100)}%")
        lines.append(f"- **Reasoning:** {r.reasoning}")
        if r.logic_trace:
            lines.append(f"- **Logic Trace:** `{r.logic_trace}`")

        lines.append("")
        lines.append("**Cited Evidence:**")
        if r.evidence_refs:
            for ev in r.evidence_refs:
                src = ev.get("source", "Knowledge Base")
                txt = ev.get("text", "")
                score = ev.get("hybrid_score", None)
                score_str = f" (Retrieval Score: {score})" if score is not None else ""
                lines.append(f"> \"{txt}\"")
                lines.append(f"> — *{src}*{score_str}")
                lines.append("")
        else:
            lines.append("> *No supporting or contradicting evidence available in knowledge corpus.*")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## Methodology & Transparency Note")
    lines.append("This report was generated using the **TRUST Agents** modular pipeline:")
    lines.append("1. **Claim Extraction:** spaCy NER + dependency parsing isolated factual propositions.")
    lines.append("2. **Hybrid Evidence Retrieval:** Multi-modal lexical (BM25) and semantic (FAISS/SBERT) search over curated records.")
    lines.append("3. **Calibrated Verification:** Confidence thresholding enforced abstention (`Uncertain`) rather than ungrounded guessing.")

    return "\n".join(lines)

def generate_report(
    verification_results: List[VerificationResult],
    model: str = DEFAULT_MODEL
) -> str:
    """
    Agent 4 entry point: Transforms a list of VerificationResults into a complete,
    human-readable markdown audit report.
    """
    if not verification_results:
        return "# Verification Report\n\n*No claims were provided for evaluation.*"

    system_prompt = load_prompt_template()
    user_prompt = build_explainer_prompt(verification_results)

    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.2}
        )
        content = response["message"]["content"].strip()
        if content and len(content) > 100:
            return content
    except Exception:
        # LLM offline or unresponsive -> deterministic fallback report
        pass

    return deterministic_report_generator(verification_results)

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    from graph.state import VerificationResult

    dummy_results = [
        VerificationResult(
            claim="The US unemployment rate was 3.5% in January 2024",
            verdict="True",
            confidence=0.88,
            reasoning="The Bureau of Labor Statistics reported that unemployment hovered near historic lows, reaching 3.5% in early 2024.",
            evidence_refs=[{
                "id": "doc_00000",
                "source": "Bureau of Labor Statistics Historical Archives",
                "text": "The United States Bureau of Labor Statistics reported that the national unemployment rate hovered between 3.4% and 3.9% in late 2023 and early 2024, reaching 3.5% in early 2024.",
                "hybrid_score": 1.0
            }]
        ),
        VerificationResult(
            claim="The Affordable Care Act mandates free sex change surgeries",
            verdict="False",
            confidence=0.82,
            reasoning="The statutory provisions of the ACA guarantee essential health benefits but do not mandate free gender reassignment procedures.",
            evidence_refs=[{
                "id": "doc_00004",
                "source": "Congressional Research Service (CRS)",
                "text": "The statutory legislation does not mandate or establish free gender confirmation surgeries.",
                "hybrid_score": 0.85
            }]
        ),
        VerificationResult(
            claim="Alien spacecraft landed in Central Park last Sunday",
            verdict="Uncertain",
            confidence=0.20,
            reasoning="No credible historical or news evidence exists in the corpus regarding extraterrestrial contact in Central Park.",
            evidence_refs=[]
        )
    ]

    print("=== Testing Agent 4: Explainer Agent ===\n")
    report = generate_report(dummy_results)
    print(report)
