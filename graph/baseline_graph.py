import sys
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langgraph.graph import StateGraph, START, END
from graph.state import TrustAgentState
from agents.claim_extractor import extract_claims
from agents.retrieval_agent import retrieve_evidence_for_claims
from agents.verifier_agent import verify_claim
from agents.explainer_agent import generate_report

# Node 1: Claim Extraction Agent
def claim_extractor_node(state: TrustAgentState) -> Dict[str, Any]:
    raw_article = state.get("article", "")
    print(f"\n[Node 1: Claim Extractor] Processing input: \"{raw_article[:60]}...\"")
    extracted_data = extract_claims(raw_article)
    claims = extracted_data["claims"]
    print(f"[Node 1] Extracted {len(claims)} claim(s): {claims}")
    return {"claims": claims}

# Node 2: Hybrid Evidence Retrieval Agent
def retrieval_node(state: TrustAgentState) -> Dict[str, Any]:
    claims = state.get("claims", [])
    print(f"\n[Node 2: Retrieval Agent] Searching hybrid evidence for {len(claims)} claim(s)...")
    evidence_map = retrieve_evidence_for_claims(claims)
    for c, evs in evidence_map.items():
        print(f"  - Claim: \"{c[:40]}...\" -> Retrieved {len(evs)} passage(s)")
    return {"evidence_map": evidence_map}

# Node 3: Grounded Verifier Agent
def verifier_node(state: TrustAgentState) -> Dict[str, Any]:
    claims = state.get("claims", [])
    evidence_map = state.get("evidence_map", {})
    print(f"\n[Node 3: Verifier Agent] Cross-examining {len(claims)} claim(s)...")

    verifications = []
    for claim in claims:
        passages = evidence_map.get(claim, [])
        result = verify_claim(claim, passages)
        verifications.append(result)
        print(f"  - Verdict: {result.verdict} (Confidence: {int(result.confidence * 100)}%) for \"{claim[:40]}...\"")

    return {"verifications": verifications}

# Node 4: Explainer Agent
def explainer_node(state: TrustAgentState) -> Dict[str, Any]:
    verifications = state.get("verifications", [])
    print(f"\n[Node 4: Explainer Agent] Generating final audit report for {len(verifications)} verification(s)...")
    report = generate_report(verifications)
    return {"final_report": report}

def build_baseline_graph():
    """
    Constructs the 4-Agent sequential LangGraph workflow.
    START -> Claim Extractor -> Retrieval Agent -> Verifier Agent -> Explainer Agent -> END
    """
    workflow = StateGraph(TrustAgentState)

    # Register agent nodes
    workflow.add_node("claim_extractor", claim_extractor_node)
    workflow.add_node("retrieval_agent", retrieval_node)
    workflow.add_node("verifier_agent", verifier_node)
    workflow.add_node("explainer_agent", explainer_node)

    # Define linear execution flow
    workflow.add_edge(START, "claim_extractor")
    workflow.add_edge("claim_extractor", "retrieval_agent")
    workflow.add_edge("retrieval_agent", "verifier_agent")
    workflow.add_edge("verifier_agent", "explainer_agent")
    workflow.add_edge("explainer_agent", END)

    app = workflow.compile()
    return app

if __name__ == "__main__":
    app = build_baseline_graph()
    test_article = "The US unemployment rate fell to 3.5% in January 2024, the lowest since 1969."
    
    print("=== Executing Baseline TRUST Agents Graph ===")
    final_output = app.invoke({"article": test_article})
    print("\n" + "="*50)
    print("FINAL COMPILED AUDIT REPORT:")
    print("="*50)
    print(final_output["final_report"])
