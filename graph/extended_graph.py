import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langgraph.graph import StateGraph, START, END
from graph.state import TrustAgentState, VerificationResult, JuryVerdict
from agents.claim_extractor import extract_claims
from agents.decomposer_agent import is_compound_claim, decompose_claim
from agents.retrieval_agent import retrieve_evidence_for_claims
from agents.verifier_agent import verify_claim
from agents.explainer_agent import generate_report
from agents.jury_agent import evaluate_as_persona
from agents.logic_aggregator import detect_logical_structure, aggregate_verdicts
from config import UNCERTAIN_CONFIDENCE_THRESHOLD


# Node 1: Claim Extraction Agent
def claim_extractor_node(state: TrustAgentState) -> Dict[str, Any]:
    raw_article = state.get("article", "")
    print(f"\n[Node 1: Claim Extractor] Processing input: \"{raw_article[:60]}...\"")
    extracted_data = extract_claims(raw_article)
    claims = extracted_data["claims"]
    
    # Check for compound claims
    has_compound = any(is_compound_claim(c) for c in claims)
    print(f"[Node 1] Extracted {len(claims)} claim(s). Compound assertions detected: {has_compound}")
    return {
        "claims": claims,
        "has_compound_claims": has_compound,
        "decomposed_claims_map": {}
    }


# Node: Decomposer Agent (Phase 8 Extension)
def decomposer_node(state: TrustAgentState) -> Dict[str, Any]:
    claims = state.get("claims", [])
    decomposed_map = state.get("decomposed_claims_map", {})
    all_atomic_claims = []

    print(f"\n[Node: Decomposer Agent] Decomposing {len(claims)} claim(s) into atomic propositions...")

    for claim in claims:
        if is_compound_claim(claim):
            print(f"  * Compound claim detected: \"{claim[:60]}...\"")
            atomic_subclaims = decompose_claim(claim)
            decomposed_map[claim] = atomic_subclaims
            print(f"    -> Decomposed into {len(atomic_subclaims)} atomic sub-claim(s):")
            for idx, sub in enumerate(atomic_subclaims, 1):
                print(f"       {idx}. \"{sub}\"")
            all_atomic_claims.extend(atomic_subclaims)
        else:
            all_atomic_claims.append(claim)

    return {
        "claims": all_atomic_claims,
        "decomposed_claims_map": decomposed_map,
        "has_compound_claims": False  # Decomposition completed
    }


# Node 2: Hybrid Evidence Retrieval Agent
def retrieval_node(state: TrustAgentState) -> Dict[str, Any]:
    claims = state.get("claims", [])
    print(f"\n[Node 2: Retrieval Agent] Searching hybrid evidence for {len(claims)} claim(s)...")
    evidence_map = retrieve_evidence_for_claims(claims)
    for c, evs in evidence_map.items():
        print(f"  - Claim: \"{c[:40]}...\" -> Retrieved {len(evs)} passage(s)")
    return {"evidence_map": evidence_map}


# Parallel Fan-Out Jury Node 1: Data Analyst
def data_analyst_jury_node(state: TrustAgentState) -> Dict[str, Any]:
    claims = state.get("claims", [])
    evidence_map = state.get("evidence_map", {})
    print(f"\n[Delphi Jury Fan-Out] Data Analyst evaluating {len(claims)} claim(s)...")

    verdicts = []
    for claim in claims:
        ev_passages = evidence_map.get(claim, [])
        v = evaluate_as_persona(claim, ev_passages, "DATA_ANALYST")
        verdicts.append(v)
        print(f"  [DATA_ANALYST] -> {v.verdict} ({int(v.confidence * 100)}%) for \"{claim[:35]}...\"")
    return {"data_analyst_verdicts": verdicts}


# Parallel Fan-Out Jury Node 2: Historian
def historian_jury_node(state: TrustAgentState) -> Dict[str, Any]:
    claims = state.get("claims", [])
    evidence_map = state.get("evidence_map", {})
    print(f"\n[Delphi Jury Fan-Out] Historian evaluating {len(claims)} claim(s)...")

    verdicts = []
    for claim in claims:
        ev_passages = evidence_map.get(claim, [])
        v = evaluate_as_persona(claim, ev_passages, "HISTORIAN")
        verdicts.append(v)
        print(f"  [HISTORIAN]    -> {v.verdict} ({int(v.confidence * 100)}%) for \"{claim[:35]}...\"")
    return {"historian_verdicts": verdicts}


# Parallel Fan-Out Jury Node 3: Policy Analyst
def policy_analyst_jury_node(state: TrustAgentState) -> Dict[str, Any]:
    claims = state.get("claims", [])
    evidence_map = state.get("evidence_map", {})
    print(f"\n[Delphi Jury Fan-Out] Policy Analyst evaluating {len(claims)} claim(s)...")

    verdicts = []
    for claim in claims:
        ev_passages = evidence_map.get(claim, [])
        v = evaluate_as_persona(claim, ev_passages, "POLICY_ANALYST")
        verdicts.append(v)
        print(f"  [POLICY_ANALYST] -> {v.verdict} ({int(v.confidence * 100)}%) for \"{claim[:35]}...\"")
    return {"policy_analyst_verdicts": verdicts}


# Fan-In Node 3: Grounded Verifier & Consensus Synthesizer
def verifier_consensus_node(state: TrustAgentState) -> Dict[str, Any]:
    claims = state.get("claims", [])
    evidence_map = state.get("evidence_map", {})
    da_verdicts = {v.claim: v for v in state.get("data_analyst_verdicts", [])}
    hist_verdicts = {v.claim: v for v in state.get("historian_verdicts", [])}
    pa_verdicts = {v.claim: v for v in state.get("policy_analyst_verdicts", [])}

    print(f"\n[Node 3: Verifier & Consensus Node] Synthesizing Delphi Jury deliberations for {len(claims)} claim(s)...")

    verifications = []
    jury_verdicts_map = {}

    for claim in claims:
        passages = evidence_map.get(claim, [])
        jury_list = [
            da_verdicts.get(claim),
            hist_verdicts.get(claim),
            pa_verdicts.get(claim)
        ]
        jury_list = [j for j in jury_list if j is not None]
        jury_verdicts_map[claim] = jury_list

        if len(jury_list) == 3:
            # Multi-agent consensus calculation
            votes = [j.verdict for j in jury_list]
            avg_conf = sum(j.confidence for j in jury_list) / 3.0

            true_votes = votes.count("True")
            false_votes = votes.count("False")

            if true_votes >= 2 and avg_conf >= UNCERTAIN_CONFIDENCE_THRESHOLD:
                final_verdict = "True"
                final_conf = avg_conf
            elif false_votes >= 2 and avg_conf >= UNCERTAIN_CONFIDENCE_THRESHOLD:
                final_verdict = "False"
                final_conf = avg_conf
            else:
                final_verdict = "Uncertain"
                final_conf = min(avg_conf, 0.45)

            # Synthesize Chain-of-Thought across the 3 lenses
            reasoning_summary = (
                f"Delphi Consensus: [Data Analyst: {jury_list[0].verdict} ({int(jury_list[0].confidence*100)}%)] - {jury_list[0].reasoning} | "
                f"[Historian: {jury_list[1].verdict} ({int(jury_list[1].confidence*100)}%)] - {jury_list[1].reasoning} | "
                f"[Policy Analyst: {jury_list[2].verdict} ({int(jury_list[2].confidence*100)}%)] - {jury_list[2].reasoning}"
            )

            result = VerificationResult(
                claim=claim,
                verdict=final_verdict,
                confidence=round(final_conf, 4),
                reasoning=reasoning_summary,
                evidence_refs=passages
            )
        else:
            # Fallback to single verifier if jury incomplete
            result = verify_claim(claim, passages)

        verifications.append(result)
        print(f"  - Synthesized Verdict: {result.verdict} (Confidence: {int(result.confidence * 100)}%) for \"{claim[:35]}...\"")

    return {
        "verifications": verifications,
        "jury_verdicts": jury_verdicts_map
    }


# Node: Logic Aggregator (Phase 10 Extension)
def logic_aggregator_node(state: TrustAgentState) -> Dict[str, Any]:
    verifications = list(state.get("verifications", []))
    decomposed_map = state.get("decomposed_claims_map", {})

    if not decomposed_map:
        return {"verifications": verifications}

    print(f"\n[Node: Logic Aggregator] Applying symbolic propositional logic over {len(decomposed_map)} compound claim(s)...")

    result_by_claim = {v.claim: v for v in verifications}
    compound_verifications = []

    for compound_claim, atomic_subclaims in decomposed_map.items():
        sub_results = [result_by_claim[sub] for sub in atomic_subclaims if sub in result_by_claim]
        if sub_results:
            op = detect_logical_structure(compound_claim)
            agg_result = aggregate_verdicts(sub_results, operator=op, compound_claim=compound_claim)
            compound_verifications.append(agg_result)
            print(f"  * Compound Assertion: \"{compound_claim[:45]}...\"")
            print(f"    Inferred Operator: {op} | Aggregated Verdict: {agg_result.verdict} ({int(agg_result.confidence * 100)}%)")
            print(f"    Logic Trace: {agg_result.logic_trace}")

    # Prioritize compound synthesized results at the top, followed by constituent atomic sub-claims
    all_verifications = compound_verifications + verifications
    return {"verifications": all_verifications}


# Node 4: Explainer Agent
def explainer_node(state: TrustAgentState) -> Dict[str, Any]:
    verifications = state.get("verifications", [])
    decomposed_map = state.get("decomposed_claims_map", {})
    jury_map = state.get("jury_verdicts", {})

    print(f"\n[Node 4: Explainer Agent] Generating final audit report for {len(verifications)} verification(s)...")
    report = generate_report(verifications)

    # Append Neurosymbolic Logic Aggregation Section
    logic_results = [v for v in verifications if v.logic_trace]
    if logic_results:
        logic_section = [
            "\n## [Neurosymbolic Formal Logic Aggregation Trace]\n",
            "Constituent sub-claim verifications were piped into formal propositional logic gates to guarantee deterministic truth bounds:\n"
        ]
        for v in logic_results:
            op = detect_logical_structure(v.claim)
            logic_section.append(f"### Compound Assertion: \"{v.claim}\"")
            logic_section.append(f"- **Formal Operator:** `{op}`")
            logic_section.append(f"- **Symbolic Logic Trace:** `{v.logic_trace}`")
            logic_section.append(f"- **Aggregated Truth Verdict:** **{v.verdict} ({int(v.confidence*100)}%)**")
            logic_section.append(f"- **Formal Reasoning:** {v.reasoning}\n")
        report += "\n" + "\n".join(logic_section)

    # Append Delphi Jury Breakdown Table
    if jury_map:
        jury_section = [
            "\n## [Delphi Multi-Agent Jury Deliberation Matrix]\n",
            "Claims were cross-examined independently by three specialized expert personas:\n",
            "| Claim | Data Analyst | Historian | Policy Analyst | Final Consensus |",
            "|:---|:---:|:---:|:---:|:---:|"
        ]
        for v in verifications:
            claim_text = v.claim
            if claim_text not in jury_map:
                continue
            juries = jury_map.get(claim_text, [])
            da = next((j for j in juries if j.persona == "DATA_ANALYST"), None)
            hi = next((j for j in juries if j.persona == "HISTORIAN"), None)
            pa = next((j for j in juries if j.persona == "POLICY_ANALYST"), None)

            da_str = f"{da.verdict} ({int(da.confidence*100)}%)" if da else "N/A"
            hi_str = f"{hi.verdict} ({int(hi.confidence*100)}%)" if hi else "N/A"
            pa_str = f"{pa.verdict} ({int(pa.confidence*100)}%)" if pa else "N/A"
            short_claim = claim_text[:50] + ("..." if len(claim_text) > 50 else "")

            jury_section.append(f"| \"{short_claim}\" | {da_str} | {hi_str} | {pa_str} | **{v.verdict} ({int(v.confidence*100)}%)** |")

        report += "\n" + "\n".join(jury_section) + "\n"

    # Append LoCal Claim Decomposition Audit Trace
    if decomposed_map:
        decomposition_section = ["\n## [LoCal Claim Decomposition Audit Trace]\n"]
        decomposition_section.append("The system identified compound assertions and split them into atomic propositions for independent verification:\n")
        for parent_claim, children in decomposed_map.items():
            decomposition_section.append(f"### Compound Assertion: \"{parent_claim}\"")
            decomposition_section.append("Atomic Sub-Claims:")
            for idx, child in enumerate(children, 1):
                decomposition_section.append(f"  {idx}. {child}")
            decomposition_section.append("")
        report += "\n" + "\n".join(decomposition_section)

    return {"final_report": report}


def route_claims(state: TrustAgentState) -> str:
    """
    Conditional routing function:
    Routes to 'decomposer_agent' if compound claims are present,
    otherwise proceeds directly to 'retrieval_agent'.
    """
    if state.get("has_compound_claims", False):
        return "decomposer_agent"
    return "retrieval_agent"


def build_extended_graph(
    enable_decomposer: bool = True,
    enable_jury: bool = True,
    enable_logic_aggregator: bool = True
):
    """
    Constructs the Full Extended TRUST Agents LangGraph workflow with:
    1. Dynamic Claim Decomposition (Phase 8, toggleable via enable_decomposer)
    2. Hybrid BM25 + FAISS Retrieval
    3. Parallel Fan-Out Delphi Multi-Agent Jury (Phase 9, toggleable via enable_jury)
    4. Consensus Verification & Synthesis
    5. Neurosymbolic Formal Logic Aggregator (Phase 10, toggleable via enable_logic_aggregator)
    6. Comprehensive Auditable Markdown Explainer
    """
    from graph.baseline_graph import verifier_node as baseline_verifier_node

    workflow = StateGraph(TrustAgentState)

    # Core nodes
    workflow.add_node("claim_extractor", claim_extractor_node)
    workflow.add_node("retrieval_agent", retrieval_node)

    # 1. Start -> Extraction -> (Decomposition or direct Retrieval)
    workflow.add_edge(START, "claim_extractor")
    if enable_decomposer:
        workflow.add_node("decomposer_agent", decomposer_node)
        workflow.add_conditional_edges(
            "claim_extractor",
            route_claims,
            {
                "decomposer_agent": "decomposer_agent",
                "retrieval_agent": "retrieval_agent"
            }
        )
        workflow.add_edge("decomposer_agent", "retrieval_agent")
    else:
        workflow.add_edge("claim_extractor", "retrieval_agent")

    # 2. Retrieval -> (Delphi Jury Fan-Out or Single Verifier)
    if enable_jury:
        workflow.add_node("data_analyst_jury", data_analyst_jury_node)
        workflow.add_node("historian_jury", historian_jury_node)
        workflow.add_node("policy_analyst_jury", policy_analyst_jury_node)
        workflow.add_node("verifier_agent", verifier_consensus_node)

        # Parallel Fan-Out
        workflow.add_edge("retrieval_agent", "data_analyst_jury")
        workflow.add_edge("retrieval_agent", "historian_jury")
        workflow.add_edge("retrieval_agent", "policy_analyst_jury")

        # Parallel Fan-In
        workflow.add_edge("data_analyst_jury", "verifier_agent")
        workflow.add_edge("historian_jury", "verifier_agent")
        workflow.add_edge("policy_analyst_jury", "verifier_agent")
    else:
        workflow.add_node("verifier_agent", baseline_verifier_node)
        workflow.add_edge("retrieval_agent", "verifier_agent")

    # 3. Verifier -> (Logic Aggregator or Explainer)
    last_node = "verifier_agent"
    if enable_logic_aggregator:
        workflow.add_node("logic_aggregator", logic_aggregator_node)
        workflow.add_edge("verifier_agent", "logic_aggregator")
        last_node = "logic_aggregator"

    # 4. Explainer -> END
    workflow.add_node("explainer_agent", explainer_node)
    workflow.add_edge(last_node, "explainer_agent")
    workflow.add_edge("explainer_agent", END)

    return workflow.compile()


if __name__ == "__main__":
    print("=== Testing Extended TRUST Agents Graph with Logic Aggregator (Phase 10) ===")
    app = build_extended_graph()

    test_article = (
        "Biden's infrastructure bill passed in 2021, creating 2 million jobs and reducing carbon emissions by 15%."
    )

    result = app.invoke({"article": test_article})
    print("\n" + "=" * 60)
    print("EXTENDED GRAPH FINAL AUDIT REPORT WITH LOGIC AGGREGATOR:")
    print("=" * 60)
    print(result["final_report"])
