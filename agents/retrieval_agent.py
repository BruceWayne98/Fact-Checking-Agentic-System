import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from retrieval.hybrid_retriever import HybridRetriever
from config import TOP_K_EVIDENCE

# Singleton retriever instance for agent reuse
_RETRIEVER_INSTANCE: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = HybridRetriever()
    return _RETRIEVER_INSTANCE

def retrieve_evidence_for_claims(
    claims: List[str],
    top_k: int = TOP_K_EVIDENCE
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Agent 2 interface: accepts a list of extracted claim strings
    and maps each claim to its top ranked evidence passages with source citations.
    """
    retriever = get_retriever()
    claim_evidence_map = {}

    for claim in claims:
        passages = retriever.retrieve(claim, top_k=top_k)
        claim_evidence_map[claim] = passages

    return claim_evidence_map

if __name__ == "__main__":
    test_claims = [
        "The US unemployment rate was 3.5% in January 2024",
        "American jobless figures at the start of 2024",
        "Coal production began declining under George W. Bush due to natural gas"
    ]
    evidence_map = retrieve_evidence_for_claims(test_claims, top_k=2)
    for c, evs in evidence_map.items():
        print(f"\nClaim: {c}")
        for ev in evs:
            print(f"  -> [{ev['hybrid_score']}] ({ev['source']}): {ev['text'][:100]}...")
