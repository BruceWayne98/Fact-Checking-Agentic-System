from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TypedDict

@dataclass
class EvidencePassage:
    id: str
    title: str
    text: str
    source: str
    hybrid_score: float

@dataclass
class VerificationResult:
    claim: str
    verdict: str                  # "True", "False", or "Uncertain"
    confidence: float             # Calibrated confidence between 0.0 and 1.0
    reasoning: str                # Chain-of-thought justification anchored ONLY in evidence
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    logic_trace: Optional[str] = None

class TrustAgentState(TypedDict, total=False):
    """
    Shared LangGraph State across the TRUST Agents fact-verification workflow.
    """
    article: str                                      # Raw input news statement or paragraph
    claims: List[str]                                 # Discrete claims extracted by Agent 1
    evidence_map: Dict[str, List[Dict[str, Any]]]     # claim -> retrieved evidence passages (Agent 2)
    verifications: List[VerificationResult]           # Verification results per claim (Agent 3)
    final_report: str                                 # Human-readable audit report in Markdown (Agent 4)
