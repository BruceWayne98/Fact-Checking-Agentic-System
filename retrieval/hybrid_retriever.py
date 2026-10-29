import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from retrieval.bm25_index import BM25Searcher
from retrieval.faiss_index import FAISSSearcher
from data.build_corpus import load_corpus
from config import SPARSE_BM25_WEIGHT, DENSE_FAISS_WEIGHT, TOP_K_EVIDENCE

class HybridRetriever:
    """
    Agent 2: Hybrid Evidence Retrieval Agent.
    Combines BM25 (sparse lexical) and FAISS (dense semantic) scores
    using Reciprocal Rank Fusion / Min-Max normalized weighted combination.
    """
    def __init__(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        bm25_weight: float = SPARSE_BM25_WEIGHT,
        faiss_weight: float = DENSE_FAISS_WEIGHT
    ):
        self.documents = documents if documents is not None else load_corpus()
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        
        print(f"Initializing Hybrid Retriever with {len(self.documents)} documents...")
        self.bm25 = BM25Searcher(self.documents)
        self.faiss = FAISSSearcher(self.documents)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_EVIDENCE
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval for a given factual claim query.
        Returns a list of top_k evidence passages with hybrid ranking scores and source citations.
        """
        # Fetch candidate pools from both retrievers (fetching 2x top_k to blend)
        candidate_k = max(top_k * 3, 10)
        bm25_results = self.bm25.search(query, top_k=candidate_k)
        faiss_results = self.faiss.search(query, top_k=candidate_k)

        # Normalize BM25 scores to [0, 1] range
        bm25_score_map = {}
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            min_bm25 = min(score for _, score in bm25_results)
            span = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            for doc, score in bm25_results:
                norm_score = (score - min_bm25) / span if max_bm25 > min_bm25 else 1.0
                bm25_score_map[doc["id"]] = (doc, norm_score)

        # Normalize FAISS cosine scores to [0, 1] range
        faiss_score_map = {}
        if faiss_results:
            max_faiss = max(score for _, score in faiss_results)
            min_faiss = min(score for _, score in faiss_results)
            span_f = max_faiss - min_faiss if max_faiss > min_faiss else 1.0
            for doc, score in faiss_results:
                # Cosine similarity typically ranges from -1 to 1; rescale relative to candidate set
                norm_f = (score - min_faiss) / span_f if max_faiss > min_faiss else score
                faiss_score_map[doc["id"]] = (doc, max(norm_f, 0.0))

        # Combine candidates
        all_doc_ids = set(bm25_score_map.keys()) | set(faiss_score_map.keys())
        combined_scores = []

        for doc_id in all_doc_ids:
            doc = bm25_score_map[doc_id][0] if doc_id in bm25_score_map else faiss_score_map[doc_id][0]
            s_bm25 = bm25_score_map[doc_id][1] if doc_id in bm25_score_map else 0.0
            s_faiss = faiss_score_map[doc_id][1] if doc_id in faiss_score_map else 0.0

            hybrid_score = (self.bm25_weight * s_bm25) + (self.faiss_weight * s_faiss)
            combined_scores.append({
                "id": doc["id"],
                "title": doc.get("title", ""),
                "text": doc["text"],
                "source": doc.get("source", "Unknown"),
                "hybrid_score": round(hybrid_score, 4),
                "bm25_norm": round(s_bm25, 4),
                "faiss_norm": round(s_faiss, 4)
            })

        # Rank descending by combined hybrid score
        combined_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return combined_scores[:top_k]

if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "The US unemployment rate fell to 3.5% in January 2024",
        "American jobless figures at the start of 2024",  # Semantic paraphrase (tests FAISS dense matching)
        "When did the decline of coal start under President George W. Bush?"
    ]

    print("\n=== Testing Agent 2: Hybrid Retriever (BM25 + FAISS) ===")
    for q in test_queries:
        print(f"\nQuery: \"{q}\"")
        results = retriever.retrieve(q, top_k=3)
        for rank, res in enumerate(results, 1):
            print(f"  [{rank}] Score: {res['hybrid_score']} (BM25: {res['bm25_norm']}, FAISS: {res['faiss_norm']})")
            print(f"      Source: {res['source']}")
            print(f"      Text: {res['text'][:120]}...")
