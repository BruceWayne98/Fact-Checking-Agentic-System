import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 lexical indexing."""
    return re.findall(r"\w+", text.lower())

class BM25Searcher:
    """
    BM25 Sparse Lexical Retriever.
    Provides term-frequency / inverse-document-frequency exact token matching.
    """
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.tokenized_corpus = [tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Queries BM25 and returns the top_k matching documents with their BM25 scores.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Rank by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score > 0.0:  # Only include non-zero lexical matches
                results.append((self.documents[idx], score))
        return results

if __name__ == "__main__":
    docs = [
        {"id": "doc_1", "text": "The US unemployment rate reached 3.5% in January 2024."},
        {"id": "doc_2", "text": "Natural gas production surged in Texas and Pennsylvania."},
        {"id": "doc_3", "text": "Coal mining experienced major job declines after 2005."}
    ]
    bm25 = BM25Searcher(docs)
    res = bm25.search("unemployment rate in 2024", top_k=2)
    print("BM25 Results:")
    for doc, score in res:
        print(f"  [{score:.4f}] {doc['text']}")
