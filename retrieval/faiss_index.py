import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

class FAISSSearcher:
    """
    FAISS Dense Semantic Retriever.
    Uses SentenceTransformer dense embeddings and Inner Product (Cosine similarity)
    to perform approximate / exact nearest-neighbor search for semantic paraphrases.
    """
    def __init__(self, documents: List[Dict[str, Any]], model_name: str = EMBEDDING_MODEL_NAME):
        self.documents = documents
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        # Extract texts and compute embeddings
        texts = [doc["text"] for doc in documents]
        print(f"Embedding {len(texts)} corpus documents into vector space...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Normalize embeddings to unit length for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.dimension = embeddings.shape[1]

        # Build Flat Inner Product Index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        print(f"[OK] FAISS Index built with {self.index.ntotal} vectors of dimension {self.dimension}.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Embeds the query and queries FAISS for the top_k cosine-similar documents.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx != -1:
                score = float(scores[0][rank])
                results.append((self.documents[idx], score))
        return results

if __name__ == "__main__":
    docs = [
        {"id": "doc_1", "text": "The US unemployment rate reached 3.5% in January 2024."},
        {"id": "doc_2", "text": "Natural gas production surged in Texas and Pennsylvania."},
        {"id": "doc_3", "text": "Coal mining experienced major job declines after 2005."}
    ]
    faiss_searcher = FAISSSearcher(docs)
    res = faiss_searcher.search("American jobless numbers early this year", top_k=2)
    print("FAISS Semantic Results:")
    for doc, score in res:
        print(f"  [Cosine: {score:.4f}] {doc['text']}")
