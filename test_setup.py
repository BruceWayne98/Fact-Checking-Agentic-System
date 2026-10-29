import os
import sys

def check_environment():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    errors = []

    # Check LangGraph
    try:
        from langgraph.graph import StateGraph
        print("[OK] LangGraph")
    except Exception as e:
        errors.append(f"LangGraph error: {e}")

    # Check spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print(f"[OK] spaCy ({nlp.meta['name']})")
    except Exception as e:
        errors.append(f"spaCy / model error: {e}")

    # Check BM25
    try:
        from rank_bm25 import BM25Okapi
        print("[OK] rank-bm25")
    except Exception as e:
        errors.append(f"rank-bm25 error: {e}")

    # Check FAISS
    try:
        import faiss
        print(f"[OK] faiss-cpu (v{faiss.__version__})")
    except Exception as e:
        errors.append(f"faiss-cpu error: {e}")

    # Check Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("[OK] sentence-transformers")
    except Exception as e:
        errors.append(f"sentence-transformers error: {e}")

    # Check Datasets & Scikit-learn
    try:
        import datasets
        import sklearn
        print("[OK] datasets & scikit-learn")
    except Exception as e:
        errors.append(f"datasets/scikit-learn error: {e}")

    # Check Ollama Python package
    try:
        import ollama
        print("[OK] ollama client library")
    except Exception as e:
        errors.append(f"ollama python client error: {e}")

    if errors:
        print("\n--- ISSUES DETECTED ---")
        for err in errors:
            print("  *", err)
        return False
    else:
        print("\nAll core dependencies verified successfully!")
        return True

if __name__ == "__main__":
    check_environment()
