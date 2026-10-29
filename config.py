import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LIAR_DIR = DATA_DIR / "liar"
CORPUS_DIR = DATA_DIR / "corpus"
PROMPTS_DIR = BASE_DIR / "prompts"
EVAL_DIR = BASE_DIR / "evaluation"

# Ollama / LLM Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Retrieval Settings
SPARSE_BM25_WEIGHT = 0.4
DENSE_FAISS_WEIGHT = 0.6
TOP_K_EVIDENCE = 5
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Verification Thresholds
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.50
