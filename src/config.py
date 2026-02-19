"""
Olympus Graph – Central Configuration
Loads environment variables and provides typed access to all settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────
load_dotenv()

# ── Project Root ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Neo4j ─────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── OpenAI ────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── Groq / LLM Provider ───────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Embeddings ────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# ── Data Paths ────────────────────────────────────────
DATA_RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
MODEL_CHECKPOINT_DIR = PROJECT_ROOT / os.getenv(
    "MODEL_CHECKPOINT_DIR", "data/processed/checkpoints"
)

# ── GNN Hyperparameters ──────────────────────────────
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 2
GNN_DROPOUT = 0.3
GNN_LEARNING_RATE = 1e-3
GNN_EPOCHS = 100
GNN_BATCH_SIZE = 1024

# ── Training Splits ──────────────────────────────────
TRAIN_YEARS = range(1896, 2016)  # 1896–2015 inclusive
TEST_YEAR = 2016

# ── Recall@K ─────────────────────────────────────────
RECALL_K = 3
