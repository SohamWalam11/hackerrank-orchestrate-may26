"""
Configuration module for the AI Support Triage Agent.

Centralizes all paths, model parameters, and environment variable access.
Paths are computed relative to this file's location so the project works
regardless of the current working directory.
"""

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths — resolved relative to the project root (parent of code/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_ROOT = _PROJECT_ROOT / "data"
TICKETS_DIR = _PROJECT_ROOT / "support_tickets"
INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV = TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_ANTHROPIC = "claude-sonnet-4-20250514"
MODEL_OPENROUTER = "anthropic/claude-sonnet-4"
MODEL_OPENAI = "gpt-4o"

# Fallback models in order of preference for OpenRouter
OPENROUTER_FALLBACK_MODELS = [
    "anthropic/claude-sonnet-4",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-3.7-sonnet",
]

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
TOP_K_CHUNKS = 8          # number of chunks returned per query
CHUNK_SIZE = 800          # characters per chunk (larger = more context)
CHUNK_OVERLAP = 200       # overlap between consecutive chunks
MIN_CHUNK_LENGTH = 50     # discard chunks shorter than this

# BM25 / TF-IDF fusion weight — higher means BM25 is weighted more
BM25_WEIGHT = 0.6
TFIDF_WEIGHT = 0.4

# ---------------------------------------------------------------------------
# Escalation thresholds
# ---------------------------------------------------------------------------
RISK_THRESHOLD = 0.55         # risk_score above this → escalate
RETRIEVAL_MIN_SCORE = 0.08    # best-chunk similarity below this → escalate
CONFIDENCE_THRESHOLD = 0.3    # company detection confidence below this → flag

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# API keys — must be set as environment variables; never hardcode
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Auto-detect provider from key format
def _detect_provider():
    key = ANTHROPIC_API_KEY
    if key.startswith("sk-or-v1-"):
        return "openrouter"
    elif key.startswith("sk-ant-"):
        return "anthropic"
    elif key.startswith("sk-proj-") or key.startswith("sk-"):
        return "openai"
    return "none"

PROVIDER = _detect_provider()
if PROVIDER == "openrouter":
    MODEL = MODEL_OPENROUTER
elif PROVIDER == "openai":
    MODEL = MODEL_OPENAI
else:
    MODEL = MODEL_ANTHROPIC

# ---------------------------------------------------------------------------
# Allowed output values (from problem_statement.md)
# ---------------------------------------------------------------------------
ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}
