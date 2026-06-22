# ==============================================================================
# config.py — Central Configuration File
#
# All tunable parameters live here. Change them once and they affect
# the whole project. No magic numbers scattered across files.
# ==============================================================================

import os
import sys
from dotenv import load_dotenv

# Reconfigure stdout/stderr to support UTF-8 characters (like emojis) on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Fallback for old Python versions where reconfigure is not available
        pass

# Load the .env file so GEMINI_API_KEY is available
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "❌ GEMINI_API_KEY is missing. "
        "Copy .env.example to .env and add your key."
    )

# ── Model Names ───────────────────────────────────────────────────────────────

# The LLM used to generate answers
LLM_MODEL = "models/gemini-2.5-flash"

# The embedding model — must be the SAME model for both ingestion and querying
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ── File Paths ────────────────────────────────────────────────────────────────

# Folder where you place your PDF and DOCX documents
DATA_DIR = "./data"

# Folder where ChromaDB saves its database files (created automatically)
DB_DIR = "./db"

# Name of the ChromaDB collection (like a table name)
COLLECTION_NAME = "document_knowledge_base"

# ── Chunking Settings ─────────────────────────────────────────────────────────

# Each chunk is at most this many characters long
CHUNK_SIZE = 1000

# How many characters the next chunk overlaps with the previous one.
# This prevents a key sentence from being cut in half between two chunks.
CHUNK_OVERLAP = 200

# ── Retrieval Settings ────────────────────────────────────────────────────────

# How many chunks to retrieve per question (top-k nearest neighbours)
TOP_K = 4

# Any chunk whose similarity score is below this is ignored (0–1 scale)
SIMILARITY_THRESHOLD = 0.4

# ── LLM Generation Settings ───────────────────────────────────────────────────

# Low temperature = factual and consistent answers (less creative)
TEMPERATURE = 0.2

# Maximum number of tokens the LLM can generate in its response
MAX_OUTPUT_TOKENS = 800
