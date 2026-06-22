# ==============================================================================
# query.py — Query Pipeline (Retrieval + Answer Generation)
#
# This module does two things:
#   1. RETRIEVAL: Searches the ChromaDB vector database for the most relevant
#      document chunks that match the user's question.
#   2. GENERATION: Feeds those chunks as context to Gemini and gets a grounded,
#      cited answer back.
#
# IMPORTANT: Run ingest.py FIRST to build the database before using this.
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import google.generativeai as genai
import chromadb
from src.embedding_helper import GeminiEmbeddingFunction

from src.config import (
    GEMINI_API_KEY, LLM_MODEL, EMBEDDING_MODEL,
    DB_DIR, COLLECTION_NAME,
    TOP_K, SIMILARITY_THRESHOLD,
    TEMPERATURE, MAX_OUTPUT_TOKENS
)

# Configure the Gemini API client once, at import time
genai.configure(api_key=GEMINI_API_KEY)


# ==============================================================================
# LOAD THE DATABASE
# We open the ChromaDB collection once and reuse it across all queries.
# This avoids reconnecting on every single question.
# ==============================================================================

def load_collection():
    """
    Opens the existing ChromaDB database from disk and returns the collection.
    Raises a clear error if the database hasn't been built yet (ingest.py not run).
    """
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(
            f"❌ Database not found at '{DB_DIR}/'.\n"
            "   Please run:  python src/ingest.py\n"
            "   to build the database first."
        )

    client = chromadb.PersistentClient(path=DB_DIR)

    embedding_fn = GeminiEmbeddingFunction(
        api_key=GEMINI_API_KEY,
        model_name=EMBEDDING_MODEL
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    except Exception:
        raise RuntimeError(
            f"❌ Collection '{COLLECTION_NAME}' not found in the database.\n"
            "   Please run:  python src/ingest.py"
        )

    return collection


# ==============================================================================
# STEP 1: RETRIEVAL
# Embed the user's question and find the closest matching document chunks.
# ==============================================================================

def retrieve_relevant_chunks(collection, user_question: str) -> list[dict]:
    """
    Searches ChromaDB for the top-k most similar chunks to the user's question.

    Returns a list of dicts like:
    [
      { "text": "...", "source": "report.pdf", "page": 3, "score": 0.87 },
      ...
    ]

    Chunks below SIMILARITY_THRESHOLD are filtered out to avoid hallucination.
    """
    # Query the database — ChromaDB embeds the question automatically using
    # the same embedding function we set up in load_collection()
    results = collection.query(
        query_texts=[user_question],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    # ChromaDB returns a "distance" (lower = more similar for cosine distance).
    # We convert it to a similarity score: similarity = 1 - distance
    raw_docs      = results["documents"][0]      # List of chunk texts
    raw_metadatas = results["metadatas"][0]      # List of metadata dicts
    raw_distances = results["distances"][0]      # List of cosine distances

    relevant_chunks = []

    for text, metadata, distance in zip(raw_docs, raw_metadatas, raw_distances):
        similarity = 1.0 - distance  # Convert distance → similarity (0 to 1)

        if similarity >= SIMILARITY_THRESHOLD:
            relevant_chunks.append({
                "text":     text,
                "source":   metadata.get("source", "Unknown"),
                "page":     metadata.get("page", "?"),
                "score":    round(similarity, 4)
            })

    return relevant_chunks


# ==============================================================================
# STEP 2: PROMPT BUILDING
# Format the retrieved chunks into a clean context block for the LLM.
# ==============================================================================

def build_prompt(user_question: str, chunks: list[dict]) -> str:
    """
    Constructs the full prompt to send to the LLM.

    Structure:
      [System instructions to prevent hallucination]
      [Chunk 1 with citation label]
      [Chunk 2 with citation label]
      ...
      [User's question]

    By labelling each chunk with its source file and page number, we instruct
    the model to reference those labels in its answer.
    """

    # Format each retrieved chunk as a clearly labelled block
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        citation = f"Source: {chunk['source']}, Page: {chunk['page']}"
        context_blocks.append(
            f"[{citation}]\n"
            f"{chunk['text']}"
        )

    # Join all blocks with a clear separator
    context_text = "\n\n---\n\n".join(context_blocks)

    # The system instruction strictly grounds the model in the retrieved context
    system_instruction = (
        "You are a professional document Q&A assistant.\n"
        "Answer the user's question using ONLY the document context provided below.\n"
        "After every fact or statement, cite the source file and page in parentheses, "
        "like this: (report.pdf, Page 3).\n"
        "If the answer is not found in the context, say exactly:\n"
        "'I am sorry, the provided documents do not contain the answer to your question.'\n"
        "Do NOT use external knowledge or make up facts."
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"DOCUMENT CONTEXT:\n"
        f"{context_text}\n\n"
        f"USER QUESTION: {user_question}\n\n"
        f"ANSWER:"
    )

    return prompt


# ==============================================================================
# STEP 3: ANSWER GENERATION
# Send the prompt to Gemini and return the response.
# ==============================================================================

def generate_answer(prompt: str) -> str:
    """
    Calls the Gemini LLM with the grounded prompt and returns the answer text.
    Low temperature (0.2) makes the response factual rather than creative.
    """
    model = genai.GenerativeModel(LLM_MODEL)

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS
        )
    )

    return response.text.strip()


# ==============================================================================
# PUBLIC API — The main function that other modules call
# ==============================================================================

def ask(collection, user_question: str) -> dict:
    """
    Full RAG pipeline for a single question.

    Args:
        collection : The ChromaDB collection (loaded once, passed in)
        user_question : The user's natural language question

    Returns a dict:
    {
        "answer"   : "The net revenue grew by 14%... (annual_report.pdf, Page 12)",
        "citations": ["Source: annual_report.pdf, Page: 12", ...],
        "chunks_used": 3
    }
    """

    # --- Retrieval ---
    chunks = retrieve_relevant_chunks(collection, user_question)

    # If nothing relevant was found, return a safe fallback without calling the LLM
    if not chunks:
        return {
            "answer": (
                "I am sorry, the provided documents do not contain "
                "the answer to your question."
            ),
            "citations": [],
            "chunks_used": 0
        }

    # --- Prompt Building ---
    prompt = build_prompt(user_question, chunks)

    # --- Generation ---
    answer = generate_answer(prompt)

    # Collect the unique citations from all retrieved chunks
    citations = [f"Source: {c['source']}, Page: {c['page']}" for c in chunks]

    return {
        "answer":      answer,
        "citations":   citations,
        "chunks_used": len(chunks)
    }
