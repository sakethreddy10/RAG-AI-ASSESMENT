# ==============================================================================
# rag_service.py — RAG Orchestration (used by the Web UI / FastAPI routes)
#
# This service is the bridge between the FastAPI web server and the
# ChromaDB + Gemini pipeline.
#
# FLOW per user message:
#   1. Receive user question
#   2. Search ChromaDB for the most relevant document chunks
#   3. Build a grounded prompt with citations
#   4. Ask Gemini to generate an answer
#   5. Return the answer + metadata (tokens, number of chunks used)
#
# NOTE: Run src/ingest.py first to build the vector database before starting
#       the web server.
# ==============================================================================

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.vectorstore.chroma_store import VectorStore
from app.services.llm_service import LLMService
from src.config import SIMILARITY_THRESHOLD


class RAGService:
    def __init__(self):
        """
        Initialize the vector store and the LLM service.
        The vector store opens the ChromaDB database from disk.
        """
        self.vector_store = VectorStore()
        self.llm_service  = LLMService()

    # ==========================================================================
    # NOTE: index_documents() has been REMOVED.
    #
    # Previously, the server auto-indexed docs.json at startup.
    # Now, indexing is a separate step (run src/ingest.py).
    # This keeps startup fast and avoids re-embedding on every server restart.
    # ==========================================================================

    def query(self, user_question: str, conversation_history: list[dict]) -> dict:
        """
        Full RAG query pipeline for the web UI.

        Steps:
          1. Search ChromaDB for relevant chunks
          2. Filter out low-relevance chunks
          3. Build a grounded prompt with inline source citations
          4. Ask Gemini to generate an answer
          5. Return result dict

        Args:
            user_question        : The user's message from the chat UI
            conversation_history : Previous messages [{"role": "user"/"assistant", "content": "..."}]

        Returns:
            {
                "reply"          : "Answer text with citations...",
                "tokensUsed"     : 312,
                "retrievedChunks": 3
            }
        """
        print(f"\n🔍 Query: {user_question}")

        # ── Step 1: Search ChromaDB ────────────────────────────────────────────
        # The VectorStore.search() method passes the raw text to ChromaDB,
        # which embeds it and returns the closest matching chunks.
        print("🔎 Searching ChromaDB...")
        search_results = self.vector_store.search(user_question, top_k=4)

        # ── Step 2: Filter by similarity threshold ─────────────────────────────
        print("📊 Similarity scores:")
        relevant_chunks = []

        for result in search_results:
            score  = result["similarity"]
            source = result.get("source", result.get("title", "Unknown"))
            page   = result.get("page", "?")
            passed = "✅" if score >= SIMILARITY_THRESHOLD else "❌ (below threshold)"
            print(f"  - '{source}' p.{page}: {score:.4f} {passed}")

            if score >= SIMILARITY_THRESHOLD:
                relevant_chunks.append(result)

        # ── Step 3: Fallback if nothing is relevant ────────────────────────────
        if not relevant_chunks:
            print("⚠️  No relevant chunks found. Returning fallback response.")
            return {
                "reply": (
                    "I am sorry, the provided documents do not contain "
                    "the answer to your question. Please try rephrasing "
                    "or contact support."
                ),
                "tokensUsed":      None,
                "retrievedChunks": 0
            }

        # ── Step 4: Build context with citations ───────────────────────────────
        # Each chunk is labelled with its file name and page number.
        # The LLM is instructed to reference these labels in its answer.
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, start=1):
            source = chunk.get("source", chunk.get("title", "Unknown"))
            page   = chunk.get("page", "?")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        print(f"📋 Using {len(relevant_chunks)} chunk(s) as context.")

        # ── Step 5: Ask Gemini ─────────────────────────────────────────────────
        print("🤖 Asking Gemini...")
        answer, tokens_used = self.llm_service.generate_answer(
            context=context,
            conversation_history=conversation_history,
            user_question=user_question
        )

        if tokens_used:
            print(f"🪙 Tokens used: {tokens_used}")

        return {
            "reply":           answer,
            "tokensUsed":      tokens_used,
            "retrievedChunks": len(relevant_chunks)
        }
