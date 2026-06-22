# ==============================================================================
# chroma_store.py — ChromaDB Vector Store Wrapper
#
# This replaces the old custom NumPy implementation.
# Now we use the real chromadb library which handles:
#   - Embedding generation (via the Gemini embedding function)
#   - Cosine similarity search (HNSW index, fast even with large datasets)
#   - Disk persistence (the database survives server restarts)
#
# This class is used by the FastAPI web server (app/services/rag_service.py).
# For the CLI, query.py talks to ChromaDB directly.
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import chromadb
from src.embedding_helper import GeminiEmbeddingFunction

from src.config import GEMINI_API_KEY, EMBEDDING_MODEL, DB_DIR, COLLECTION_NAME


class VectorStore:
    """
    A thin wrapper around ChromaDB.
    Provides the same simple interface (search, is_empty) that rag_service.py expects.
    """

    def __init__(self):
        """
        Open the ChromaDB database from disk.
        If the database doesn't exist yet, it will be created as an empty store.
        """
        os.makedirs(DB_DIR, exist_ok=True)

        # PersistentClient reads/writes data to the DB_DIR folder on disk
        self.client = chromadb.PersistentClient(path=DB_DIR)

        self.embedding_fn = GeminiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL
        )

        # Get or create the collection (like a table in a regular database)
        # "hnsw:space": "cosine" means we measure similarity using cosine distance
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        count = self.collection.count()
        print(f"📦 VectorStore ready. {'Loaded ' + str(count) + ' chunks from ChromaDB.' if count > 0 else 'Empty (run ingest.py first).'}")

    def search(self, user_question: str, top_k: int = 4) -> list[dict]:
        """
        Find the top-k document chunks most similar to the user's question.

        Unlike the old NumPy version which received a pre-computed embedding vector,
        this version receives the raw question text. ChromaDB embeds it automatically
        using the same Gemini embedding function.

        Returns a list of dicts:
        [
            {
                "text":       "The company's revenue grew by 14%...",
                "title":      "annual_report.pdf",   # source file name
                "source":     "annual_report.pdf",
                "page":       12,
                "similarity": 0.87
            },
            ...
        ]
        """
        if self.is_empty():
            return []

        results = self.collection.query(
            query_texts=[user_question],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        # Convert ChromaDB results into our standard dict format
        chunks = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite)
            # We convert to similarity (1 = identical, -1 = opposite)
            similarity = 1.0 - distance

            chunks.append({
                "text":       text,
                "title":      metadata.get("source", "Unknown"),  # for rag_service.py compatibility
                "source":     metadata.get("source", "Unknown"),
                "page":       metadata.get("page", "?"),
                "similarity": round(similarity, 4)
            })

        return chunks

    def is_empty(self) -> bool:
        """Returns True if no documents have been indexed yet."""
        return self.collection.count() == 0
