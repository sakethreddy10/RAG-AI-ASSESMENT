# ==============================================================================
# Embedding Service - Uses Gemini API to convert text into vector embeddings
#
# What is an embedding?
# - A list of numbers (floats) that represents the "meaning" of text
# - Similar meanings = similar vectors
# - This is how we do semantic search (not keyword matching!)
#
# Example:
# "How to reset password?" → [0.12, -0.45, 0.78, ...]  (768 numbers)
# "Password recovery steps" → [0.11, -0.43, 0.75, ...]  (similar!)
# "Today's weather in Paris" → [0.91, 0.23, -0.12, ...]  (very different!)
# ==============================================================================

import google.generativeai as genai
import os


class EmbeddingService:
    def __init__(self):
        """
        Initialize the Gemini embedding client.
        We use the 'text-embedding-004' model which creates 768-dimensional vectors.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables!")

        genai.configure(api_key=api_key)

        # Gemini's embedding model
        self.model_name = "models/text-embedding-004"
        print(f"🔢 Embedding Service ready using: {self.model_name}")

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single piece of text into a vector embedding.

        Args:
            text: The text to embed (a document chunk or a user query)

        Returns:
            A list of floats representing the text's meaning
        """
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document"  # Optimized for document storage
        )
        return result["embedding"]

    def embed_query(self, query: str) -> list[float]:
        """
        Convert a user's search query into a vector embedding.
        Uses "retrieval_query" task type (slightly different optimization than document embedding).

        Args:
            query: The user's question

        Returns:
            A list of floats representing the query's meaning
        """
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"  # Optimized for search queries
        )
        return result["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts at once (used during indexing).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i, text in enumerate(texts):
            print(f"  📊 Embedding chunk {i+1}/{len(texts)}...")
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
