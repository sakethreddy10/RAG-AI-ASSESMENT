# ==============================================================================
# embedding_helper.py — Custom Gemini Embedding Function
#
# Workaround for the chromadb bug:
# "ValueError: ClientOptions does not accept an option 'headers'"
# This happens in some versions of google-generativeai / google-api-core.
# ==============================================================================

import google.generativeai as genai
from chromadb import EmbeddingFunction, Documents, Embeddings

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    A custom ChromaDB embedding function that uses google-generativeai
    directly without passing client_options headers that cause library conflicts.
    """
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name
        # Configure genai client directly
        genai.configure(api_key=self.api_key)

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embeds a list of document strings using Gemini's text-embedding-004 model.
        """
        # Ensure input is a list of strings
        if not all(isinstance(item, str) for item in input):
            raise ValueError("Gemini embedding function only supports text documents.")

        embeddings_list = []
        for text in input:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings_list.append(response["embedding"])

        return embeddings_list
