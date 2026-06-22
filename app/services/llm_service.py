# ==============================================================================
# llm_service.py — Gemini LLM Answer Generation
#
# This service takes:
#   - Retrieved context chunks (with source citations)
#   - Conversation history
#   - The user's question
#
# And asks Gemini to write a grounded, cited answer.
# ==============================================================================

import google.generativeai as genai
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import GEMINI_API_KEY, LLM_MODEL, TEMPERATURE, MAX_OUTPUT_TOKENS


class LLMService:
    def __init__(self):
        """
        Configure the Gemini client and set up the generative model.
        """
        if not GEMINI_API_KEY:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables!")

        genai.configure(api_key=GEMINI_API_KEY)

        self.model = genai.GenerativeModel(LLM_MODEL)
        print(f"🤖 LLM Service ready using: {LLM_MODEL}")

    def generate_answer(
        self,
        context: str,
        conversation_history: list[dict],
        user_question: str
    ) -> tuple[str, int]:
        """
        Build the prompt and generate a grounded answer using Gemini.

        The prompt structure is:
          [System: strict grounding rules + citation instructions]
          [Context: retrieved chunks with source labels]
          [History: last few messages so Gemini can handle follow-ups]
          [Question: what the user just asked]

        Args:
            context              : Retrieved chunks formatted with citation labels
            conversation_history : List of {"role": "user"/"assistant", "content": "..."}
            user_question        : The user's current question

        Returns:
            Tuple of (answer_text: str, tokens_used: int | None)
        """

        # Format conversation history as readable text
        if conversation_history:
            history_lines = []
            for message in conversation_history:
                role = "User" if message["role"] == "user" else "Assistant"
                history_lines.append(f"{role}: {message['content']}")
            history_text = "\n".join(history_lines)
        else:
            history_text = "No previous messages."

        # Build the full prompt
        # The system instruction tells Gemini exactly how to behave:
        #   - Only use the provided context (no hallucinating)
        #   - Always cite the source file and page number
        prompt = f"""You are a precise document Q&A assistant.

RULES (follow these strictly):
- Answer ONLY using the context provided below.
- After every fact you state, add an inline citation like: (filename.pdf, Page 3)
- If the answer is not in the context, say exactly:
  "I am sorry, the provided documents do not contain the answer to your question."
- Do NOT use your own knowledge. Do NOT make up facts.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_question}

ANSWER:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )
            )

            answer = response.text.strip()

            # Extract token count if available
            tokens_used = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = response.usage_metadata.total_token_count

            return answer, tokens_used

        except Exception as e:
            error_msg = str(e)
            print(f"❌ LLM Error: {error_msg}")

            # Provide specific error messages based on the type of failure
            if "API_KEY" in error_msg.upper() or "INVALID" in error_msg.upper():
                raise ValueError("Invalid or missing Gemini API key. Please check your .env file.")
            elif "QUOTA" in error_msg.upper() or "RATE" in error_msg.upper():
                raise RuntimeError("API rate limit reached. Please try again in a moment.")
            elif "TIMEOUT" in error_msg.lower():
                raise TimeoutError("Request to Gemini API timed out. Please try again.")
            else:
                raise RuntimeError(f"LLM generation failed: {error_msg}")
