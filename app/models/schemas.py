# ==============================================================================
# Pydantic Models - Defines the shape of request and response data
# FastAPI uses these to validate incoming JSON automatically
# ==============================================================================

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """
    What the frontend sends to us.
    Example:
    {
        "sessionId": "abc123",
        "message": "How do I reset my password?"
    }
    """
    sessionId: str = Field(..., description="Unique session ID for conversation tracking")
    message: str = Field(..., description="The user's question", min_length=1)


class ChatResponse(BaseModel):
    """
    What we send back to the frontend.
    Example:
    {
        "reply": "You can reset your password from Settings > Security.",
        "tokensUsed": 120,
        "retrievedChunks": 3
    }
    """
    reply: str
    tokensUsed: Optional[int] = None
    retrievedChunks: int = 0


class ErrorResponse(BaseModel):
    """
    Structured error response.
    Example:
    {
        "error": "Message field is required"
    }
    """
    error: str
