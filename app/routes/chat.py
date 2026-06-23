# ==============================================================================
# Chat Route - Handles POST /api/chat requests
#
# This is the main endpoint the frontend calls.
# Flow:
# 1. Validate incoming request
# 2. Get conversation history for session
# 3. Run RAG query (retrieval + generation)
# 4. Save messages to memory
# 5. Return response
# ==============================================================================

from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
from app.utils.memory import conversation_memory

router = APIRouter()

# Create one instance of RAGService (shared across requests)
rag_service = RAGService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Receives: { "sessionId": "abc123", "message": "How do I reset my password?" }
    Returns:  { "reply": "...", "tokensUsed": 120, "retrievedChunks": 3 }
    """

    # Validate that message is not empty (Pydantic handles min_length=1 already,
    # but let's also strip whitespace)
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=400,
            detail={"error": "Message field is required and cannot be empty"}
        )

    session_id = request.sessionId
    print(f"\n💬 Session [{session_id}] - Message: {message}")

    try:
        # Step 1: Get existing conversation history for this session
        history = conversation_memory.get_history(session_id)
        print(f"📜 History: {len(history)} messages in session")

        # Step 2: Run the full RAG pipeline (retrieve + generate)
        result = rag_service.query(
            user_question=message,
            conversation_history=history
        )

        # Step 3: Save this exchange to conversation memory
        conversation_memory.add_message(session_id, "user", message)
        conversation_memory.add_message(session_id, "assistant", result["reply"])

        # Step 4: Return the response
        return ChatResponse(
            reply=result["reply"],
            tokensUsed=result.get("tokensUsed"),
            retrievedChunks=result.get("retrievedChunks", 0)
        )

    except ValueError as e:
        # Configuration errors (bad API key, etc.)
        print(f"❌ ValueError: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

    except RuntimeError as e:
        # API errors (rate limits, etc.)
        print(f"❌ RuntimeError: {e}")
        raise HTTPException(status_code=503, detail={"error": str(e)})

    except TimeoutError as e:
        # Timeout errors
        print(f"❌ TimeoutError: {e}")
        raise HTTPException(status_code=504, detail={"error": str(e)})

    except Exception as e:
        # Catch-all for unexpected errors
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"An unexpected error occurred: {str(e)}"}
        )


@router.post("/chat/clear")
async def clear_session(session_id: str):
    """
    Clear conversation history for a session.
    Called when user clicks 'New Chat'.
    """
    conversation_memory.clear_session(session_id)
    return {"status": "cleared", "sessionId": session_id}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to dynamically upload and ingest a document.
    """
    import tempfile
    import shutil
    import os

    filename = file.filename
    if not filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail={"error": "Only PDF and DOCX files are supported."}
        )

    try:
        # Create a temp file to save the uploaded content
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        try:
            # Index the document chunks
            chunks_added = rag_service.ingest_file(temp_path, filename)
            
            # Save it to the permanent data/ folder so it's persisted for future ingest runs
            from src.config import DATA_DIR
            os.makedirs(DATA_DIR, exist_ok=True)
            perm_path = os.path.join(DATA_DIR, filename)
            shutil.copy(temp_path, perm_path)
            
        finally:
            # Clean up the temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return {
            "status": "success",
            "message": f"Successfully ingested '{filename}'",
            "chunks_added": chunks_added
        }

    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to ingest file: {str(e)}"}
        )
