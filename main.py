# ==============================================================================
# main.py — FastAPI Web Server Entry Point
#
# This starts the web server that serves the chat UI.
#
# IMPORTANT: Run src/ingest.py FIRST to build the vector database.
# Then start this server with:   python main.py
# ==============================================================================

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything else so API keys are available

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from app.routes.chat import router as chat_router

# Create the FastAPI application
app = FastAPI(
    title="Document Q&A Bot",
    description="A RAG-powered Q&A bot using Gemini + ChromaDB. Run src/ingest.py first.",
    version="2.0.0"
)

# Allow the frontend (served from any origin during development) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the /api/chat routes
app.include_router(chat_router, prefix="/api")

# Serve the frontend HTML/CSS/JS files from the frontend/ folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    """Serve the main chat page."""
    return FileResponse("frontend/index.html")


@app.get("/health")
def health_check():
    """Simple health check — useful for monitoring."""
    return {"status": "healthy", "version": "2.0.0"}


@app.on_event("startup")
async def startup_event():
    """
    Server startup check.
    We no longer index documents here — that's done by src/ingest.py.
    We just verify that the database exists and is ready to use.
    """
    import os
    from src.config import DB_DIR

    print("\n🚀 Starting Document Q&A Bot Web Server...")

    if os.path.exists(DB_DIR):
        print(f"✅ Found ChromaDB database at '{DB_DIR}/' — ready to answer questions.")
    else:
        print(f"⚠️  WARNING: No database found at '{DB_DIR}/'.")
        print("   The bot will not be able to answer questions.")
        print("   Run:  python src/ingest.py  to build the database first.")

    print("🌐 Web UI available at: http://localhost:8000\n")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
