---
title: RAG AI Assessment
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 Document Q&A Bot — RAG System

A production-grade AI assistant that answers questions about **your own documents** (PDFs and Word files) using **Retrieval-Augmented Generation (RAG)**.

Powered by **Google Gemini** (LLM + Embeddings) and **ChromaDB** (vector database).

---

## 📌 How It Works

```
Your Documents (PDF, DOCX)
        │
        ▼
[ ingest.py: Extract text → Chunk → Embed → Save to ChromaDB ]
        │
        ▼
   ChromaDB (on disk)
        │
        ▼ (at query time)
User Question → Embed Question → Search ChromaDB → Top-4 Matching Chunks
        │
        ▼
[ Grounded Prompt: Context + Citations + Question → Gemini LLM ]
        │
        ▼
  Answer with inline citations  (e.g. "...grew by 14% (report.pdf, Page 3)")
```

---

## 📁 Project Structure

```
document-qa-bot/
│
├── .env                  # Your API keys (never commit this!)
├── .env.example          # Template — copy to .env
├── .gitignore            # Files git ignores
├── requirements.txt      # All Python dependencies
├── main.py               # FastAPI web server entry point
│
├── data/                 # ← PUT YOUR PDF/DOCX FILES HERE
│   └── README.md
│
├── db/                   # ChromaDB database (auto-created by ingest.py)
│
├── src/                  # Core pipeline code
│   ├── __init__.py
│   ├── config.py         # All settings in one place (chunk size, model names, etc.)
│   ├── ingest.py         # STEP 1: Read docs → chunk → embed → save to ChromaDB
│   ├── query.py          # STEP 2: Embed question → search → generate answer
│   └── main.py           # CLI interface (interactive question loop)
│
├── app/                  # FastAPI web server code
│   ├── routes/
│   │   └── chat.py       # POST /api/chat endpoint
│   ├── services/
│   │   ├── rag_service.py    # Orchestrates retrieval + generation for web UI
│   │   ├── embedding_service.py  # (legacy, not used by web UI anymore)
│   │   └── llm_service.py    # Gemini LLM call
│   ├── models/
│   │   └── schemas.py    # Request/response data shapes
│   ├── utils/
│   │   └── memory.py     # Conversation history (per session)
│   └── vectorstore/
│       └── chroma_store.py  # ChromaDB wrapper for the web UI
│
└── frontend/             # Web UI (HTML + CSS + JS)
    ├── index.html
    ├── styles.css
    └── app.js
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
copy .env.example .env
```

Open `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_key_here
```

Get a free key at: https://aistudio.google.com/app/apikey

### 3. Add your documents

Drop PDF and DOCX files into the `data/` folder.

### 4. Run the ingestion pipeline (once)

```bash
python src/ingest.py
```

This reads your documents, embeds them, and saves the database to `db/`.
You only need to do this once (or when you add new documents).

---

## 🚀 Running the Q&A Bot

### Option A: Command-Line Interface

```bash
python src/main.py
```

Interactive chat in your terminal. Type questions, press Enter, get cited answers.

### Option B: Web UI (Chat Interface)

```bash
python main.py
```

Then open: **http://localhost:8000**

---

## 📊 Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| LLM             | Gemini 2.5 Flash (`google-generativeai`) |
| Embeddings      | `text-embedding-004` (768 dimensions) |
| Vector Database | ChromaDB (disk-persistent)          |
| PDF Parsing     | `pypdf`                             |
| DOCX Parsing    | `python-docx`                       |
| Web Backend     | FastAPI + Uvicorn                   |
| Frontend        | HTML + CSS + Vanilla JavaScript     |

---

## 🔑 Key Design Decisions

| Decision | Why |
|----------|-----|
| **Separate ingest.py** | Embedding is slow and uses API tokens. Run once, reuse forever. |
| **Chunk overlap (200 chars)** | Prevents key sentences from being cut between chunks. |
| **Similarity threshold (0.4)** | Filters out unrelated chunks to avoid hallucination. |
| **Citation labels in prompt** | Forces the LLM to reference which file and page each fact came from. |
| **Low temperature (0.2)** | Makes answers factual and consistent, not creative. |
