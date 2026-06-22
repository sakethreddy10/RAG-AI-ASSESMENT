# ==============================================================================
# ingest.py — Document Ingestion Pipeline
#
# Run this script ONCE to read your documents, break them into chunks,
# generate embeddings, and save everything to the ChromaDB vector database.
#
# HOW TO USE:
#   python src/ingest.py
#
# After running this, you never need to run it again unless you add new docs.
# ==============================================================================

import os
import sys

# Make sure Python can find our src/ package when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pypdf import PdfReader
from docx import Document
import chromadb
from tqdm import tqdm
from src.embedding_helper import GeminiEmbeddingFunction

from src.config import (
    GEMINI_API_KEY, EMBEDDING_MODEL,
    DATA_DIR, DB_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP
)


# ==============================================================================
# STEP 1: DOCUMENT EXTRACTION
# Read files from the data/ folder and extract raw text with page metadata.
# ==============================================================================

def extract_from_pdf(file_path: str) -> list[dict]:
    """
    Reads a PDF file and returns a list of pages.
    Each page is a dict: { "text": "...", "metadata": { "source": "file.pdf", "page": 1 } }
    """
    pages = []
    file_name = os.path.basename(file_path)

    print(f"  📄 Reading PDF: {file_name}")
    try:
        reader = PdfReader(file_path)

        for page_index, page in enumerate(reader.pages):
            raw_text = page.extract_text()

            # Skip empty pages (blank pages, image-only pages, etc.)
            if not raw_text or not raw_text.strip():
                continue

            # Clean up extra spaces and line breaks
            clean_text = " ".join(raw_text.split())

            pages.append({
                "text": clean_text,
                "metadata": {
                    "source": file_name,
                    "page": page_index + 1  # Make it 1-indexed (reader-friendly)
                }
            })

    except Exception as e:
        print(f"  ⚠️  Could not read {file_name}: {e}")

    return pages


def extract_from_docx(file_path: str) -> list[dict]:
    """
    Reads a Word (.docx) file and returns a list of paragraphs, grouped as 'pages'.
    Word docs don't have strict page boundaries, so we group every 30 paragraphs
    as a single logical page.
    """
    pages = []
    file_name = os.path.basename(file_path)

    print(f"  📝 Reading DOCX: {file_name}")
    try:
        doc = Document(file_path)

        # Collect all non-empty paragraphs
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        # Group every 30 paragraphs into one logical "page"
        GROUP_SIZE = 30
        for group_index in range(0, len(paragraphs), GROUP_SIZE):
            group_text = " ".join(paragraphs[group_index : group_index + GROUP_SIZE])
            page_number = (group_index // GROUP_SIZE) + 1

            pages.append({
                "text": group_text,
                "metadata": {
                    "source": file_name,
                    "page": page_number
                }
            })

    except Exception as e:
        print(f"  ⚠️  Could not read {file_name}: {e}")

    return pages


def load_all_documents(data_dir: str) -> list[dict]:
    """
    Scans the data/ folder and extracts text from all PDF and DOCX files.
    Returns a combined list of page-level dicts.
    """
    all_pages = []

    if not os.path.exists(data_dir):
        print(f"❌ Data folder not found: {data_dir}")
        print("   Please create it and add your PDF/DOCX files inside.")
        return []

    # Find all supported files
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".pdf", ".docx"))]

    if not files:
        print(f"❌ No PDF or DOCX files found in {data_dir}/")
        print("   Add some documents and try again.")
        return []

    print(f"\n📂 Found {len(files)} document(s) in {data_dir}/:")

    for file_name in files:
        file_path = os.path.join(data_dir, file_name)

        if file_name.lower().endswith(".pdf"):
            pages = extract_from_pdf(file_path)
        elif file_name.lower().endswith(".docx"):
            pages = extract_from_docx(file_path)
        else:
            pages = []  # Skip unsupported types

        all_pages.extend(pages)

    print(f"\n✅ Extracted {len(all_pages)} pages total from {len(files)} file(s).")
    return all_pages


# ==============================================================================
# STEP 2: TEXT CHUNKING
# Break long pages into smaller, overlapping chunks.
# ==============================================================================

def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Splits each page into smaller overlapping chunks.

    WHY OVERLAP?
    If a key sentence falls exactly on a chunk boundary, the overlap ensures
    both surrounding chunks still contain enough context to make sense.

    Example with chunk_size=10, overlap=3 (on characters):
      Text:  "ABCDEFGHIJKLMNOPQRST"
      Chunk1: "ABCDEFGHIJ"           (0 to 10)
      Chunk2: "HIJKLMNOPQ"           (7 to 17, overlap of 3)
      Chunk3: "OPQRST"               (14 to 20)
    """
    all_chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        text_length = len(text)

        start = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            # Record the character range so we can trace back the exact location
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"chars {start}–{end}"
                }
            })

            # Move the window forward (minus the overlap)
            # This means the next chunk starts inside the current one
            step = chunk_size - overlap
            start += step

    return all_chunks


# ==============================================================================
# STEP 3: EMBED AND SAVE TO CHROMADB
# Store the chunks in a real, persistent vector database on disk.
# ==============================================================================

def save_to_chromadb(chunks: list[dict]) -> None:
    """
    Embeds each chunk using Gemini's text-embedding-004 and stores
    everything in a ChromaDB database saved to the DB_DIR folder.

    After this runs, the database lives on disk. You can restart the server
    or CLI without re-running this script.
    """
    print(f"\n💾 Connecting to ChromaDB at: {DB_DIR}")
    os.makedirs(DB_DIR, exist_ok=True)

    # PersistentClient saves data to disk automatically
    client = chromadb.PersistentClient(path=DB_DIR)

    # This embedding function uses the Gemini API to convert text → vectors
    embedding_fn = GeminiEmbeddingFunction(
        api_key=GEMINI_API_KEY,
        model_name=EMBEDDING_MODEL
    )

    # Get or create the collection (think of it like a table in a database)
    # "hnsw:space": "cosine" means we use cosine similarity for search
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Check if we already have data so we don't double-index
    existing_count = collection.count()
    if existing_count > 0:
        print(f"⚠️  Database already has {existing_count} chunks.")
        answer = input("   Do you want to clear and re-index? (y/n): ").strip().lower()
        if answer == "y":
            client.delete_collection(COLLECTION_NAME)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            print("   🗑️  Old database cleared. Re-indexing...")
        else:
            print("   ✅ Keeping existing database. Exiting.")
            return

    # Prepare data for ChromaDB
    ids        = [f"chunk_{i}" for i in range(len(chunks))]
    documents  = [chunk["text"] for chunk in chunks]
    metadatas  = [chunk["metadata"] for chunk in chunks]

    # Upload in batches of 100 to avoid API rate limits
    # ChromaDB automatically calls the embedding function for each document
    BATCH_SIZE = 100
    print(f"\n🔢 Embedding and indexing {len(chunks)} chunks...")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Uploading batches"):
        batch_end = min(i + BATCH_SIZE, len(chunks))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )

    print(f"\n✅ Done! {len(chunks)} chunks stored in ChromaDB at '{DB_DIR}/'")
    print(f"   You can now run: python src/main.py")


# ==============================================================================
# MAIN — Run this script directly: python src/ingest.py
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  📚 Document Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load all documents from the data/ folder
    pages = load_all_documents(DATA_DIR)
    if not pages:
        sys.exit(1)  # Stop if no documents found

    # Step 2: Split pages into smaller overlapping chunks
    print(f"\n✂️  Chunking text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_pages(pages)
    print(f"   Created {len(chunks)} chunks from {len(pages)} pages.")

    # Step 3: Embed chunks and store in ChromaDB
    save_to_chromadb(chunks)
