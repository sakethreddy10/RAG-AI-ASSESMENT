# ==============================================================================
# main.py — Interactive CLI Chat Interface
#
# This is the command-line version of the Document Q&A Bot.
# It loads the ChromaDB database once and lets you ask questions in a loop.
#
# HOW TO USE:
#   1. First, ingest your documents:   python src/ingest.py
#   2. Then start the chat:            python src/main.py
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.query import load_collection, ask


def print_divider():
    print("─" * 60)


def print_header():
    print_divider()
    print("  🧠 Document Q&A Bot  |  Powered by Gemini + ChromaDB")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print_divider()


def display_result(result: dict):
    """
    Pretty-prints the LLM's answer and the source citations.
    """
    print(f"\n🤖 Answer:\n")
    print(result["answer"])

    # Show the sources that were used
    if result["citations"]:
        print(f"\n📚 Sources used ({result['chunks_used']} chunk(s) retrieved):")
        # De-duplicate citations (the same page may be cited multiple times)
        seen = set()
        for citation in result["citations"]:
            if citation not in seen:
                print(f"   • {citation}")
                seen.add(citation)
    else:
        print("\n📭 No relevant document chunks were found for this question.")

    print()


def main():
    print_header()

    # Load the ChromaDB collection once at startup.
    # This is fast because the data is already on disk — no embedding needed.
    print("\n⏳ Loading vector database...")
    try:
        collection = load_collection()
        count = collection.count()
        print(f"✅ Database loaded. {count} chunks ready to search.\n")
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
        sys.exit(1)

    # ── Main Chat Loop ─────────────────────────────────────────────────────────
    while True:
        print_divider()

        # Get the user's question
        try:
            user_input = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or piped input ending
            print("\n\n👋 Goodbye!")
            break

        # Exit command
        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\n👋 Goodbye!")
            break

        # Skip blank input
        if not user_input:
            print("   ⚠️  Please type a question.")
            continue

        # Run the RAG pipeline and show the result
        print("\n🔍 Searching documents and generating answer...")
        try:
            result = ask(collection, user_input)
            display_result(result)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
