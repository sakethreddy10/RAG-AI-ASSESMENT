# ==============================================================================
# Conversation Memory - Manages chat history per session
#
# We keep the last 5 message pairs (user + assistant) per session.
# This is stored in memory (a Python dictionary).
# In production, you'd use Redis or a database for this.
# ==============================================================================

from collections import defaultdict

# Maximum number of message pairs to remember per session
MAX_HISTORY_PAIRS = 5


class ConversationMemory:
    def __init__(self):
        """
        Simple in-memory storage using a dictionary.
        Key: sessionId (string)
        Value: list of messages [{"role": "user"/"assistant", "content": "..."}]
        """
        # defaultdict automatically creates an empty list for new session IDs
        self.sessions = defaultdict(list)

    def get_history(self, session_id: str) -> list[dict]:
        """
        Get the conversation history for a session.

        Args:
            session_id: Unique identifier for the conversation

        Returns:
            List of message dicts [{"role": "user", "content": "..."}, ...]
        """
        return self.sessions[session_id].copy()

    def add_message(self, session_id: str, role: str, content: str):
        """
        Add a new message to the conversation history.
        Automatically trims to keep only the last MAX_HISTORY_PAIRS pairs.

        Args:
            session_id: Unique identifier for the conversation
            role: "user" or "assistant"
            content: The message text
        """
        self.sessions[session_id].append({
            "role": role,
            "content": content
        })

        # Keep only the last MAX_HISTORY_PAIRS * 2 messages (user + assistant = 2 per pair)
        max_messages = MAX_HISTORY_PAIRS * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"🗑️  Cleared session: {session_id}")

    def get_all_sessions(self) -> list[str]:
        """Get all active session IDs"""
        return list(self.sessions.keys())


# Create a single global instance (singleton pattern)
# This is imported and reused by the chat route
conversation_memory = ConversationMemory()
