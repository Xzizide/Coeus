import os
import uuid
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from typing import List, Dict, Optional

load_dotenv()

class ConversationMemory:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "conversation_memory", session_timeout_minutes: int = 30):
        self.embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._last_message_time = None
        self._current_session_id = self._generate_session_id()

    def _generate_session_id(self) -> str:
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _check_session_timeout(self) -> bool:
        """Check if session has timed out and start new one if needed. Returns True if new session started."""
        if self._last_message_time is None:
            return False

        if datetime.now() - self._last_message_time > self.session_timeout:
            self._current_session_id = self._generate_session_id()
            return True
        return False

    def start_new_session(self) -> str:
        self._current_session_id = self._generate_session_id()
        self._last_message_time = None
        return self._current_session_id

    def get_current_session_id(self) -> str:
        return self._current_session_id

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def add_memory(self, user_message: str, assistant_response: str, session_id: Optional[str] = None) -> str:
        # Auto-detect new session based on time gap
        self._check_session_timeout()

        now = datetime.now()
        timestamp = now.isoformat()
        session = session_id or self._current_session_id
        memory_id = f"memory_{timestamp}_{uuid.uuid4().hex[:8]}"

        combined_text = f"User: {user_message}\nAssistant: {assistant_response}"
        embedding = self._get_embedding(combined_text)

        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[combined_text],
            metadatas=[{
                "user_message": user_message,
                "assistant_response": assistant_response,
                "timestamp": timestamp,
                "session_id": session,
                "message_index": self._get_session_message_count(session)
            }]
        )

        self._last_message_time = now
        return memory_id

    def _get_session_message_count(self, session_id: str) -> int:
        try:
            results = self.collection.get(where={"session_id": session_id})
            return len(results["ids"]) if results["ids"] else 0
        except:
            return 0

    def search_memories(self, query: str, n_results: int = 5) -> List[Dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = self._get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count())
        )

        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                memories.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })
        return memories

    def clear_memories(self) -> int:
        count = self.collection.count()
        if count > 0:
            all_ids = self.collection.get()["ids"]
            self.collection.delete(ids=all_ids)
        return count

    def get_memory_count(self) -> int:
        return self.collection.count()

    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        if not memories:
            return ""

        formatted = "Relevant past conversations:\n"
        for i, memory in enumerate(memories, 1):
            timestamp = memory["metadata"].get("timestamp", "unknown")
            formatted += f"\n[{i}] ({timestamp})\n{memory['content']}\n"
        return formatted

    def get_conversation_by_id(self, conv_id: str) -> List[Dict]:
        """Retrieve all messages from a conversation by session ID."""
        try:
            results = self.collection.get(where={"session_id": conv_id})

            if not results["ids"]:
                return []

            messages = []
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                messages.append({
                    "id": results["ids"][i],
                    "content": doc,
                    "user_message": meta.get("user_message", ""),
                    "assistant_response": meta.get("assistant_response", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "message_index": meta.get("message_index", i)
                })

            # Sort by message_index for chronological order
            messages.sort(key=lambda x: (x.get("timestamp", ""), x.get("message_index", 0)))
            return messages
        except Exception:
            return []

    def list_all_conversations(self) -> List[Dict]:
        """List all conversation sessions with metadata."""
        all_data = self.collection.get()

        if not all_data["ids"]:
            return []

        sessions = {}
        for meta in (all_data["metadatas"] or []):
            session_id = meta.get("session_id", "unknown")
            timestamp = meta.get("timestamp", "")

            if session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "message_count": 0,
                    "first_timestamp": timestamp,
                    "last_timestamp": timestamp
                }

            sessions[session_id]["message_count"] += 1

            # Track first and last timestamps
            if timestamp < sessions[session_id]["first_timestamp"]:
                sessions[session_id]["first_timestamp"] = timestamp
            if timestamp > sessions[session_id]["last_timestamp"]:
                sessions[session_id]["last_timestamp"] = timestamp

        # Sort by last_timestamp descending (most recent first)
        sorted_sessions = sorted(
            sessions.values(),
            key=lambda x: x["last_timestamp"],
            reverse=True
        )
        return sorted_sessions

    def search_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Search conversations within a date range.
        Dates should be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).
        """
        all_data = self.collection.get()

        if not all_data["ids"]:
            return []

        # Normalize dates for comparison
        if len(start_date) == 10:  # YYYY-MM-DD format
            start_date = f"{start_date}T00:00:00"
        if len(end_date) == 10:
            end_date = f"{end_date}T23:59:59"

        results = []
        for i, meta in enumerate(all_data["metadatas"] or []):
            timestamp = meta.get("timestamp", "")

            if start_date <= timestamp <= end_date:
                results.append({
                    "id": all_data["ids"][i],
                    "content": all_data["documents"][i],
                    "user_message": meta.get("user_message", ""),
                    "assistant_response": meta.get("assistant_response", ""),
                    "timestamp": timestamp,
                    "session_id": meta.get("session_id", "")
                })

        # Sort by timestamp
        results.sort(key=lambda x: x["timestamp"])
        return results

    def export_conversation(self, conv_id: str) -> Dict:
        """
        Export a complete conversation with full history in chronological order.
        Returns a dict with session metadata and all messages.
        """
        messages = self.get_conversation_by_id(conv_id)

        if not messages:
            return {"error": f"Conversation not found: {conv_id}"}

        return {
            "session_id": conv_id,
            "message_count": len(messages),
            "first_timestamp": messages[0]["timestamp"] if messages else "",
            "last_timestamp": messages[-1]["timestamp"] if messages else "",
            "messages": [
                {
                    "index": i,
                    "timestamp": msg["timestamp"],
                    "user": msg["user_message"],
                    "assistant": msg["assistant_response"]
                }
                for i, msg in enumerate(messages)
            ]
        }

    def get_all_messages_from_session(self, session_id: str) -> List[Dict]:
        """Alias for get_conversation_by_id for clarity."""
        return self.get_conversation_by_id(session_id)
