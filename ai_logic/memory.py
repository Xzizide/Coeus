import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import List, Dict, Optional

load_dotenv()

class ConversationMemory:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "conversation_memory"):
        self.embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def add_memory(self, user_message: str, assistant_response: str) -> str:
        timestamp = datetime.now().isoformat()
        memory_id = f"memory_{timestamp}"

        combined_text = f"User: {user_message}\nAssistant: {assistant_response}"
        embedding = self._get_embedding(combined_text)

        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[combined_text],
            metadatas=[{
                "user_message": user_message,
                "assistant_response": assistant_response,
                "timestamp": timestamp
            }]
        )
        return memory_id

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
