import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

DOCUMENTS_DIR = "./documents"
RAG_DB_DIR = "./chroma_rag_db"


class DocumentRAG:
    def __init__(
        self,
        documents_dir: str = DOCUMENTS_DIR,
        persist_directory: str = RAG_DB_DIR,
        collection_name: str = "document_rag",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(exist_ok=True)

        self.embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(texts, normalize_embeddings=True).tolist()

    def _estimate_tokens(self, text: str) -> int:
        return len(text.split())

    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_index": len(chunks),
                "start_word": start,
                "end_word": min(end, len(words))
            })

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _read_txt(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _read_md(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, file_path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(file_path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                return f"[PDF support requires pypdf: pip install pypdf]"

    def _read_document(self, file_path: Path) -> Optional[str]:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return self._read_txt(file_path)
        elif suffix == ".md":
            return self._read_md(file_path)
        elif suffix == ".pdf":
            return self._read_pdf(file_path)
        return None

    def load_documents(self) -> Dict:
        supported_extensions = {".txt", ".md", ".pdf"}
        loaded = []
        skipped = []
        total_chunks = 0

        for file_path in self.documents_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                # Check if already loaded
                existing = self.collection.get(where={"source": str(file_path.name)})
                if existing and existing["ids"]:
                    skipped.append(file_path.name)
                    continue

                content = self._read_document(file_path)
                if content and not content.startswith("[PDF support"):
                    chunks = self._chunk_text(content, file_path.name)

                    if chunks:
                        ids = [f"{file_path.name}_chunk_{i}_{datetime.now().timestamp()}" for i in range(len(chunks))]
                        texts = [c["text"] for c in chunks]
                        embeddings = self._get_embeddings_batch(texts)
                        metadatas = [{
                            "source": c["source"],
                            "chunk_index": c["chunk_index"],
                            "start_word": c["start_word"],
                            "end_word": c["end_word"],
                            "loaded_at": datetime.now().isoformat()
                        } for c in chunks]

                        self.collection.add(
                            ids=ids,
                            embeddings=embeddings,
                            documents=texts,
                            metadatas=metadatas
                        )

                        loaded.append(file_path.name)
                        total_chunks += len(chunks)

        return {
            "loaded": loaded,
            "skipped": skipped,
            "total_chunks": total_chunks,
            "total_documents": len(loaded)
        }

    def add_document(self, file_path: str) -> Dict:
        path = Path(file_path)

        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            return {"error": f"Unsupported format: {path.suffix}"}

        # Copy to documents folder if not already there
        dest_path = self.documents_dir / path.name
        if not dest_path.exists():
            import shutil
            shutil.copy(path, dest_path)

        content = self._read_document(path)
        if not content:
            return {"error": "Could not read document"}

        if content.startswith("[PDF support"):
            return {"error": content}

        # Remove existing chunks for this document
        existing = self.collection.get(where={"source": path.name})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        chunks = self._chunk_text(content, path.name)

        if chunks:
            ids = [f"{path.name}_chunk_{i}_{datetime.now().timestamp()}" for i in range(len(chunks))]
            texts = [c["text"] for c in chunks]
            embeddings = self._get_embeddings_batch(texts)
            metadatas = [{
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "start_word": c["start_word"],
                "end_word": c["end_word"],
                "loaded_at": datetime.now().isoformat()
            } for c in chunks]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

        return {
            "success": True,
            "document": path.name,
            "chunks_created": len(chunks)
        }

    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = self._get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count())
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append({
                    "content": doc,
                    "source": results["metadatas"][0][i].get("source", "unknown") if results["metadatas"] else "unknown",
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0) if results["metadatas"] else 0,
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return documents

    def list_documents(self) -> List[Dict]:
        all_data = self.collection.get()
        sources = {}

        if all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                source = meta.get("source", "unknown")
                if source not in sources:
                    sources[source] = {"name": source, "chunks": 0, "loaded_at": meta.get("loaded_at")}
                sources[source]["chunks"] += 1

        return list(sources.values())

    def clear_rag_database(self) -> int:
        count = self.collection.count()
        if count > 0:
            all_ids = self.collection.get()["ids"]
            self.collection.delete(ids=all_ids)
        return count

    def remove_document(self, document_name: str) -> Dict:
        existing = self.collection.get(where={"source": document_name})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            return {"success": True, "removed": document_name, "chunks_removed": len(existing["ids"])}
        return {"error": f"Document not found: {document_name}"}

    def get_chunk_count(self) -> int:
        return self.collection.count()

    def format_context_for_prompt(self, chunks: List[Dict]) -> str:
        if not chunks:
            return ""

        formatted = "Relevant document excerpts:\n"
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")
            formatted += f"\n[{i}] From '{source}':\n{content}\n"

        return formatted
