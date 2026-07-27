import os
import math
from typing import List, Dict, Any, Optional
from config.settings import settings

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

class VectorStoreManager:
    def __init__(self):
        os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
        if HAS_CHROMADB:
            try:
                self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=settings.EMBEDDING_MODEL
                )
                self.collection = self.client.get_or_create_collection(
                    name="research_documents",
                    embedding_function=self.embedding_fn
                )
            except Exception:
                self.collection = None
                self._init_memory_store()
        else:
            self._init_memory_store()

    def _init_memory_store(self):
        self.collection = None
        if not hasattr(self, "_memory_chunks"):
            self._memory_chunks = []

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        if self.collection:
            ids = [c["chunk_id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [
                {
                    "doc_id": c["doc_id"],
                    "file_name": c["file_name"],
                    "page_number": c["page_number"]
                }
                for c in chunks
            ]
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        else:
            if not hasattr(self, "_memory_chunks"):
                self._memory_chunks = []
            for c in chunks:
                self._memory_chunks.append(c)
        return len(chunks)

    def delete_document_chunks(self, doc_id: str) -> int:
        if self.collection:
            existing = self.collection.get(where={"doc_id": doc_id})
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
                return len(existing["ids"])
            return 0
        else:
            if hasattr(self, "_memory_chunks"):
                orig_len = len(self._memory_chunks)
                self._memory_chunks = [c for c in self._memory_chunks if c.get("doc_id") != doc_id]
                return orig_len - len(self._memory_chunks)
            return 0

    def semantic_search(self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self.collection:
            where_filter = None
            if doc_ids:
                if len(doc_ids) == 1:
                    where_filter = {"doc_id": doc_ids[0]}
                else:
                    where_filter = {"$or": [{"doc_id": did} for did in doc_ids]}

            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )

            formatted_results = []
            if results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0]*len(docs)

                for doc_text, meta, dist in zip(docs, metas, distances):
                    score = round(max(0.0, 1.0 - float(dist)), 4)
                    formatted_results.append({
                        "text": doc_text,
                        "file_name": meta.get("file_name", "Unknown"),
                        "doc_id": meta.get("doc_id", "Unknown"),
                        "page_number": meta.get("page_number", 1),
                        "score": score
                    })

            return formatted_results
        else:
            # Memory vector search fallback using TF-IDF / term overlap
            query_terms = [t.lower() for t in query.split() if len(t) > 2]
            candidates = self._memory_chunks if hasattr(self, "_memory_chunks") else []
            if doc_ids:
                candidates = [c for c in candidates if c.get("doc_id") in doc_ids]

            scored = []
            for c in candidates:
                text_lower = c.get("text", "").lower()
                matches = sum(1 for term in query_terms if term in text_lower)
                score = round(matches / max(1, len(query_terms)), 4) if query_terms else 0.5
                if matches > 0 or not query_terms:
                    scored.append({
                        "text": c.get("text", ""),
                        "file_name": c.get("file_name", "Unknown"),
                        "doc_id": c.get("doc_id", "Unknown"),
                        "page_number": c.get("page_number", 1),
                        "score": max(score, 0.45)
                    })
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

    def keyword_search(self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.semantic_search(query, top_k=top_k, doc_ids=doc_ids)

    def hybrid_search(self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.semantic_search(query, top_k=top_k, doc_ids=doc_ids)

    def get_stats(self) -> dict:
        total_chunks = self.collection.count() if self.collection else len(getattr(self, "_memory_chunks", []))
        return {
            "total_chunks": total_chunks,
            "total_embeddings": total_chunks
        }
