import os
import math
from typing import List, Dict, Any, Optional
from config.settings import settings
from src.database.base import Base, engine, SessionLocal
from src.database.models import DocumentChunk

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

class VectorStoreManager:
    def __init__(self):
        os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
        Base.metadata.create_all(bind=engine)
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
        self._reload_memory_from_db()

    def _reload_memory_from_db(self):
        self._memory_chunks = []
        try:
            db = SessionLocal()
            db_chunks = db.query(DocumentChunk).all()
            for dc in db_chunks:
                self._memory_chunks.append({
                    "chunk_id": dc.id,
                    "doc_id": dc.doc_id,
                    "file_name": dc.file_name,
                    "page_number": dc.page_number,
                    "text": dc.chunk_text
                })
            db.close()
        except Exception as e:
            print(f"Reload memory error: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        # Save to SQLite Database for 100% persistence
        try:
            db = SessionLocal()
            for c in chunks:
                existing = db.query(DocumentChunk).filter(DocumentChunk.id == c["chunk_id"]).first()
                if not existing:
                    db_chunk = DocumentChunk(
                        id=c["chunk_id"],
                        doc_id=c["doc_id"],
                        file_name=c["file_name"],
                        page_number=c["page_number"],
                        chunk_text=c["text"]
                    )
                    db.add(db_chunk)
            db.commit()
            db.close()
        except Exception as e:
            print(f"Error persisting chunks to SQLite: {e}")

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

        self._reload_memory_from_db()
        return len(chunks)

    def delete_document_chunks(self, doc_id: str) -> int:
        try:
            db = SessionLocal()
            db.query(DocumentChunk).filter(DocumentChunk.doc_id == doc_id).delete()
            db.commit()
            db.close()
        except Exception:
            pass

        if self.collection:
            existing = self.collection.get(where={"doc_id": doc_id})
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])

        self._reload_memory_from_db()
        return 0

    def semantic_search(self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        self._reload_memory_from_db()

        if self.collection and self.collection.count() > 0:
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

            if formatted_results:
                return formatted_results

        # Memory / SQLite text search fallback
        query_terms = [t.lower() for t in query.split() if len(t) > 1]
        candidates = self._memory_chunks if hasattr(self, "_memory_chunks") else []
        if doc_ids:
            candidates = [c for c in candidates if c.get("doc_id") in doc_ids]

        scored = []
        for c in candidates:
            text_lower = c.get("text", "").lower()
            matches = sum(1 for term in query_terms if term in text_lower)
            score = round(matches / max(1, len(query_terms)), 4) if query_terms else 0.5
            scored.append({
                "text": c.get("text", ""),
                "file_name": c.get("file_name", "Unknown"),
                "doc_id": c.get("doc_id", "Unknown"),
                "page_number": c.get("page_number", 1),
                "score": max(score, 0.5)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def keyword_search(self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.semantic_search(query, top_k=top_k, doc_ids=doc_ids)

    def hybrid_search(self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.semantic_search(query, top_k=top_k, doc_ids=doc_ids)

    def get_stats(self) -> dict:
        self._reload_memory_from_db()
        total_chunks = len(self._memory_chunks)
        return {
            "total_chunks": total_chunks,
            "total_embeddings": total_chunks
        }
