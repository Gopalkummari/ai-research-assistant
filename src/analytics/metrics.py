from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database.models import DocumentMetadata, ChatSession, QueryAnalytics
from src.vector_store.manager import VectorStoreManager

class AnalyticsEngine:
    def __init__(self, db: Session, vector_store: VectorStoreManager):
        self.db = db
        self.vector_store = vector_store

    def get_system_stats(self) -> dict:
        total_docs = self.db.query(DocumentMetadata).count()
        total_pages = self.db.query(func.sum(DocumentMetadata.total_pages)).scalar() or 0
        total_db_chunks = self.db.query(func.sum(DocumentMetadata.total_chunks)).scalar() or 0
        
        vector_stats = self.vector_store.get_stats()
        total_chunks = max(total_db_chunks, vector_stats.get("total_chunks", 0))

        # Categories breakdown
        categories_query = self.db.query(
            DocumentMetadata.category, func.count(DocumentMetadata.doc_id)
        ).group_by(DocumentMetadata.category).all()
        category_distribution = {cat: count for cat, count in categories_query}

        # Query stats
        total_questions = self.db.query(ChatSession).count()

        top_queried = self.db.query(
            QueryAnalytics.doc_id, func.count(QueryAnalytics.id).label("query_count")
        ).group_by(QueryAnalytics.doc_id).order_by(func.count(QueryAnalytics.id).desc()).limit(5).all()

        top_docs = []
        for did, count in top_queried:
            doc = self.db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == did).first()
            top_docs.append({
                "doc_id": did,
                "file_name": doc.file_name if doc else "Unknown",
                "queries_count": count
            })

        return {
            "total_documents": total_docs,
            "total_pages": total_pages,
            "total_processed_chunks": total_chunks,
            "total_embeddings": vector_stats.get("total_embeddings", total_chunks),
            "total_questions_answered": total_questions,
            "category_distribution": category_distribution,
            "top_queried_documents": top_docs
        }
