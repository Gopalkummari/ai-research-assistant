from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, Float
from src.database.base import Base

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    doc_id = Column(String, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(String, default="PENDING")  # PENDING, PROCESSING, PROCESSED, FAILED
    category = Column(String, default="Unclassified")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True)
    doc_id = Column(String, index=True, nullable=False)
    file_name = Column(String, nullable=False)
    page_number = Column(Integer, default=1)
    chunk_text = Column(Text, nullable=False)

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class QueryAnalytics(Base):
    __tablename__ = "query_analytics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, index=True)
    query_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
