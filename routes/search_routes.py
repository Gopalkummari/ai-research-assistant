import json
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.database.models import ChatSession, QueryAnalytics
from src.vector_store.manager import VectorStoreManager
from src.rag.qa_chain import RAGEngine

router = APIRouter(tags=["Search & AI Question Answering"])
vector_store = VectorStoreManager()
rag_engine = RAGEngine(vector_store)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 4
    doc_ids: Optional[List[str]] = None

class RAGRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"
    doc_ids: Optional[List[str]] = None
    top_k: int = 4

@router.post("/search/semantic")
def semantic_search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    results = vector_store.semantic_search(req.query, top_k=req.top_k, doc_ids=req.doc_ids)
    return {"query": req.query, "mode": "semantic", "results_count": len(results), "results": results}

@router.post("/search/hybrid")
def hybrid_search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    results = vector_store.hybrid_search(req.query, top_k=req.top_k, doc_ids=req.doc_ids)
    return {"query": req.query, "mode": "hybrid", "results_count": len(results), "results": results}

@router.post("/rag/ask")
def ask_question(req: RAGRequest, db: Session = Depends(get_db)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve previous conversation history for session
    chat_records = db.query(ChatSession).filter(ChatSession.session_id == req.session_id).order_by(ChatSession.timestamp.asc()).all()
    history = [{"user": r.user_query, "assistant": r.assistant_response} for r in chat_records]

    # Generate grounded response
    res = rag_engine.answer_question(
        query=req.query,
        session_id=req.session_id,
        chat_history=history,
        doc_ids=req.doc_ids,
        top_k=req.top_k
    )

    # Save to ChatSession history database
    session_entry = ChatSession(
        session_id=req.session_id,
        user_query=req.query,
        assistant_response=res["answer"],
        citations_json=json.dumps(res["citations"])
    )
    db.add(session_entry)

    # Log analytics
    if req.doc_ids:
        for did in req.doc_ids:
            db.add(QueryAnalytics(doc_id=did, query_text=req.query))
    elif res["citations"]:
        for cit in res["citations"]:
            if "doc_id" in cit and cit["doc_id"] != "Unknown":
                db.add(QueryAnalytics(doc_id=cit["doc_id"], query_text=req.query))

    db.commit()

    return res
