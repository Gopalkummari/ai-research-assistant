from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.database.models import DocumentMetadata
from src.vector_store.manager import VectorStoreManager
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from src.ml.predictor import DocumentClassifier

router = APIRouter(prefix="/analysis", tags=["Document Analysis & ML"])
vector_store = VectorStoreManager()
summarizer = DocumentSummarizer(vector_store)
comparator = DocumentComparator(vector_store)
classifier = DocumentClassifier()

class SummarizeRequest(BaseModel):
    doc_id: str

class CompareRequest(BaseModel):
    doc_ids: List[str]

class ClassifyTextRequest(BaseModel):
    text: str

@router.post("/summarize")
def summarize_document(req: SummarizeRequest, db: Session = Depends(get_db)):
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == req.doc_id).first()
    file_name = doc.file_name if doc else "Uploaded Document"
    summary = summarizer.summarize_document(req.doc_id, file_name=file_name)
    return summary

@router.post("/compare")
def compare_documents(req: CompareRequest, db: Session = Depends(get_db)):
    if len(req.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 document IDs are required for comparison.")
    
    doc_names = []
    for did in req.doc_ids:
        doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == did).first()
        doc_names.append(doc.file_name if doc else f"Document {did[:8]}")

    result = comparator.compare_documents(req.doc_ids, doc_names=doc_names)
    return result

@router.post("/classify")
def classify_text(req: ClassifyTextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = classifier.predict(req.text)
    return result
