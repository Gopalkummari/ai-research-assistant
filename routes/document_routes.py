import os
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from config.settings import settings
from src.database.base import get_db
from src.database.models import DocumentMetadata
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker
from src.ml.predictor import DocumentClassifier
from src.vector_store.manager import VectorStoreManager

router = APIRouter(prefix="/documents", tags=["Document Management"])
pdf_parser = PDFParser()
chunker = DocumentChunker()
classifier = DocumentClassifier()
vector_store = VectorStoreManager()

def process_pdf_pipeline(doc_id: str, file_path: str, file_name: str, db_session_factory):
    db = db_session_factory()
    try:
        doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
        if doc:
            doc.processing_status = "PROCESSING"
            db.commit()

        # 1. Page extraction
        pages = pdf_parser.extract_pages(file_path, doc_id, file_name)
        total_pages = len(pages)

        # 2. Text classification via TensorFlow model
        full_text = " ".join([p["text"] for p in pages])
        prediction = classifier.predict(full_text)
        predicted_category = prediction.get("category", "Unclassified")

        # 3. Text chunking
        chunks = chunker.create_chunks(pages)
        total_chunks = len(chunks)

        # 4. Vector DB Indexing
        vector_store.delete_document_chunks(doc_id)
        vector_store.add_chunks(chunks)

        # 5. Update Database Record
        if doc:
            doc.total_pages = total_pages
            doc.total_chunks = total_chunks
            doc.category = predicted_category
            doc.processing_status = "PROCESSED"
            db.commit()
    except Exception as e:
        print(f"Error processing pipeline for {doc_id}: {e}")
        doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
        if doc:
            doc.processing_status = "FAILED"
            db.commit()
    finally:
        db.close()

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(settings.RAW_DOCUMENTS_DIR, f"{doc_id}_{file.filename}")

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    doc_record = DocumentMetadata(
        doc_id=doc_id,
        file_name=file.filename,
        file_path=save_path,
        upload_timestamp=datetime.utcnow(),
        processing_status="PENDING",
        category="Processing..."
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    from src.database.base import SessionLocal
    background_tasks.add_task(process_pdf_pipeline, doc_id, save_path, file.filename, SessionLocal)

    return {
        "message": "Document uploaded successfully. Background processing started.",
        "doc_id": doc_id,
        "file_name": file.filename,
        "status": "PENDING"
    }

@router.get("")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(DocumentMetadata).all()
    return {"total": len(docs), "documents": docs}

@router.get("/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc

@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception:
            pass

    vector_store.delete_document_chunks(doc_id)

    db.delete(doc)
    db.commit()

    return {"message": f"Document {doc_id} and associated vector embeddings successfully deleted."}

@router.post("/{doc_id}/reprocess")
def reprocess_document(doc_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc.processing_status = "PENDING"
    db.commit()

    from src.database.base import SessionLocal
    background_tasks.add_task(process_pdf_pipeline, doc_id, doc.file_path, doc.file_name, SessionLocal)

    return {"message": f"Reprocessing triggered for document {doc_id}."}
