import os
from sqlalchemy.orm import Session
from src.database.models import DocumentMetadata
from src.vector_store.manager import VectorStoreManager

def delete_document_record(doc_id: str, db: Session, vector_store: VectorStoreManager) -> bool:
    """
    Deletes a document metadata record, associated vector embeddings/chunks, and raw file from disk.
    Returns True if deletion succeeded, False if document was not found.
    """
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        return False

    # Remove physical file from disk if it exists
    if doc.file_path and os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception as e:
            print(f"Warning: Could not remove physical file {doc.file_path}: {e}")

    # Remove vector store embeddings and chunk DB records
    vector_store.delete_document_chunks(doc_id)

    # Remove document metadata record from DB
    db.delete(doc)
    db.commit()
    return True
