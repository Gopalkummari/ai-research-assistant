import os
import uuid
import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from main import app
from src.database.base import SessionLocal
from src.database.models import DocumentMetadata, DocumentChunk
from src.database.crud import delete_document_record
from src.vector_store.manager import VectorStoreManager

client = TestClient(app)

def test_delete_document_record_unit():
    db = SessionLocal()
    vector_store = VectorStoreManager()

    doc_id = f"test_del_{uuid.uuid4().hex[:8]}"
    test_file_path = f"data/raw_documents/{doc_id}_test.pdf"

    # Create dummy raw file
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
    with open(test_file_path, "w") as f:
        f.write("%PDF-1.4 dummy test content")

    # Add metadata record
    doc_record = DocumentMetadata(
        doc_id=doc_id,
        file_name="test_delete_doc.pdf",
        file_path=test_file_path,
        upload_timestamp=datetime.utcnow(),
        total_pages=1,
        total_chunks=1,
        processing_status="PROCESSED",
        category="Computer Science"
    )
    db.add(doc_record)

    # Add dummy chunk
    chunks = [{
        "chunk_id": f"{doc_id}_c1",
        "doc_id": doc_id,
        "file_name": "test_delete_doc.pdf",
        "page_number": 1,
        "text": "This is a test chunk to be deleted."
    }]
    vector_store.add_chunks(chunks)
    db.commit()

    # Verify document exists before deletion
    assert db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first() is not None
    assert db.query(DocumentChunk).filter(DocumentChunk.doc_id == doc_id).count() > 0

    # Perform deletion
    success = delete_document_record(doc_id, db, vector_store)
    assert success is True

    # Verify document and file are deleted
    assert db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first() is None
    assert db.query(DocumentChunk).filter(DocumentChunk.doc_id == doc_id).count() == 0
    assert not os.path.exists(test_file_path)

    # Test non-existent document deletion returns False
    assert delete_document_record("non_existent_id", db, vector_store) is False
    db.close()


def test_delete_document_api_endpoint():
    db = SessionLocal()
    vector_store = VectorStoreManager()

    doc_id = f"test_api_del_{uuid.uuid4().hex[:8]}"
    test_file_path = f"data/raw_documents/{doc_id}_test.pdf"

    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
    with open(test_file_path, "w") as f:
        f.write("%PDF-1.4 API test delete content")

    doc_record = DocumentMetadata(
        doc_id=doc_id,
        file_name="api_test_delete.pdf",
        file_path=test_file_path,
        upload_timestamp=datetime.utcnow(),
        total_pages=1,
        total_chunks=1,
        processing_status="PROCESSED",
        category="Physics"
    )
    db.add(doc_record)
    db.commit()
    db.close()

    # Call DELETE API endpoint
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert "successfully deleted" in response.json()["message"]

    # Call DELETE API endpoint for non-existent doc ID -> 404
    response_404 = client.delete(f"/documents/{doc_id}")
    assert response_404.status_code == 404
