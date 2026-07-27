import pytest
from src.document_processing.chunker import DocumentChunker

def test_document_chunker():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    sample_pages = [
        {
            "doc_id": "doc_123",
            "file_name": "sample.pdf",
            "page_number": 1,
            "text": "This is page one of a sample research document. It contains detailed technical descriptions of neural network architectures."
        }
    ]
    chunks = chunker.create_chunks(sample_pages)
    assert len(chunks) > 0
    assert chunks[0]["doc_id"] == "doc_123"
    assert chunks[0]["page_number"] == 1
    assert "chunk_id" in chunks[0]
