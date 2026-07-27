import pytest
from src.vector_store.manager import VectorStoreManager
from src.rag.qa_chain import RAGEngine

def test_rag_qa_pipeline():
    vector_store = VectorStoreManager()
    
    # Add dummy test chunk
    test_chunk = [{
        "chunk_id": "test_doc_p1_c0",
        "doc_id": "test_doc",
        "file_name": "Test_Paper.pdf",
        "page_number": 1,
        "text": "The proposed ResNet architecture uses residual skip connections to train deep neural networks without gradient vanishing."
    }]
    vector_store.add_chunks(test_chunk)

    rag = RAGEngine(vector_store)
    result = rag.answer_question(query="What architecture uses residual skip connections?", doc_ids=["test_doc"])

    assert "answer" in result
    assert "citations" in result
    assert len(result["citations"]) > 0
    assert result["citations"][0]["document"] == "Test_Paper.pdf"
    assert result["citations"][0]["page"] == 1
