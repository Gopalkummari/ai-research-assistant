import pytest
from src.ml.predictor import DocumentClassifier

def test_document_classifier():
    classifier = DocumentClassifier()
    sample_text = "Transformer architectures and attention mechanisms for machine translation and large language models."
    result = classifier.predict(sample_text)
    assert "category" in result
    assert "confidence" in result
    assert result["category"] in classifier.categories
