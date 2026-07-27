import requests
from typing import Dict, Any, Optional
from config.settings import settings
from src.vector_store.manager import VectorStoreManager

class DocumentSummarizer:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def summarize_document(self, doc_id: str, file_name: str = "Uploaded Document") -> Dict[str, Any]:
        """
        Generates Executive Summary, Technical Summary, Bullet Points, and Key Takeaways.
        """
        chunks = self.vector_store.semantic_search(query="overview summary technical methodology results conclusion", top_k=6, doc_ids=[doc_id])
        
        if not chunks:
            return {
                "executive_summary": "No content available for summarization.",
                "technical_summary": "No content available for summarization.",
                "bullet_points": [],
                "key_takeaways": []
            }

        extracted_text = "\n\n".join([c["text"] for c in chunks])

        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
            try:
                headers = {
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                prompt = (
                    f"Summarize the following document content into JSON with keys: "
                    f"'executive_summary', 'technical_summary', 'bullet_points' (list), 'key_takeaways' (list).\n\n"
                    f"Content:\n{extracted_text[:3000]}"
                )
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=15)
                if res.status_code == 200:
                    output = res.json()["choices"][0]["message"]["content"]
                    return json.loads(output)
            except Exception:
                pass

        # Rule-based structured summary fallback
        paragraphs = [p.strip() for p in extracted_text.split("\n\n") if len(p.strip()) > 40]
        exec_sum = paragraphs[0] if paragraphs else "This document covers technical research and analysis."
        tech_sum = paragraphs[1] if len(paragraphs) > 1 else exec_sum

        bullet_points = [f"Focuses on key concepts described in: {p[:120]}..." for p in paragraphs[:4]]
        key_takeaways = [
            f"Provides in-depth insights into {file_name} methodology.",
            "Establishes analytical frameworks and experimental findings.",
            "Serves as a reference for domain architecture and technical metrics."
        ]

        return {
            "document_id": doc_id,
            "file_name": file_name,
            "executive_summary": exec_sum,
            "technical_summary": tech_sum,
            "bullet_points": bullet_points,
            "key_takeaways": key_takeaways
        }
