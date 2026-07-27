import json
import requests
from typing import Dict, Any, Optional
from config.settings import settings
from src.vector_store.manager import VectorStoreManager
from src.database.base import SessionLocal
from src.database.models import DocumentChunk

class DocumentSummarizer:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def summarize_document(self, doc_id: str, file_name: str = "Uploaded Document") -> Dict[str, Any]:
        """
        Generates Executive Summary, Technical Summary, Bullet Points, and Key Takeaways
        from actual document content.
        """
        # Fetch all chunks for this doc_id
        chunks = []
        try:
            with SessionLocal() as db:
                db_chunks = db.query(DocumentChunk).filter(DocumentChunk.doc_id == doc_id).all()
                for c in db_chunks:
                    chunks.append({
                        "text": c.chunk_text,
                        "page_number": c.page_number
                    })
        except Exception as e:
            print(f"Error fetching chunks for summarization: {e}")

        if not chunks:
            # Fallback to vector store search
            search_res = self.vector_store.semantic_search(query="overview summary technical results", top_k=8, doc_ids=[doc_id])
            chunks = [{"text": c["text"], "page_number": c.get("page_number", 1)} for c in search_res]

        if not chunks:
            return {
                "document_id": doc_id,
                "file_name": file_name,
                "executive_summary": "No content available for summarization.",
                "technical_summary": "No content available for summarization.",
                "bullet_points": [],
                "key_takeaways": []
            }

        combined_text = "\n\n".join([c["text"] for c in chunks])

        # LLM Synthesis if OpenAI API key is configured
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip() and not settings.OPENAI_API_KEY.startswith("your_"):
            try:
                headers = {
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                prompt = (
                    f"Summarize the following document '{file_name}' into valid JSON with exact keys: "
                    f"'executive_summary' (string), 'technical_summary' (string), 'bullet_points' (list of strings), 'key_takeaways' (list of strings).\n\n"
                    f"Content:\n{combined_text[:4000]}"
                )
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=15)
                if res.status_code == 200:
                    raw_content = res.json()["choices"][0]["message"]["content"].strip()
                    parsed = json.loads(raw_content)
                    parsed["document_id"] = doc_id
                    parsed["file_name"] = file_name
                    return parsed
            except Exception as e:
                print(f"LLM Summarizer API error: {e}")

        # Real Extractive Content Summarizer Fallback
        lines = [line.strip() for line in combined_text.split("\n") if len(line.strip()) > 30]
        paragraphs = [p.strip() for p in combined_text.split("\n\n") if len(p.strip()) > 40]

        # Executive Summary: First meaningful paragraph / opening statements
        exec_sum = (
            paragraphs[0] if paragraphs else
            (lines[0] if lines else f"This document ({file_name}) provides technical analysis and specifications.")
        )
        if len(paragraphs) > 1 and len(exec_sum) < 150:
            exec_sum += " " + paragraphs[1]

        # Technical Summary: Middle / technical sections
        tech_candidates = [p for p in paragraphs if any(w in p.lower() for w in ["system", "data", "model", "method", "architecture", "analysis", "result", "feature", "implementation", "process"])]
        tech_sum = " ".join(tech_candidates[:2]) if tech_candidates else (paragraphs[-1] if len(paragraphs) > 2 else exec_sum)

        # Key Bullet Points: Representative excerpts across the document
        step = max(1, len(lines) // 5)
        selected_lines = [lines[i] for i in range(0, len(lines), step)][:5]
        bullet_points = [f"{line[:180]}..." if len(line) > 180 else line for line in selected_lines]

        # Key Takeaways
        takeaways = [
            f"Document '{file_name}' contains {len(chunks)} processed section chunks spanning {max([c['page_number'] for c in chunks], default=1)} page(s).",
            f"Primary focus centers around: {lines[0][:120]}..." if lines else "Structured domain technical documentation.",
            f"Key technical details include: {bullet_points[0]}" if bullet_points else "Comprehensive domain analysis."
        ]

        return {
            "document_id": doc_id,
            "file_name": file_name,
            "executive_summary": exec_sum,
            "technical_summary": tech_sum,
            "bullet_points": bullet_points,
            "key_takeaways": takeaways
        }
