import json
import requests
from typing import List, Dict, Any, Optional
from config.settings import settings
from src.vector_store.manager import VectorStoreManager

FALLBACK_MESSAGE = "I cannot determine the answer from the provided documents."

class RAGEngine:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def answer_question(
        self,
        query: str,
        session_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Retrieves top relevant chunks and generates a grounded response with page/file citations.
        """
        # Resolve query if conversational context exists
        effective_query = query
        if chat_history and len(chat_history) > 0:
            last_turn = chat_history[-1]
            last_q = last_turn.get("user", "")
            last_a = last_turn.get("assistant", "")
            # If query contains pronouns like 'it', 'its', 'this paper', append previous context hint
            pronouns = ["its", "it", "this paper", "the paper", "these", "their"]
            if any(p in query.lower().split() for p in pronouns):
                effective_query = f"{last_q} {query}"

        # Retrieve relevant chunks from vector database
        retrieved_chunks = self.vector_store.hybrid_search(
            query=effective_query,
            top_k=top_k,
            doc_ids=doc_ids
        )

        if not retrieved_chunks:
            return {
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "retrieved_context": [],
                "confidence_score": 0.0
            }

        # Build context string and citations array
        context_blocks = []
        citations = []
        scores = []

        for chunk in retrieved_chunks:
            doc_name = chunk.get("file_name", "Unknown Document")
            page_no = chunk.get("page_number", 1)
            text_content = chunk.get("text", "")
            score = chunk.get("score", 0.0)

            context_blocks.append(f"--- Source: {doc_name} (Page {page_no}) ---\n{text_content}")
            citations.append({
                "document": doc_name,
                "page": page_no,
                "doc_id": chunk.get("doc_id", "Unknown")
            })
            scores.append(score)

        avg_confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        # Check if context score is too low
        if avg_confidence < 0.15:
            return {
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "retrieved_context": [c["text"] for c in retrieved_chunks],
                "confidence_score": avg_confidence
            }

        # Attempt to synthesize response using OpenAI API if available, else local RAG synthesis engine
        combined_context = "\n\n".join(context_blocks)
        answer = self._generate_response(query, combined_context, chat_history)

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_context": [c["text"] for c in retrieved_chunks],
            "confidence_score": avg_confidence
        }

    def _generate_response(self, query: str, context: str, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """
        Synthesizes response using OpenAI API if key exists, or rule-based intelligent contextual synthesis.
        """
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
            try:
                headers = {
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                history_str = ""
                if chat_history:
                    history_str = "\n".join([f"User: {h.get('user', '')}\nAssistant: {h.get('assistant', '')}" for h in chat_history])

                system_prompt = (
                    "You are an AI Research & Knowledge Assistant. Answer the question based ONLY on the provided context.\n"
                    "If the context does not contain sufficient details to answer, reply exactly: 'I cannot determine the answer from the provided documents.'\n"
                    "Always mention source document names and page numbers in your response."
                )
                user_prompt = f"History:\n{history_str}\n\nContext:\n{context}\n\nQuestion: {query}"

                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0
                }

                res = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=15)
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"LLM API call failed: {e}. Falling back to local RAG synthesis.")

        # Local Grounded RAG Synthesizer
        lines = [line.strip() for line in context.split("\n") if line.strip() and not line.startswith("--- Source:")]
        relevant_extracts = lines[:4]
        
        sources_used = list(set([line for line in context.split("\n") if line.startswith("--- Source:")]))
        sources_str = ", ".join([s.replace("--- Source: ", "").replace(" ---", "") for s in sources_used])

        answer_text = (
            f"Based on the retrieved context from {sources_str}:\n\n"
            + "\n".join([f"• {ext[:250]}..." for ext in relevant_extracts])
            + f"\n\nSource References: {sources_str}"
        )
        return answer_text
