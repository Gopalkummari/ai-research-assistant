import fitz  # PyMuPDF
from typing import List, Dict, Any

class PDFParser:
    def __init__(self):
        pass

    def extract_pages(self, pdf_path: str, doc_id: str, file_name: str) -> List[Dict[str, Any]]:
        """
        Extracts text from PDF page by page while preserving page numbers and document metadata.
        """
        doc = fitz.open(pdf_path)
        pages_data = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page_number": page_idx + 1,
                    "text": text
                })
        doc.close()
        return pages_data
