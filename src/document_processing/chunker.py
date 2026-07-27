from typing import List, Dict, Any

class DocumentChunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits page texts into overlapping chunks, maintaining document and page metadata.
        """
        chunks = []
        global_chunk_idx = 0

        for page in pages_data:
            text = page["text"]
            doc_id = page["doc_id"]
            file_name = page["file_name"]
            page_number = page["page_number"]

            start = 0
            text_len = len(text)

            if text_len <= self.chunk_size:
                chunks.append({
                    "chunk_id": f"{doc_id}_p{page_number}_c{global_chunk_idx}",
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page_number": page_number,
                    "text": text
                })
                global_chunk_idx += 1
            else:
                while start < text_len:
                    end = start + self.chunk_size
                    chunk_text = text[start:end]

                    chunks.append({
                        "chunk_id": f"{doc_id}_p{page_number}_c{global_chunk_idx}",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page_number": page_number,
                        "text": chunk_text
                    })
                    global_chunk_idx += 1
                    start += (self.chunk_size - self.chunk_overlap)

        return chunks
