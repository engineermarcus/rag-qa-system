from PyPDF2 import PdfReader
from docx import Document 
import uuid
from datetime import datetime
from typing import Tuple, List, Dict

class DocumentProcessor:
    # process chunks for rag 
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    async def process_and_store(self, file_content: bytes, filename: str,vector_store) -> Tuple[str, int]:
        if filename.endswith('.pdf'):
            text = self._extract_pdf(file_content)
        elif filename.endswith('.docx'):
            text = self._extract_docx(file_content)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        # split into chunks 
        chunks = self._create_chunks(text)
        # generate unique doc
        doc_id = str(uuid.uuid4())
        # prepare chunks with meta data 
        chunks_documents = []
        for i, chunk_text in enumerate(chunks):
             chunk_documents.append({
                "doc_id": doc_id,
                "filename": filename, 
                "content": chunk_text,
                "chunk_index": i,
                "uploaded_at": datetime.utcnow(),
                "metadata": {
                "total_chunks": len(chunks),
                "file_type": filename.split('.')[-1]
            } 

        })

        # store in vector database 
        await vector_store.store_chunks(chunk_documents)
        return doc_id, len(chunks)

    def _extract_pdf(self, file_content: bytes) -> str:
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            if not text.strip():
                raise ValueError("Not text could be extracted from PDF")
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    def _extract_docx(self, file_content: bytes) -> str:
        try:
            doc = Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n\n"
                if not text.strip():
                    raise ValueError("Not text could be extracted from DOCX")
                return text
        except Exception as e:
            raise ValueError()
    def _create_chunks(self, text: str) -> List[str]:
        """
        Basic chunk splitting 
        for production use LangChain's text splitters
        """
        text = text.strip()
        if not text:
            return []
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            # Get chunk end position 
            end = start + self.chunk_size
            # if not the last chunk try to brea sentences/paragraphs 
            if end < text_length:
                paragraph_break != text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start:
                    end = paragraph_break 
                else:
                    # look for sentence break
                    if sentence_break != -1 and sentence_break > start:
                        end = sentence_break + 1
            # extract chunk 
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # Move start position with overlap
            start = end - self.chunk_overlap
            # prevent infinite loop
            if start <= end - self.chunk_size:
                start = end
            return chunks
    def get_chunk_preview(self, text: str, max_length: int = 100) -> str:
        if len(text) <= max_length:
            return text 
        return text[:max_length] + "..."



 

