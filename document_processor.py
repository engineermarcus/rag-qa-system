from docx import Document 
import uuid
import io
from datetime import datetime
from typing import Tuple, List, Dict
import fitz  # PyMuPDF

# OCR libraries
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

class DocumentProcessor:
    """Process documents and chunk them for RAG"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def process_and_store(self, file_content: bytes, filename: str, vector_store) -> Tuple[str, int]:
        """Process and store document in vector database"""
        
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = self._extract_pdf(file_content)
        elif filename.endswith('.docx'):
            text = self._extract_docx(file_content)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
        # Split into chunks 
        chunks = self._create_chunks(text)
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Prepare chunks with metadata 
        chunk_documents = []
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

        # Store in vector database 
        await vector_store.store_chunks(chunk_documents)
        
        return doc_id, len(chunks)

    def _extract_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF (with OCR fallback)"""
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            
            # Try extracting text
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n\n"
            
            # If we got text, return it
            if text.strip():
                pdf_document.close()
                print(f"✓ Extracted text from PDF ({len(pdf_document)} pages)")
                return text
            
            # If no text, try OCR
            print("⚠ No text found, attempting OCR...")
            ocr_text = self._extract_pdf_with_ocr(pdf_document)
            pdf_document.close()
            return ocr_text
            
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_pdf_with_ocr(self, pdf_document) -> str:
        """Extract text from image-based PDF using OCR"""
        if not OCR_AVAILABLE:
            raise ValueError(
                "PDF appears to be image-based, but OCR is not available. "
                "Install: pip install pytesseract pillow --break-system-packages"
            )
        
        try:
            text = ""
            num_pages = len(pdf_document)
            
            for page_num in range(num_pages):
                print(f"  OCR processing page {page_num + 1}/{num_pages}...")
                page = pdf_document[page_num]
                
                # Render page as image (300 DPI for better OCR)
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # OCR the image
                page_text = pytesseract.image_to_string(img)
                text += page_text + "\n\n"
            
            if not text.strip():
                raise ValueError("No text could be extracted even with OCR")
            
            print(f"✓ Successfully extracted text using OCR from {num_pages} pages")
            return text
            
        except Exception as e:
            raise ValueError(f"OCR failed: {str(e)}")
    
    def _extract_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(io.BytesIO(file_content))
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n\n"
            
            if not text.strip():
                raise ValueError("No text could be extracted from DOCX")
            
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {str(e)}")
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        text = text.strip()
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            if end < text_length:
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start:
                    end = paragraph_break 
                else:
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start:
                        end = sentence_break + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            
            if start <= end - self.chunk_size:
                start = end
        
        return chunks
    
    def get_chunk_preview(self, text: str, max_length: int = 100) -> str:
        """Get preview of chunk text"""
        if len(text) <= max_length:
            return text 
        return text[:max_length] + "..."