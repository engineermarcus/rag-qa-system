from docx import Document 
import uuid
import io
from datetime import datetime
from typing import Tuple, List, Dict
import fitz  # PyMuPDF
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR not available")

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Document processor initialized (chunk_size={chunk_size})")
    
    async def process_and_store(self, file_content: bytes, filename: str, vector_store) -> Tuple[str, int]:
        logger.info(f"Processing: {filename}")
        
        try:
            if filename.endswith('.pdf'):
                text = self._extract_pdf(file_content)
            elif filename.endswith('.docx'):
                text = self._extract_docx(file_content)
            else:
                raise ValueError(f"Unsupported: {filename}")
            
            logger.info(f"Extracted {len(text)} characters")
            
            chunks = self._create_chunks(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            doc_id = str(uuid.uuid4())
            
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

            logger.info("Storing chunks in database...")
            await vector_store.store_chunks(chunk_documents)
            logger.info(f"✓ Document processed: {doc_id}")
            
            return doc_id, len(chunks)
            
        except Exception as e:
            logger.error(f"Process and store failed: {e}", exc_info=True)
            raise

    def _extract_pdf(self, file_content: bytes) -> str:
        logger.info("Extracting PDF text...")
        pdf_document = None
        
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n\n"
            
            if text.strip():
                logger.info(f"✓ Extracted text from {len(pdf_document)} pages")
                return text
            
            # OCR fallback
            if OCR_AVAILABLE:
                logger.info("No text found, using OCR...")
                ocr_text = self._extract_pdf_with_ocr(pdf_document)
                return ocr_text
            
            raise ValueError("No text in PDF and OCR not available")
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}", exc_info=True)
            raise ValueError(f"PDF extraction failed: {e}")
        finally:
            if pdf_document:
                pdf_document.close()
                logger.info("PDF document closed")
    
    def _extract_pdf_with_ocr(self, pdf_document) -> str:
        text = ""
        num_pages = len(pdf_document)
        
        try:
            for page_num in range(num_pages):
                logger.info(f"OCR page {page_num + 1}/{num_pages}...")
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
                text += page_text + "\n\n"
                
                # Free memory
                del img
                del pix
            
            if not text.strip():
                raise ValueError("OCR failed - no text extracted")
            
            logger.info(f"✓ OCR completed for {num_pages} pages")
            return text
            
        except Exception as e:
            logger.error(f"OCR failed: {e}", exc_info=True)
            raise ValueError(f"OCR failed: {e}")
    
    def _extract_docx(self, file_content: bytes) -> str:
        logger.info("Extracting DOCX text...")
        try:
            doc = Document(io.BytesIO(file_content))
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n\n"
            
            if not text.strip():
                raise ValueError("No text in DOCX")
            
            logger.info("✓ DOCX extracted")
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}", exc_info=True)
            raise ValueError(f"DOCX extraction failed: {e}")
    
    def _create_chunks(self, text: str) -> List[str]:
        logger.info("Creating chunks...")
        
        try:
            text = text.strip()
            if not text:
                logger.warning("Empty text provided for chunking")
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
            
            logger.info(f"✓ Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}", exc_info=True)
            raise ValueError(f"Chunking failed: {e}")