from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn 
from dotenv import load_dotenv
from typing import List, Optional
import logging
import asyncio

from mongo_vectorstore import MongoDBVectorStore
from document_processor import DocumentProcessor
from qa_chain import QAChain 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="RAG Q&A System",
    description="Fast RAG with Jina AI + Groq",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Initializing components...")
try:
    vector_store = MongoDBVectorStore()
    doc_processor = DocumentProcessor()
    qa_chain = QAChain(vector_store)
    logger.info("✓ All components initialized")
except Exception as e:
    logger.error(f"Initialization failed: {e}", exc_info=True)
    raise

class QueryRequest(BaseModel):
    question: str
    num_sources: int = 3

class QueryResponse(BaseModel): 
    answer: str
    sources: List[dict]
    confidence: Optional[str] = None

class DocumentResponse(BaseModel):
    doc_id: str
    filename: str 
    num_chunks: int 
    message: str 

@app.get("/")
async def root():
    return {
        "message": "RAG Q&A System",
        "version": "2.0.0",
        "stack": {
            "embeddings": "Jina AI",
            "llm": "Groq Llama 3.3",
            "database": "MongoDB Atlas"
        }
    }

@app.get("/health")
async def health():
    try:
        vector_store.test_connection()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(500, f"Unhealthy: {e}")

@app.post("/upload", response_model=DocumentResponse)
async def upload(file: UploadFile = File(...)):
    logger.info(f"Upload request: {file.filename}")
    
    try:
        if not file.filename.endswith(('.pdf', '.docx')):
            raise HTTPException(400, "Only PDF and DOCX supported")
        
        logger.info("Reading file content...")
        content = await file.read()
        logger.info(f"File size: {len(content)} bytes")
        
        logger.info("Processing document...")
        
        # Add timeout to prevent hanging
        try:
            doc_id, num_chunks = await asyncio.wait_for(
                doc_processor.process_and_store(content, file.filename, vector_store),
                timeout=300.0  # 5 minute timeout
            )
        except asyncio.TimeoutError:
            logger.error("Upload timeout after 5 minutes")
            raise HTTPException(504, "Processing timeout - file too large")

        logger.info(f"✓ Upload complete: {num_chunks} chunks")
        return DocumentResponse(
            doc_id=doc_id,
            filename=file.filename,
            num_chunks=num_chunks,
            message=f"Uploaded: {num_chunks} chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(500, f"Upload failed: {str(e)}")
            
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    logger.info(f"Query request: {req.question[:50]}...")
    
    try:
        if not req.question.strip():
            raise HTTPException(400, "Question cannot be empty")
        
        logger.info("Processing query...")
        
        # Add timeout
        try:
            result = await asyncio.wait_for(
                qa_chain.ask(req.question, req.num_sources),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.error("Query timeout")
            raise HTTPException(504, "Query timeout")
        
        logger.info("✓ Query complete")
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result.get('confidence')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Query failed: {str(e)}")
            
@app.get("/documents")
async def list_docs():
    try: 
        docs = await vector_store.list_documents()
        return {"total": len(docs), "documents": docs}
    except Exception as e:
        logger.error(f"List failed: {e}")
        raise HTTPException(500, f"Failed: {e}")
    
@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str):
    try:
        count = await vector_store.delete_document(doc_id)
        if count == 0:
            raise HTTPException(404, "Document not found")
        return {"message": "Deleted", "doc_id": doc_id, "chunks_deleted": count}
    except HTTPException:
        raise 
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(500, f"Delete failed: {e}")

@app.get("/stats")
async def stats():
    try:
        return await vector_store.get_stats()
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(500, f"Stats failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting RAG Q&A System")
    logger.info("📍 http://localhost:8000")
    logger.info("📚 http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)