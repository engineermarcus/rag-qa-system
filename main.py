from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn 
from dotenv import load_dotenv
import os 
from typing import List, Optional

from mongo_vectorstore import MongoDBVectorStore
from document_processor import DocumentProcessor
from qa_chain import QAChain 

load_dotenv()

app = FastAPI(
    title="RAG Document Q&A System",
    description="AI-Powered document question answering with Gemini and MongoDB",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


vector_store = MongoDBVectorStore()
doc_processor = DocumentProcessor()
qa_chain = QAChain(vector_store)

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
    """API root endpoint"""
    return {
        "message": "RAG Document Q&A System - Powered by Gemini & MongoDB Atlas",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload PDF or DOCX documents",
            "query": "POST /query - Ask questions about documents",
            "delete": "DELETE /delete/{doc_id} - Delete a document by ID",
        } 
    }

@app.get("/health")
async def health_check():
    try:
        vector_store.test_connection()
        return {
            "status": "healthy",
            "database": "connected",
            "ai_model": "gemini-2.5-flash"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")


@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process document (PDF and Docx)
    The document will be:
    - Split into chunks 
    - Embedded using hugging face Model 
    - Stored in MongoDB 
    """
    try:
        if not file.filename.endswith(('.pdf', '.docx')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and Docx files are supported"
            )
        # read file content 
        content = await file.read()

        # process file 
        doc_id, num_chunks = await doc_processor.process_and_store(
            content,
            file.filename, 
            vector_store
        )

        return DocumentResponse(
            doc_id=doc_id,
            filename=file.filename,
            num_chunks=num_chunks,
            message=f"Document uploaded and indexed successfully with {num_chunks} chunks"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

            
@app.post("/query", response_model=QueryResponse)
async def query_document(query_request: QueryRequest):
    try:
        if not query_request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # get answer from QA chain 
        result = await qa_chain.ask(query_request.question, query_request.num_sources)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result.get('confidence')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

            
@app.get("/documents")
async def list_documents():
    """ 
    List all indexed documents
    """
    try: 
        docs = await vector_store.list_documents()
        return {
            "total": len(docs),
            "documents": docs 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
    

@app.delete("/delete/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document by ID 
    """
    try:
        delete_count = await vector_store.delete_document(doc_id)
        if delete_count == 0:
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")

        return {
            "message": f"Document deleted successfully!",
            "doc_id": doc_id,
            "chunks_deleted": delete_count
        }
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/stats")
async def get_stats():
    """System Statistics"""
    try:
        stats = await vector_store.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    print("Starting RAG Document Q&A System")
    print("API will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)