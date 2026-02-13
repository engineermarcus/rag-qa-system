from pymongo import MongoClient 
from pymongo.errors import ConnectionFailure
import httpx
import os 
from typing import List, Dict 
import numpy as np
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBVectorStore:
    """MongoDB vector store using Jina AI embeddings"""
    
    def __init__(self):
        logger.info("Initializing vector store...")
        
        self.connection_string = os.getenv("MONGO_URI")
        if not self.connection_string:
            raise ValueError("MONGO_URI not set")
        
        logger.info("Connecting to MongoDB...")
        self.client = MongoClient(self.connection_string)
        self.db = self.client["rag_qa_system"]
        self.collection = self.db["documents"]
        
        self.jina_api_key = os.getenv("JINA_API_KEY", "")
        
        self._setup_indexes()
        logger.info("✓ Vector store initialized")

    def _setup_indexes(self):
        try:
            self.collection.create_index("doc_id")
            self.collection.create_index("filename")
            logger.info("✓ Indexes created")
        except Exception as e:
            logger.warning(f"Index warning: {e}")

    def test_connection(self):
        try:
            self.client.admin.command("ping")
            logger.info("✓ MongoDB connected")
            return True
        except ConnectionFailure as e:
            logger.error(f"✗ MongoDB connection failed: {e}")
            raise Exception(f"MongoDB failed: {e}")
    
    async def generate_embeddings_async(self, text: str, retry=3) -> List[float]:
        """Generate embeddings using Jina AI with retry logic"""
        logger.info(f"Generating embedding ({len(text)} chars)...")
        
        for attempt in range(retry):
            try:
                headers = {"Content-Type": "application/json"}
                if self.jina_api_key:
                    headers["Authorization"] = f"Bearer {self.jina_api_key}"
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://api.jina.ai/v1/embeddings",
                        headers=headers,
                        json={
                            "model": "jina-embeddings-v2-base-en",
                            "input": [text[:8000]]
                        }
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"Jina API error (attempt {attempt + 1}): {response.status_code}")
                        if attempt < retry - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        raise Exception(f"Jina API failed: {response.text}")
                    
                    result = response.json()
                    embedding = result["data"][0]["embedding"]
                    logger.info(f"✓ Got embedding (dim: {len(embedding)})")
                    return embedding
                    
            except httpx.TimeoutException:
                logger.warning(f"Timeout (attempt {attempt + 1})")
                if attempt < retry - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise Exception("Jina API timeout after retries")
            except Exception as e:
                logger.error(f"Embedding failed (attempt {attempt + 1}): {e}")
                if attempt < retry - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        
        raise Exception("Failed after all retries")
    
    async def store_chunks(self, chunks: List[Dict]):
        logger.info(f"Storing {len(chunks)} chunks...")
        
        try:
            # Generate embeddings with delay to avoid rate limits
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
                
                try:
                    chunk["embedding"] = await self.generate_embeddings_async(chunk["content"])
                    
                    # Small delay to avoid rate limits (Jina free tier)
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Failed to embed chunk {i+1}: {e}")
                    raise Exception(f"Embedding chunk {i+1} failed: {e}")
            
            logger.info("All embeddings generated, inserting into MongoDB...")
            result = self.collection.insert_many(chunks)
            logger.info(f"✓ Stored {len(result.inserted_ids)} chunks")
            return len(result.inserted_ids)
            
        except Exception as e:
            logger.error(f"store_chunks failed: {e}", exc_info=True)
            raise
    
    async def similarity_search(self, query: str, num_results: int = 3) -> List[Dict]:
        logger.info(f"Searching for: {query[:50]}...")
        
        try:
            query_embedding = await self.generate_embeddings_async(query)
            
            logger.info("Fetching documents...")
            all_docs = list(self.collection.find({}))
            logger.info(f"Found {len(all_docs)} total chunks")
            
            if not all_docs:
                logger.warning("No documents in database")
                return []
            
            logger.info("Calculating similarities...")
            similarities = []
            for doc in all_docs:
                if 'embedding' in doc:
                    similarity = self._cosine_similarity(query_embedding, doc['embedding'])
                    similarities.append((doc, similarity))
            
            similarities.sort(reverse=True, key=lambda x: x[1])
            logger.info(f"✓ Found {len(similarities)} similar chunks")
            
            results = []
            for doc, similarity in similarities[:num_results]:
                results.append({
                    'content': doc['content'],
                    'filename': doc['filename'],
                    'chunk_index': doc['chunk_index'],
                    'similarity_score': float(similarity),
                    'metadata': doc.get('metadata', {}) 
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    async def list_documents(self) -> List[Dict]:
        logger.info("Listing documents...")
        pipeline = [
            {"$group": {
                "_id": "$doc_id",
                "filename": {"$first": "$filename"},
                "num_chunks": {"$sum": 1},
                "uploaded_at": {"$first": "$uploaded_at"}
            }},
            {"$project": {
                "_id": 0,
                "doc_id": "$_id", 
                "filename": 1,
                "num_chunks": 1,
                "uploaded_at": 1
            }}
        ]
        docs = list(self.collection.aggregate(pipeline))
        logger.info(f"✓ Found {len(docs)} documents")
        return docs
    
    async def delete_document(self, doc_id: str) -> int:
        logger.info(f"Deleting document {doc_id}...")
        result = self.collection.delete_many({"doc_id": doc_id})
        logger.info(f"✓ Deleted {result.deleted_count} chunks")
        return result.deleted_count 
    
    async def get_stats(self) -> Dict:
        total_chunks = self.collection.count_documents({})
        unique_docs = len(await self.list_documents())
        
        return {
            "total_chunks": total_chunks,
            "total_documents": unique_docs,
            "database": "MongoDB Atlas",
            "embedding_model": "jina-embeddings-v2-base-en",
            "embedding_dimension": 768
        }
    
    def close(self):
        self.client.close()
        logger.info("✓ Connection closed")