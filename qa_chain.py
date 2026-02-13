import httpx
import os
from typing import Dict, List
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class QAChain:
    """Question-Answering using Groq"""
    
    def __init__(self, vector_store):
        logger.info("Initializing QA chain...")
        self.vector_store = vector_store
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        self.model = "llama-3.3-70b-versatile"
        logger.info("✓ QA chain initialized")
        
        self.system_prompt = """Answer questions based ONLY on the context provided.
Rules: Only use context info. If insufficient, say "I don't have enough information." Cite sources."""
    
    async def ask(self, question: str, num_sources: int = 3) -> Dict:
        logger.info(f"Processing question: {question[:50]}...")
        
        # Get relevant chunks
        search_results = await self.vector_store.similarity_search(
            question, 
            num_results=num_sources
        )
        
        if not search_results:
            logger.warning("No search results found")
            return {
                "answer": "No documents uploaded yet.",
                "sources": [],
                "confidence": "none"
            }
        
        # Format context
        context = self._format_context(search_results)
        logger.info(f"Context prepared ({len(context)} chars)")
        
        # Generate answer
        try:
            logger.info("Calling Groq API...")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1000
                    }
                )
                
                logger.info(f"Groq response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Groq error: {response.text}")
                    raise Exception(f"Groq API failed: {response.text}")
                
                answer = response.json()["choices"][0]["message"]["content"]
                logger.info(f"✓ Got answer ({len(answer)} chars)")
                
            avg_similarity = sum(r['similarity_score'] for r in search_results) / len(search_results)
            confidence = self._get_confidence(avg_similarity)
            
        except httpx.TimeoutException:
            logger.error("✗ Groq API timeout")
            answer = "Error: API timeout"
            confidence = "error"
        except Exception as e:
            logger.error(f"✗ Answer generation failed: {e}")
            answer = f"Error: {str(e)}"
            confidence = "error"
        
        return {
            "answer": answer,
            "sources": [
                {
                    "filename": r["filename"],
                    "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                    "similarity_score": round(r["similarity_score"], 3),
                    "chunk_index": r["chunk_index"]
                }
                for r in search_results
            ],
            "confidence": confidence
        }
    
    def _format_context(self, results: List[Dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Source {i} - {r['filename']} (chunk {r['chunk_index']})]:\n"
                f"{r['content']}\n"
            )
        return "\n".join(parts)
    
    def _get_confidence(self, avg_sim: float) -> str:
        if avg_sim >= 0.7:
            return "high"
        elif avg_sim >= 0.5:
            return "medium"
        return "low"