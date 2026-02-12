from google import genai
from google.genai import types
import os
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()
class QAChain:
    """
    Question-Answering chain using Gemini (current 2025 SDK)
    
    Retrieves relevant context from MongoDB and generates answers using Gemini
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
        # Configure Gemini with current SDK
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Create client (new SDK approach)
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash'  # Current model
        
        # System prompt for RAG
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Only use information from the provided context to answer questions
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the documents to answer that question."
3. Always cite which document the information comes from
4. Be concise but thorough
5. If you're not certain, express your uncertainty

Context will be provided with each question."""
    
    async def ask(self, question: str, num_sources: int = 3) -> Dict:
        """
        Answer a question using RAG
        
        Steps:
        1. Retrieve relevant context from MongoDB
        2. Format context with question
        3. Generate answer using Gemini
        4. Return answer with sources
        """
        
        # Step 1: Retrieve relevant context
        search_results = await self.vector_store.similarity_search(
            question, 
            num_results=num_sources
        )
        
        if not search_results:
            return {
                "answer": "No documents have been uploaded yet. Please upload documents first before asking questions.",
                "sources": [],
                "confidence": "none"
            }
        
        # Step 2: Format context
        context = self._format_context(search_results)
        
        # Step 3: Create prompt
        prompt = f"""{self.system_prompt}

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

ANSWER (cite sources):"""
        
        # Step 4: Generate answer using Gemini (new SDK)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            answer = response.text
            
            # Determine confidence based on similarity scores
            avg_similarity = sum(r['similarity_score'] for r in search_results) / len(search_results)
            confidence = self._get_confidence_level(avg_similarity)
            
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            confidence = "error"
        
        # Step 5: Format response
        return {
            "answer": answer,
            "sources": [
                {
                    "filename": result["filename"],
                    "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "similarity_score": round(result["similarity_score"], 3),
                    "chunk_index": result["chunk_index"]
                }
                for result in search_results
            ],
            "confidence": confidence
        }
    
    def _format_context(self, search_results: List[Dict]) -> str:
        """Format search results into context string"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"[Source {i} - {result['filename']} (chunk {result['chunk_index']})]:\n"
                f"{result['content']}\n"
                f"(Relevance: {result['similarity_score']:.2f})\n"
            )
        
        return "\n".join(context_parts)
    
    def _get_confidence_level(self, avg_similarity: float) -> str:
        """Determine confidence level based on average similarity score"""
        if avg_similarity >= 0.7:
            return "high"
        elif avg_similarity >= 0.5:
            return "medium"
        else:
            return "low"