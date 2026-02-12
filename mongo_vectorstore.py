from pymongo import MongoClient 
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer 
import os 
from typing import List, Dict 
import numpy as np

class MongoDBVectorStore:
    """
    MongoDB vector store with vector search capabilities
    Uses MongoDB Atlas Vector Search for semantic similarity  

    """
    def __init__(self):
        # connect to mongodb atlas 
        self.connection_string = os.getenv("MONGO_URI")
        if not self.connection_string:
            raise ValueError("MONGO_URI Variable Not Set")
        self.client = MongoClient(self.connection_string)

        self.db = self.client["rag_qa_system"]
        self.collection = self.db["documents"]


        # initialize embedding model

        print("Loading Embedding Model")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding Model Loaaded successfully!")

        # create indexes 
        self._setup_indexes()

    def _setup_indexes(self):

        try:
            self.collection.create_index("doc_id")
            self.collection.create_index("filename")

            print("Indexes created successfully!")
            # Note: Vector search index must be created in MongoDB Atlas UI 
            # Instructions in the README file 
        except Exception as e:
            print(f"Failed to create indexes: {str(e)}")

    def test_connection():
        try:
            self.client.admin.command("ping")
            return True
        except ConnectionFailure:
            raise Exception("Failed To Connect To MongoDB Atlas")
    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeding Vector for text using hugging face model
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    async def store_chunks(self, chunks: List[Dict]):
        for chunk in chunks:
            # generate embedding for all chunks 
            chunk["embeddding"] = self.generate_embeddings(chunk["content"])
            # Insert into MongoDB 
            result = self.collection.insert_many(chunks)
            return len(result.inserted_ids)
    async def similarity_search(self, query: str, num_results: int = 3) -> List[Dict]:
        """
        Perform vector similarity search 
        Note: This uses basic cosine similarity. For production with large datasets
        use MongoDB Atlas Vector Search index for performance
        """
        query_embedding = self.generate_embeddings(query)
        # Get all documents (for small datasets)
        # For production: use $vectorSearch aggregation with Atlas vector Search Index
        if not all_docs:
            return []
        # calculate cosine similarity 
        similarities = []
        for doc in all_docs:
            if 'embedding' in doc:
                similarity = self.cosine_similarity(query_embedding, doc['embedding'])
                similarities.append((doc, similarity))
        # sort by similarity 
        similarities.sort(reverse=True, key=lambda x: x[0])
        results = []
        for similarity, doc in similarities[:num_results]:
            results.append({
                'content': doc['content'],
                'filename': doc['filename'],
                'chunk_index': doc[chunk_index],
                'similarity_score': float(similarity),
                'metadata': doc.get('metadata', {}) 
            })
            return results
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    async def lis_documents(self) -> List[Dict]:
        """
        list unique documents

        """
        pipeline = [
            {
                "$group":{
                    "_id": "$doc_id",
                    "filename": {"$first":  "$filename"},
                    "num_chunks": {"$sum": 1},
                    "uploaded_at": {"$first": "$uploaded_at"}

                }
            },{
                "$project": {
                    "_id": 0,
                    "doc_id": "$_id", 
                    "filename": 1,
                    "num_chunks": 1,
                    "uploaded_at": 1
                }
            }
        ]
        return list(self.collection.aggregate(pipeline))
    async def delete_document(self, doc_id: str) -> int:
        # delete all chunks of a document
        result = self.collection.delete_many({"doc_id": doc_is})
        return result.deleted_count 
    async def get_stats(self) -> Dict:
        """ Get database statistics"""
        total_chunks = self.collection.count_documents({})
        # count unique documents 
        unique_docs = len(await self.list_documents())
        return{
            "total_chunks": total_chunks,
            "total_documents": unique_docs,
            "database": "MongoBD Atlas",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384
        }
    def close(self):
        # close mongodb connection 
        self.client.close()







        

        





    

    





