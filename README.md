# RAG Document Q&A System

An intelligent document question-answering system using **Retrieval Augmented Generation (RAG)**. Upload PDFs, DOCX, or text files and ask questions about their content.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen)
![Gemini](https://img.shields.io/badge/Gemini-Pro-orange)

## Features

- **Upload Documents**: Support for PDF, DOCX, and TXT files
- **Semantic Search**: Uses HuggingFace embeddings for intelligent document retrieval
- **AI-Powered Answers**: Gemini Pro generates accurate, context-aware responses
- **Vector Storage**: MongoDB Atlas for scalable vector search
- **RESTful API**: Clean, documented API with automatic Swagger docs
- **Source Citations**: All answers include source documents and confidence scores

##  Architecture

```
User Upload (PDF/DOCX/TXT)
    ↓
Document Processor (splits into chunks)
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
    ↓
MongoDB Atlas (vector storage)
    ↓
User Query → Vector Search → Top 3 relevant chunks
    ↓
Gemini Pro (generates answer with context)
    ↓
Answer + Sources
```

##  Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: MongoDB Atlas (with vector search)
- **Embeddings**: HuggingFace Sentence Transformers (free, runs locally)
- **AI Model**: Google Gemini 2.5 Flash (current 2025 SDK - `google-genai`)
- **Document Processing**: PyPDF2, python-docx

##  Prerequisites

1. **Python 3.10+**
2. **MongoDB Atlas Account** (free tier works great)
   - [Sign up here](https://www.mongodb.com/cloud/atlas/register)
   - Get your connection string
3. **Gemini API Key** (free tier: 60 requests/minute)
   - [Get key here](https://makersuite.google.com/app/apikey)

##  Quick Start

### 1. Clone/Download the Project

```bash
# If you have git
git clone <your-repo-url>
cd rag-qa-system

# Or just download the folder
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your keys:
# - MONGODB_URI (your MongoDB Atlas connection string)
# - GEMINI_API_KEY (your Google Gemini API key)
```

Your `.env` should look like:
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
GEMINI_API_KEY=AIzaSy...your_key_here
```

### 4. Run the Application

```bash
python main.py
```

The API will start at: **http://localhost:8000**

API Documentation: **http://localhost:8000/docs** (interactive Swagger UI)

##  API Usage

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "doc_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "your_document.pdf",
  "num_chunks": 15,
  "message": "Document uploaded and indexed successfully with 15 chunks"
}
```

### 2. Ask a Question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings?",
    "num_sources": 3
  }'
```

Response:
```json
{
  "answer": "Based on the documents, the main findings are...",
  "sources": [
    {
      "filename": "your_document.pdf",
      "content": "Relevant text excerpt...",
      "similarity_score": 0.85,
      "chunk_index": 3
    }
  ],
  "confidence": "high"
}
```

### 3. List Documents

```bash
curl "http://localhost:8000/documents"
```

### 4. Delete a Document

```bash
curl -X DELETE "http://localhost:8000/documents/{doc_id}"
```

##  Testing with Swagger UI

1. Go to **http://localhost:8000/docs**
2. You'll see an interactive API documentation
3. Click "Try it out" on any endpoint
4. Upload files and ask questions directly from the browser!

## 💡 Example Usage

```python
import requests

# Upload a document
with open('research_paper.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f}
    )
    print(response.json())

# Ask a question
response = requests.post(
    'http://localhost:8000/query',
    json={
        'question': 'What is the methodology used in this research?',
        'num_sources': 3
    }
)
print(response.json()['answer'])
```

## 🔧 Configuration

You can modify these settings in the code:

**Document Processor** (`document_processor.py`):
- `chunk_size`: Size of text chunks (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks (default: 200 characters)

**QA Chain** (`qa_chain.py`):
- `num_sources`: Number of relevant chunks to retrieve (default: 3)

**MongoDB** (`mongodb_vectorstore.py`):
- `embedding_model`: Change embedding model (default: 'all-MiniLM-L6-v2')

## 🚀 Deployment

### Deploy to Railway

1. Sign up at [Railway.app](https://railway.app)
2. Create new project from GitHub
3. Add environment variables in Railway dashboard
4. Deploy automatically!

### Deploy to Render

1. Sign up at [Render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Add environment variables
5. Deploy!

## 📊 Performance

- **Embedding Generation**: ~100ms per chunk (runs locally)
- **Vector Search**: ~50-200ms depending on database size
- **Gemini Response**: ~1-3 seconds
- **Total Response Time**: ~2-5 seconds

For large document collections (1000+ documents), consider:
- Using MongoDB Atlas Vector Search index (requires setup in Atlas UI)
- Caching frequent queries
- Implementing async processing for uploads

## 🧪 Testing

Create a test file:

```python
# test_api.py
import requests
import os

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    print("✓ Health check passed")

def test_upload():
    # Create a test file
    with open("test_doc.txt", "w") as f:
        f.write("This is a test document about artificial intelligence.")
    
    with open("test_doc.txt", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": f}
        )
    
    assert response.status_code == 200
    print("✓ Upload test passed")
    
    # Cleanup
    os.remove("test_doc.txt")
    return response.json()["doc_id"]

def test_query(doc_id):
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": "What is this document about?"}
    )
    
    assert response.status_code == 200
    assert "artificial intelligence" in response.json()["answer"].lower()
    print("✓ Query test passed")

if __name__ == "__main__":
    test_health()
    doc_id = test_upload()
    test_query(doc_id)
    print("\n✓ All tests passed!")
```

Run tests:
```bash
python test_api.py
```

## 🎯 Use Cases

- **Research Assistant**: Upload research papers, ask about methodologies
- **Legal Document Analysis**: Query contracts and legal documents
- **Technical Documentation**: Quick answers from technical manuals
- **Study Aid**: Upload textbooks and lecture notes
- **Business Intelligence**: Query reports and business documents

## 🛣️ Roadmap

- [ ] Add support for more file types (Excel, CSV)
- [ ] Implement document summarization
- [ ] Add user authentication
- [ ] Create web interface
- [ ] Support for images in PDFs
- [ ] Multi-language support
- [ ] Export Q&A history

## 🐛 Troubleshooting

**Issue**: "MONGODB_URI not found"
- **Solution**: Make sure you created `.env` file and added your connection string

**Issue**: "Failed to connect to MongoDB"
- **Solution**: Check your MongoDB Atlas connection string and whitelist your IP

**Issue**: "Gemini API error"
- **Solution**: Verify your API key at https://makersuite.google.com/app/apikey

**Issue**: "No module named 'sentence_transformers'"
- **Solution**: Run `pip install -r requirements.txt` again

**Issue**: Slow response times
- **Solution**: First query downloads the embedding model (~80MB). Subsequent queries are faster.

## 📝 Notes

- First run downloads the HuggingFace model (~80MB) - this is normal
- Free tiers are sufficient for development and demos
- For production, consider upgrading MongoDB Atlas and Gemini tiers
- The system works offline for embedding generation (only needs internet for Gemini API)

## 📄 License

MIT License - feel free to use this for your portfolio!

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome! Open an issue or submit a PR.

## 👤 Author

**Marcus Onyango**
- Portfolio: [Your GitHub Profile]
- LinkedIn: [Your LinkedIn]
- Email: [Your Email]

---

## 🎓 What This Demonstrates

This project showcases:
- ✅ Working with AI APIs (Gemini, HuggingFace)
- ✅ Vector database integration (MongoDB Atlas)
- ✅ RESTful API design (FastAPI)
- ✅ Document processing and chunking
- ✅ Retrieval Augmented Generation (RAG)
- ✅ Production-ready code structure
- ✅ Error handling and validation
- ✅ API documentation
- ✅ Cloud deployment ready

Perfect for demonstrating AI engineering skills! 🚀