FROM python:3.11-slim

WORKDIR /app

# Combine everything into one layer and clean up apt immediately
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    tesseract-ocr \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir PyMuPDF pytesseract Pillow \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]