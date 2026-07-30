FROM python:3.11-slim

# System deps needed for PyMuPDF, EasyOCR, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first to avoid pulling in huge CUDA/nvidia packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects the PORT env var at runtime (defaults to 8080)
EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}