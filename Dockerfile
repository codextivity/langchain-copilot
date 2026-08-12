# Dockerfile — works on both Render and Hugging Face Spaces

FROM python:3.11-slim

# Create non-root user
# Hugging Face requires this for security
# Render works fine with it too — good practice everywhere
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy requirements first to leverage Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY samples/ ./samples/

# Create directories and set permissions
RUN mkdir -p /data/chroma_db && \
    chown -R appuser:appuser /app /data

USER appuser

# Document both possible ports
# Render uses $PORT (dynamic), Hugging Face uses 7860
EXPOSE 7860 8000

# PORT environment variable controls which port to use
# Render injects $PORT automatically
# Hugging Face does not inject PORT so it falls back to 7860
# Local development uses 8000 (set in .env)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]