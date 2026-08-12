# Dockerfile
# Builds a container image for the LangChain Research Copilot API.

# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim is the official Python image without unnecessary extras.
# "slim" removes build tools and documentation — reduces image size by ~200MB.
# We use 3.11 specifically because some dependencies have issues with 3.12+.
FROM python:3.11-slim

# ── Set working directory ─────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container.
# This is the standard convention for Python web applications.
WORKDIR /app

# ── Install system dependencies ───────────────────────────────────────────────
# These are needed by ChromaDB and PDF processing libraries.
# We install them before Python packages to leverage Docker's layer cache —
# system deps change less often than Python deps.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/* removes the apt cache to keep image size small

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first — before copying the rest of the code.
# Why? Docker caches each layer. If requirements.txt has not changed,
# Docker skips this step on rebuild. If we copied all code first,
# any code change would invalidate the cache and reinstall everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir prevents pip from storing download cache in the image

# ── Copy application code ─────────────────────────────────────────────────────
COPY app/ ./app/

# ── Create data directory ─────────────────────────────────────────────────────
# This directory will hold the ChromaDB files inside the container.
# In docker-compose we mount a volume here so data persists across restarts.
RUN mkdir -p /app/chroma_db

# ── Expose port ───────────────────────────────────────────────────────────────
# Documents which port the container listens on.
# Does not actually publish the port — that happens in docker-compose.
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# --host 0.0.0.0 makes the server accessible from outside the container.
# Without this, the server only listens on localhost inside the container
# and your requests from the host machine would never reach it.
# --workers 1 keeps it simple — one process, no shared state issues with ChromaDB.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]