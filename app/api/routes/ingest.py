# app/api/routes/ingest.py
# Handles PDF upload and ingestion into the vector store.

import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from app.core.ingestion import ingest, load_existing_vectorstore
from app.core.agent import build_research_agent

router = APIRouter()

class IngestResponse(BaseModel):
    message: str
    file_name: str
    chunks_added: int
    total_documents: int

@router.post("", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF file and ingest it into the vector store.

    The file is saved temporarily, ingested, then deleted.
    Duplicate detection is handled automatically — uploading the same
    file twice skips ingestion and returns the existing count.
    """
    # Validate file type before doing any work
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Save uploaded file to a temp location.
    # Why tempfile? Because we need a real file path for PyPDFLoader.
    # The temp file is automatically deleted after the with block.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run ingestion — idempotent, handles duplicates automatically
        vectorstore = ingest(tmp_path)

        # Update the shared app state so /chat uses the new documents
        request.app.state.vectorstore = vectorstore
        request.app.state.agent = build_research_agent(vectorstore)

        # Count total chunks in the store
        all_docs = vectorstore.get()
        total_chunks = len(all_docs["ids"])

    finally:
        # Always delete the temp file, even if ingestion failed
        os.unlink(tmp_path)

    return IngestResponse(
        message=f"Successfully processed {file.filename}",
        file_name=file.filename,
        chunks_added=total_chunks,
        total_documents=len(set(
            m.get("file_name", "unknown")
            for m in all_docs["metadatas"]
        ))
    )