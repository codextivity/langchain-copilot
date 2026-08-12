# app/api/routes/documents.py
# Lists and manages ingested documents.

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

class DocumentInfo(BaseModel):
    file_name: str
    chunk_count: int

class DocumentsResponse(BaseModel):
    total_chunks: int
    documents: list[DocumentInfo]

@router.get("", response_model=DocumentsResponse)
async def list_documents(request: Request):
    """
    Returns a list of all ingested documents and their chunk counts.
    """
    vectorstore = request.app.state.vectorstore

    if vectorstore is None:
        return DocumentsResponse(total_chunks=0, documents=[])

    all_docs = vectorstore.get()
    total_chunks = len(all_docs["ids"])

    # Count chunks per document
    chunk_counts: dict[str, int] = {}
    for metadata in all_docs["metadatas"]:
        name = metadata.get("file_name", "unknown")
        chunk_counts[name] = chunk_counts.get(name, 0) + 1

    documents = [
        DocumentInfo(file_name=name, chunk_count=count)
        for name, count in chunk_counts.items()
    ]

    return DocumentsResponse(
        total_chunks=total_chunks,
        documents=documents
    )