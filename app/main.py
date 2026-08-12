# app/main.py

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from app.api.routes import health, ingest, chat, extract, documents
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")

    from app.core.ingestion import load_existing_vectorstore, ingest as ingest_pdf
    from app.core.agent import build_research_agent

    chroma_path = Path(settings.chroma_path)

    if chroma_path.exists() and any(chroma_path.iterdir()):
        # Vector store exists and has data — load it normally
        print(f"Vector store loaded from {settings.chroma_path}")
        app.state.vectorstore = load_existing_vectorstore()
        app.state.agent = build_research_agent(app.state.vectorstore)

    else:
        # No vector store found — check for sample PDFs to auto-ingest
        # This handles the case where the service restarts on Render
        # and the ephemeral chroma_db/ is wiped
        sample_dir = Path("samples")
        sample_pdfs = list(sample_dir.glob("*.pdf")) if sample_dir.exists() else []

        if sample_pdfs:
            # Auto-ingest the first sample PDF found
            print(f"No vector store found. Auto-ingesting: {sample_pdfs[0].name}")
            app.state.vectorstore = ingest_pdf(str(sample_pdfs[0]))
            app.state.agent = build_research_agent(app.state.vectorstore)
            print("Sample document ingested successfully.")
        else:
            # No sample PDF either — start empty
            # User must call POST /ingest manually
            print("No vector store found. Use POST /ingest to add documents.")
            app.state.vectorstore = None
            app.state.agent = None

    yield

    print("Shutting down...")


app = FastAPI(
    title="LangChain Research Copilot",
    description="RAG-powered research assistant with tool calling",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(extract.router, prefix="/extract", tags=["Extraction"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])