# app/main.py
# FastAPI application entry point.
# This file creates the app, registers routes, and handles startup/shutdown.
# app/main.py — add these two lines at the very top, before everything else

from dotenv import load_dotenv
load_dotenv()  # Must be called before any other imports that need env vars

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.routes import health, ingest, chat, extract, documents
from app.config import settings

# ── Lifespan handler ─────────────────────────────────────────────────────────
# Code in the lifespan runs once at startup and once at shutdown.
# We use it to initialize shared resources — the vector store and agent —
# so they are ready before the first request arrives.
#
# Why not initialize inside each endpoint?
# Because loading ChromaDB and building the agent takes 1-2 seconds.
# Doing it on every request would make your API feel broken.
# Doing it once at startup means every request is fast.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    print("Starting up...")

    from pathlib import Path
    from app.core.ingestion import load_existing_vectorstore
    from app.core.agent import build_research_agent

    # Load vector store if it exists, otherwise set to None.
    # Endpoints that need it will return a helpful error if it is None.
    if Path(settings.chroma_path).exists():
        app.state.vectorstore = load_existing_vectorstore()
        app.state.agent = build_research_agent(app.state.vectorstore)
        print(f"Vector store loaded from {settings.chroma_path}")
    else:
        app.state.vectorstore = None
        app.state.agent = None
        print("No vector store found. Use POST /ingest to add documents.")

    yield  # application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print("Shutting down...")

# ── Create app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LangChain Research Copilot",
    description="RAG-powered research assistant with tool calling",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# Allows browsers to call your API from a different domain.
# During development, allow all origins.
# In production, restrict to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Change to ["https://yourfrontend.com"] in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ───────────────────────────────────────────────────────────
# Each router handles one group of related endpoints.
# The prefix is prepended to every route in that router.
app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(extract.router, prefix="/extract", tags=["Extraction"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])