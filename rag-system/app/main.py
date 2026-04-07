"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings, PROJECT_ROOT
from app.models import HealthResponse
from app.services.embedder import embedder
from app.services.vectorstore import vector_store
from app.routers import documents, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources on startup, clean up on shutdown."""
    # ── Startup ──────────────────────────────────────────────
    print("=" * 60)
    print("  RAG System — Starting up …")
    print("=" * 60)

    # Load the embedding model
    embedder.load()

    # Initialise (or load) the FAISS index
    vector_store.init_index(dimension=embedder.dimension)

    print(f"  Embedding model : {settings.EMBEDDING_MODEL}")
    print(f"  LLM             : {settings.OLLAMA_MODEL} @ {settings.OLLAMA_BASE_URL}")
    print(f"  Indexed chunks  : {vector_store.total_chunks}")
    print("=" * 60)
    print("  Ready! Open http://localhost:8000 in your browser.")
    print("=" * 60)

    yield

    # ── Shutdown ─────────────────────────────────────────────
    print("[Shutdown] Saving FAISS index …")
    vector_store.save()
    print("[Shutdown] Done.")


# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="RAG System",
    description="Retrieval-Augmented Generation with FastAPI, FAISS, SentenceTransformers & Ollama/Mistral",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (allow the static frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(documents.router)
app.include_router(query.router)


# ── Health check (must be registered BEFORE the static catch-all) ──
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.OLLAMA_MODEL,
        documents_indexed=vector_store.total_chunks,
    )


# Serve the frontend (catch-all mount — must be LAST)
static_dir = PROJECT_ROOT / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
