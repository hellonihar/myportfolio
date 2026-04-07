"""
Document upload and ingestion router.
"""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.config import settings
from app.models import DocumentUploadResponse
from app.services.embedder import embedder
from app.services.vectorstore import vector_store
from app.services.chunker import chunk_text
from app.utils.file_parsers import extract_text, SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document file, parse it, chunk it, embed the chunks,
    and add them to the FAISS index.
    """
    # Validate extension
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Save uploaded file
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename

    content = await file.read()
    file_path.write_bytes(content)

    # Extract text
    try:
        raw_text = await asyncio.to_thread(extract_text, file_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No text content extracted from file.")

    # Chunk
    chunks = chunk_text(raw_text, source=filename)
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced from file.")

    # Embed (CPU-bound, run in thread)
    texts = [c["text"] for c in chunks]
    embeddings = await asyncio.to_thread(embedder.encode, texts)

    # Store
    metadatas = [
        {"text": c["text"], "source": c["source"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]
    vector_store.add(embeddings, metadatas)
    vector_store.save()

    return DocumentUploadResponse(
        filename=filename,
        chunks_created=len(chunks),
        total_chunks_in_index=vector_store.total_chunks,
        message=f"Successfully ingested '{filename}' — {len(chunks)} chunks added.",
    )


@router.get("/")
async def list_documents():
    """Return a list of all uploaded documents."""
    upload_dir = Path(settings.UPLOAD_DIR)
    if not upload_dir.exists():
        return {"documents": [], "total_chunks": 0}

    files = [
        {"name": f.name, "size_bytes": f.stat().st_size}
        for f in sorted(upload_dir.iterdir())
        if f.is_file()
    ]
    return {
        "documents": files,
        "total_chunks": vector_store.total_chunks,
    }
