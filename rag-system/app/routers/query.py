"""
Query router – ask questions and get RAG-augmented answers.
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models import QueryRequest, QueryResponse, SourceChunk
from app.services.embedder import embedder
from app.services.vectorstore import vector_store
from app.services.llm import generate_answer, generate_answer_stream

router = APIRouter(prefix="/api/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """
    Embed the question, retrieve relevant chunks from FAISS,
    and generate an answer via Ollama.
    """
    if vector_store.total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload documents first.",
        )

    # Embed question (CPU-bound)
    q_embedding = await asyncio.to_thread(
        embedder.encode, [req.question]
    )

    # Retrieve
    results = vector_store.search(q_embedding[0], top_k=settings.TOP_K)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    # Generate answer
    try:
        answer = await generate_answer(req.question, results)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM generation failed. Is Ollama running? Error: {e}",
        )

    sources = [
        SourceChunk(
            text=r["text"],
            source=r["source"],
            chunk_index=r["chunk_index"],
            score=r["score"],
        )
        for r in results
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        model=settings.OLLAMA_MODEL,
    )


@router.post("/stream")
async def query_documents_stream(req: QueryRequest):
    """
    Streaming version — returns an SSE stream of answer tokens
    followed by source metadata.
    """
    if vector_store.total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet.",
        )

    q_embedding = await asyncio.to_thread(
        embedder.encode, [req.question]
    )
    results = vector_store.search(q_embedding[0], top_k=settings.TOP_K)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    async def event_stream():
        try:
            async for token in generate_answer_stream(req.question, results):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Send sources at the end
            sources = [
                {
                    "text": r["text"],
                    "source": r["source"],
                    "chunk_index": r["chunk_index"],
                    "score": r["score"],
                }
                for r in results
            ]
            yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
