"""
Async Ollama LLM client for generating answers from retrieved context.
"""

import httpx

from app.config import settings

SYSTEM_PROMPT = """You are a helpful, accurate AI assistant. Answer the user's question based ONLY on the provided context below. Follow these rules strictly:

1. If the context contains the answer, provide a clear, well-structured response.
2. If the context does NOT contain enough information to answer, say: "I don't have enough information in the provided documents to answer this question."
3. Always cite which source document(s) your answer came from.
4. Do NOT make up information that is not in the context.
5. Keep your answers concise but thorough."""


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build the full prompt with context and question."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"--- Source: {chunk['source']} (chunk {chunk['chunk_index']}) ---\n"
            f"{chunk['text']}"
        )
    context_str = "\n\n".join(context_parts)

    return (
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


async def generate_answer(
    question: str,
    context_chunks: list[dict],
) -> str:
    """
    Call Ollama's /api/generate endpoint and return the full response.
    Uses streaming internally for reliability but collects the full text.
    """
    prompt = _build_prompt(question, context_chunks)

    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 1024,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()


async def generate_answer_stream(
    question: str,
    context_chunks: list[dict],
):
    """
    Streaming version — yields response tokens as they arrive.
    Used by the SSE endpoint.
    """
    prompt = _build_prompt(question, context_chunks)

    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 1024,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
