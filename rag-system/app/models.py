"""
Pydantic models for request / response validation.
"""

from pydantic import BaseModel, Field


# ── Query ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")


class SourceChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str


# ── Documents ────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_created: int
    total_chunks_in_index: int
    message: str


# ── Health ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_model: str
    documents_indexed: int
