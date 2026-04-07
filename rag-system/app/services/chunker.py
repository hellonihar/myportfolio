"""
Recursive character text splitter for document chunking.
"""

from app.config import settings


def chunk_text(
    text: str,
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Split *text* into overlapping chunks and attach metadata.

    Uses a hierarchy of separators (double-newline → single-newline →
    sentence-ending punctuation → space) to keep semantically coherent
    pieces together.

    Returns:
        list of { "text": str, "source": str, "chunk_index": int }
    """
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP
    separators = ["\n\n", "\n", ". ", " "]

    chunks = _recursive_split(text, separators, size)

    # Apply overlap
    merged: list[str] = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            prev_tail = chunks[i - 1][-overlap:]
            chunk = prev_tail + chunk
        stripped = chunk.strip()
        if stripped:
            merged.append(stripped)

    return [
        {"text": c, "source": source, "chunk_index": i}
        for i, c in enumerate(merged)
    ]


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
) -> list[str]:
    """Recursively split text using the first separator that produces chunks."""
    if len(text) <= chunk_size:
        return [text]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # If a single part exceeds chunk_size, split deeper
                    if len(part) > chunk_size:
                        remaining_seps = separators[separators.index(sep) + 1 :]
                        if remaining_seps:
                            chunks.extend(
                                _recursive_split(part, remaining_seps, chunk_size)
                            )
                        else:
                            # Hard split on character boundary
                            for j in range(0, len(part), chunk_size):
                                chunks.append(part[j : j + chunk_size])
                        current = ""
                    else:
                        current = part
            if current:
                chunks.append(current)
            return chunks

    # Fallback: hard character split
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
