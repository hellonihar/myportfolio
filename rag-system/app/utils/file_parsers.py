"""
File content extraction utilities.
Supports: .txt, .md, .pdf
"""

from pathlib import Path

from PyPDF2 import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(file_path: str | Path) -> str:
    """
    Extract plain text from a file based on its extension.

    Raises:
        ValueError: if the file type is not supported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        return _extract_pdf(path)

    raise ValueError(
        f"Unsupported file type: {ext}. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _extract_pdf(path: Path) -> str:
    """Extract text from all pages of a PDF."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)
