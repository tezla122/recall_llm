"""Text chunking and Ollama embedding (nomic-embed-text) with fixed batch size."""

from __future__ import annotations

from datetime import datetime
import ollama

OLLAMA_EMBED_MODEL = "nomic-embed-text"
_EMBED_OPTIONS = {"num_thread": 6}


def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Split *text* into overlapping character windows."""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_embed_text(raw_text: str, caption: str, ts: datetime) -> str:
    """Fuse OCR text, vision caption, and capture time for embedding."""
    ts_iso = ts.isoformat() if isinstance(ts, datetime) else str(ts)
    return (
        f"[Captured: {ts_iso}]\n"
        f"[Caption]: {caption}\n\n"
        f"[Text]:\n{raw_text}"
    )


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed *texts* with ``nomic-embed-text``.

    Texts are processed in **batches of up to eight** (full groups of eight until the
    remainder). Each string is embedded sequentially within a batch with
    ``{"num_thread": 6}`` to match your CPU budget.
    """
    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), 8):
        batch = texts[i : i + 8]
        for text in batch:
            # ollama>=0.4: deprecated `embeddings(input=...)` → use `prompt=` or `embed(input=...)`.
            resp = ollama.embeddings(
                model=OLLAMA_EMBED_MODEL,
                prompt=text,
                options=_EMBED_OPTIONS,
            )
            out.append(resp["embedding"])
    return out
