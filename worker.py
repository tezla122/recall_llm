"""Async ingestion worker draining the bounded ingest queue."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from recall_llm.embedder import build_embed_text, embed_batch
from recall_llm.parsers.pdf_parser import parse_pdf_async
from recall_llm.queue_manager import ingest_queue
from recall_llm.store import upsert_memory
from recall_llm.vision import caption_image


def _stable_doc_id(path: Path) -> str:
    p = path.resolve()
    st = p.stat()
    key = f"{p}|{st.st_mtime_ns}|{st.st_size}"
    return hashlib.sha256(key.encode()).hexdigest()


def _raw_text_with_source(path: Path, body: str) -> str:
    """Prefix OCR/body with the filename so FTS matches queries like ``red-apple.jpg``."""
    prefix = f"[Source file: {path.name}]\n"
    b = body.strip()
    return prefix + b if b else prefix.rstrip()


async def _persist_document(
    path: Path,
    file_type: str,
    ocr_text: str,
    caption: str,
) -> None:
    captured_at = datetime.now(timezone.utc)
    raw_stored = _raw_text_with_source(path, ocr_text)
    embed_input = build_embed_text(raw_stored, caption, captured_at)
    vectors = await asyncio.to_thread(embed_batch, [embed_input])
    upsert_memory(
        doc_id=_stable_doc_id(path),
        captured_at=captured_at,
        doc_type=file_type,
        raw_text=raw_stored,
        caption=caption,
        image_path=str(path.resolve()),
        vector=vectors[0],
    )


async def ingest_worker() -> None:
    """Process tasks from :data:`ingest_queue` until cancelled."""
    while True:
        task = await ingest_queue.get()
        path = Path(task.path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                ocr_res, cap_res = await asyncio.gather(
                    parse_pdf_async(path),
                    caption_image(path),
                    return_exceptions=True,
                )
                if isinstance(ocr_res, Exception):
                    logger.exception("PDF OCR failed for {}", path)
                    raise ocr_res
                ocr_text = ocr_res
                if isinstance(cap_res, Exception):
                    logger.exception("PDF caption failed for {} (OCR still saved)", path)
                    caption = ""
                else:
                    caption = cap_res
                await _persist_document(path, task.file_type, ocr_text, caption)
                logger.success(
                    "Ingested PDF {} (OCR chars={}, caption len={})",
                    path,
                    len(ocr_text),
                    len(caption),
                )
            elif suffix in (".png", ".jpg", ".jpeg"):
                try:
                    caption = await caption_image(path)
                except Exception:
                    logger.exception("Image caption failed for {} (filename still indexed)", path)
                    caption = ""
                await _persist_document(path, task.file_type, "", caption)
                logger.success(
                    "Ingested image {} (caption len={})",
                    path,
                    len(caption),
                )
            else:
                logger.warning("Unsupported file type for ingest: {}", path)
        except Exception:
            logger.exception("Ingest failed for {}", path)
        finally:
            ingest_queue.task_done()
