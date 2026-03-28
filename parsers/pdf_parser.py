"""PDF text extraction with optional OCR fallback (CPU-bound workers)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fitz
from PIL import Image
import pytesseract

_executor = ProcessPoolExecutor(max_workers=6)


def _ocr_page(pdf_path: str, page_index: int) -> tuple[int, str]:
    """Extract text from one page; OCR at 200 DPI if embedded text is too short."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        text = page.get_text() or ""
        if len(text.strip()) >= 50:
            return (page_index, text)
        pix = page.get_pixmap(dpi=200)
        mode = "RGBA" if pix.n == 4 else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if img.mode != "RGB":
            img = img.convert("RGB")
        ocr_text = pytesseract.image_to_string(img)
        return (page_index, ocr_text)
    finally:
        doc.close()


async def parse_pdf_async(path: Path) -> str:
    """Run :func:`_ocr_page` for every page in parallel via the process pool."""
    path = Path(path).resolve()
    loop = asyncio.get_running_loop()
    doc = fitz.open(path)
    try:
        n = len(doc)
    finally:
        doc.close()
    if n == 0:
        return ""
    pdf_str = str(path)
    futures = [
        loop.run_in_executor(_executor, _ocr_page, pdf_str, i) for i in range(n)
    ]
    parts = await asyncio.gather(*futures)
    parts.sort(key=lambda t: t[0])
    return "\n\n".join(p for _, p in parts)
