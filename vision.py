"""Moondream2 image captioning on CPU (float32) to avoid GPU VRAM limits."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

_MODEL_LOCK = threading.Lock()
_model = None

VISION_SEMAPHORE = asyncio.Semaphore(1)

MOONDREAM_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-06-21"


def _ensure_model():
    global _model
    with _MODEL_LOCK:
        if _model is None:
            _model = AutoModelForCausalLM.from_pretrained(
                MOONDREAM_ID,
                revision=MOONDREAM_REVISION,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
                trust_remote_code=True,
            )
    return _model


def _normalize_caption_piece(out: object) -> str:
    if isinstance(out, dict):
        cap = out.get("caption", "")
    else:
        cap = out
    if isinstance(cap, str):
        return cap.strip()
    if isinstance(cap, bytes):
        return cap.decode("utf-8", errors="replace").strip()
    try:
        return "".join(str(x) for x in cap).strip()  # type: ignore[arg-type]
    except TypeError:
        return str(cap).strip()


def _normalize_query_answer(out: object) -> str:
    if isinstance(out, dict):
        ans = out.get("answer") or out.get("response") or ""
    else:
        ans = out
    if isinstance(ans, str):
        return ans.strip()
    return str(ans).strip() if ans else ""


def _load_pil(path: Path) -> Image.Image:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".pdf":
        doc = fitz.open(path)
        try:
            if len(doc) == 0:
                raise ValueError(f"PDF has no pages: {path}")
            page = doc[0]
            pix = page.get_pixmap(dpi=200)
            mode = "RGBA" if pix.n == 4 else "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        finally:
            doc.close()
        return img.convert("RGB")
    img = Image.open(path)
    return img.convert("RGB")


def _caption_inference_sync(path: Path) -> str:
    model = _ensure_model()
    # PIL image only; no .to("cuda") — weights live on CPU via device_map.
    image = _load_pil(path)
    parts: list[str] = []
    try:
        out = model.caption(image, length="long")
    except Exception:
        out = model.caption(image, length="normal")
    cap = _normalize_caption_piece(out)
    if cap:
        parts.append(cap)
    q_fn = getattr(model, "query", None)
    if callable(q_fn):
        try:
            q_out = q_fn(
                image,
                "What are the main colors of the main subject (coat, fur, skin, clothing)? "
                "What animals, people, or objects are clearly visible? Answer in 2 short sentences.",
            )
            qa = _normalize_query_answer(q_out)
            if qa:
                parts.append(f"[Visual details]: {qa}")
        except Exception:
            pass
    return "\n\n".join(parts) if parts else ""


async def caption_image(img_path: Path) -> str:
    """Caption an image or the first page of a PDF; serialized to avoid CPU overload."""
    async with VISION_SEMAPHORE:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            _caption_inference_sync,
            Path(img_path),
        )
