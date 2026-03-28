"""RAG answer generation with bounded context and streaming Ollama output."""

from __future__ import annotations

import os
from typing import Generator

import ollama
from ollama import ResponseError

from recall_llm.retriever import hybrid_search

# Override with: RECALL_LLM_CHAT_MODEL=llama3.2:1b ./venv/bin/python main.py
CHAT_MODEL = os.environ.get("RECALL_LLM_CHAT_MODEL", "phi3.5").strip() or "phi3.5"

_CONTEXT_TOP_N = 3
_MAX_TOKENS_PER_CHUNK = 600
# Rough heuristic (~4 chars/token) to avoid heavy tokenizers while staying under budget.
_CHARS_PER_TOKEN_EST = 4


def _truncate_to_token_budget(text: str, max_tokens: int = _MAX_TOKENS_PER_CHUNK) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN_EST
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def build_context(results: list[dict]) -> str:
    """
    Format retrieved rows for the LLM.

    Uses only the **top 3** fused hits and applies a **~600 token** cap per chunk
    (character-based estimate) to keep prefill small on CPU.
    """
    if not results:
        return "No relevant documents were retrieved."
    parts: list[str] = []
    for i, row in enumerate(results[:_CONTEXT_TOP_N], start=1):
        doc_id = row.get("doc_id", "")
        caption = str(row.get("caption", "") or "")
        raw = str(row.get("raw_text", "") or "")
        body = _truncate_to_token_budget(f"Caption: {caption}\n\nText:\n{raw}")
        parts.append(f"### Source {i} (doc_id={doc_id})\n{body}")
    return "\n\n".join(parts)


def answer_stream(
    query: str,
    *,
    retrieval_results: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Retrieve context, then stream tokens from Ollama (see ``CHAT_MODEL`` / env)."""
    results = (
        retrieval_results if retrieval_results is not None else hybrid_search(query, k=5)
    )
    context = build_context(results)
    prompt = (
        "You are Recall-LLM. Answer clearly using the context when it helps. "
        "The context may include Captions and [Visual details] from ingested images—use those for colors, objects, and scenes.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    stream = ollama.generate(
        model=CHAT_MODEL,
        prompt=prompt,
        stream=True,
        options={"num_thread": 6, "num_predict": 512, "temperature": 0.1},
    )
    try:
        for chunk in stream:
            piece = chunk.get("response")
            if piece:
                yield piece
    except ResponseError as e:
        # Streaming raises while iterating; 404 or Ollama "model ... not found" body.
        msg = str(e).lower()
        if e.status_code == 404 or "not found" in msg:
            yield (
                f"Ollama has no model `{CHAT_MODEL}`. Install it with:\n\n"
                f"```\nollama pull {CHAT_MODEL}\n```\n\n"
                "Or use another local model:\n\n"
                "```\nexport RECALL_LLM_CHAT_MODEL=llama3.2:1b\n```"
            )
            return
        raise
