"""LanceDB-backed vector store on local disk (no /dev/shm, no in-memory tables)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import lancedb
import pyarrow as pa

# Strict on-disk path to keep RAM usage predictable on 16GB systems.
DB_PATH = Path.home() / ".local/share/recall-llm/db"

# PyArrow 23+ uses list_(..., list_size=) instead of top-level fixed_size_list().
_VECTOR_TYPE = pa.list_(pa.float32(), list_size=768)

MEMORIES_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string()),
        pa.field("captured_at", pa.timestamp("us")),
        pa.field("doc_type", pa.string()),
        pa.field("raw_text", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("image_path", pa.string()),
        pa.field("vector", _VECTOR_TYPE),
    ]
)


def _empty_memories_table() -> pa.Table:
    return pa.table(
        {
            "doc_id": pa.array([], type=pa.string()),
            "captured_at": pa.array([], type=pa.timestamp("us")),
            "doc_type": pa.array([], type=pa.string()),
            "raw_text": pa.array([], type=pa.string()),
            "caption": pa.array([], type=pa.string()),
            "image_path": pa.array([], type=pa.string()),
            "vector": pa.array([], type=_VECTOR_TYPE),
        },
        schema=MEMORIES_SCHEMA,
    )


def get_table():
    """Connect to LanceDB under ``~/.local/share/recall-llm/db`` and open/create ``memories``."""
    DB_PATH.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(DB_PATH))
    if "memories" not in db.table_names():
        db.create_table("memories", data=_empty_memories_table())
    return db.open_table("memories")


def upsert_memory(
    *,
    doc_id: str,
    captured_at: datetime,
    doc_type: str,
    raw_text: str,
    caption: str,
    image_path: str,
    vector: list[float],
) -> None:
    """Replace any existing row with the same ``doc_id``, then append one row."""
    table = get_table()
    table.delete(f"doc_id = '{doc_id}'")
    table.add(
        [
            {
                "doc_id": doc_id,
                "captured_at": captured_at,
                "doc_type": doc_type,
                "raw_text": raw_text,
                "caption": caption,
                "image_path": image_path,
                "vector": vector,
            }
        ],
    )
