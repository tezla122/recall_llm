"""Bounded async ingestion queue for Recall-LLM."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

INGEST_QUEUE_MAXSIZE = 20


@dataclass
class IngestTask:
    path: Path
    file_type: str
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 3


ingest_queue: asyncio.Queue[IngestTask] = asyncio.Queue(maxsize=INGEST_QUEUE_MAXSIZE)


async def enqueue(task: IngestTask) -> None:
    """Put *task* on the queue, waiting up to 2s for capacity. Drop on timeout."""
    try:
        await asyncio.wait_for(ingest_queue.put(task), timeout=2.0)
        logger.success(
            "Ingest task enqueued; queue size is {}.",
            ingest_queue.qsize(),
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Ingest queue full (maxsize={}); task dropped: {}",
            INGEST_QUEUE_MAXSIZE,
            task.path,
        )
