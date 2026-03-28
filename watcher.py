"""Filesystem watcher with inotify (no polling) and path+mtime deduplication."""

from __future__ import annotations

import asyncio
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers.inotify import InotifyObserver

from recall_llm.queue_manager import IngestTask, enqueue

_WATCH_SUFFIXES = frozenset({".pdf", ".png", ".jpg", ".jpeg"})


def _file_type_label(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return ext or "unknown"


def _ingest_dedup_key(path: Path) -> tuple[str, int, int] | None:
    """Identity for deduping watcher noise; changes when the file is rewritten or touched."""
    try:
        p = path.resolve()
        st = p.stat()
    except OSError:
        return None
    return (str(p), st.st_mtime_ns, st.st_size)


def _eligible_path(path: Path) -> Path | None:
    path = Path(path).resolve()
    if not path.is_file():
        return None
    if path.suffix.lower() not in _WATCH_SUFFIXES:
        return None
    return path


async def schedule_file_if_eligible_async(
    path: Path,
    seen: set[tuple[str, int, int]],
) -> None:
    """Awaiting enqueue from the asyncio loop (e.g. startup scan) — avoids deadlock."""
    p = _eligible_path(path)
    if p is None:
        return
    key = _ingest_dedup_key(p)
    if key is None or key in seen:
        return
    seen.add(key)
    await enqueue(IngestTask(path=p, file_type=_file_type_label(p)))


async def scan_inbox_on_startup(
    watch_path: Path,
    seen: set[tuple[str, int, int]],
) -> None:
    """Enqueue files already in the inbox (inotify does not emit created for past files)."""
    watch_path = Path(watch_path).resolve()
    if not watch_path.is_dir():
        return
    for child in sorted(watch_path.iterdir()):
        await schedule_file_if_eligible_async(child, seen)


def schedule_file_if_eligible_threadsafe(
    path: Path,
    loop: asyncio.AbstractEventLoop,
    seen: set[tuple[str, int, int]],
) -> None:
    """Called from the watchdog thread; must use run_coroutine_threadsafe."""
    p = _eligible_path(path)
    if p is None:
        return
    key = _ingest_dedup_key(p)
    if key is None or key in seen:
        return
    seen.add(key)
    task = IngestTask(path=p, file_type=_file_type_label(p))
    asyncio.run_coroutine_threadsafe(enqueue(task), loop)


class _IngestEventHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, seen: set[tuple[str, int, int]]) -> None:
        self._loop = loop
        self._seen = seen

    def _dispatch_ingest(self, src_path: str) -> None:
        schedule_file_if_eligible_threadsafe(Path(src_path), self._loop, self._seen)

    def on_created(self, event):  # noqa: ANN001 — watchdog API
        if getattr(event, "is_directory", False):
            return
        self._dispatch_ingest(event.src_path)

    def on_modified(self, event):  # noqa: ANN001
        if getattr(event, "is_directory", False):
            return
        self._dispatch_ingest(event.src_path)


def start_inotify_watcher(
    watch_path: Path,
    loop: asyncio.AbstractEventLoop,
    seen: set[tuple[str, int, int]],
    *,
    recursive: bool = False,
) -> InotifyObserver:
    """Start an inotify-based observer. Call :func:`scan_inbox_on_startup` before this."""
    watch_path = Path(watch_path).resolve()
    handler = _IngestEventHandler(loop, seen)
    observer = InotifyObserver()
    observer.schedule(handler, str(watch_path), recursive=recursive)
    observer.start()
    return observer
