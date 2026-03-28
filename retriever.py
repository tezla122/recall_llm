"""Hybrid retrieval: temporal pre-filter, sequential FTS then ANN, RRF merge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import dateparser
import pandas as pd
from dateparser.search import search_dates

from recall_llm.embedder import embed_batch
from recall_llm.store import get_table

_RRF_K = 60
_FTS_LIMIT = 50
_ANN_LIMIT = 50


def _utc_day_bounds(dt: datetime) -> tuple[datetime, datetime]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def _temporal_predicate(query: str) -> str | None:
    """
    If *query* contains a parsable date/time phrase (via dateparser), return a Lance
    ``where`` clause restricting ``captured_at`` to that UTC calendar day.
    """
    base = datetime.now(timezone.utc)
    hits = search_dates(
        query,
        settings={
            "PREFER_DATES_FROM": "past",
            "RELATIVE_BASE": base,
        },
    )
    if not hits:
        return None
    snippet, dt_guess = hits[0]
    first_dt = dateparser.parse(
        snippet,
        settings={"RELATIVE_BASE": base, "PREFER_DATES_FROM": "past"},
    )
    if first_dt is None:
        first_dt = dt_guess
    start, end = _utc_day_bounds(first_dt)
    return (
        f"captured_at >= timestamp '{start.isoformat()}' "
        f"AND captured_at < timestamp '{end.isoformat()}'"
    )


def _reciprocal_rank_fusion(
    fts_ids: list[str],
    ann_ids: list[str],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(fts_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(ann_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _rows_by_doc_id(*frames):
    merged: dict[str, dict] = {}
    for df in frames:
        if df is None or getattr(df, "empty", True):
            continue
        for _, row in df.iterrows():
            doc_id = str(row["doc_id"])
            if doc_id not in merged:
                merged[doc_id] = row.to_dict()
    return merged


def _recent_memories_fallback(table, k: int, *, max_age_hours: int = 48) -> list[dict]:
    """When FTS/vector miss, return the newest rows only if they look like a recent upload."""
    try:
        df = table.to_pandas()
    except Exception:
        return []
    if df.empty or "captured_at" not in df.columns:
        return []
    df = df.sort_values("captured_at", ascending=False)
    latest_raw = df.iloc[0]["captured_at"]
    if pd.isna(latest_raw):
        return []
    if isinstance(latest_raw, pd.Timestamp):
        latest = latest_raw.to_pydatetime()
    else:
        latest = latest_raw
    if getattr(latest, "tzinfo", None) is None:
        latest = latest.replace(tzinfo=timezone.utc)
    else:
        latest = latest.astimezone(timezone.utc)
    age_s = (datetime.now(timezone.utc) - latest).total_seconds()
    if age_s > max_age_hours * 3600:
        return []
    return df.head(k).to_dict("records")


def hybrid_search(query: str, k: int = 5) -> list[dict]:
    """
    If the query looks temporal, pre-filter ``captured_at``.

    Then run **FTS (BM25) first**, then **ANN vector search** (sequential, not parallel),
    and merge with reciprocal rank fusion. Returns up to *k* rows as dicts.
    """
    table = get_table()
    if table.count_rows() == 0:
        return []

    where_clause = _temporal_predicate(query)

    try:
        table.create_fts_index(["raw_text", "caption"], replace=False)
    except Exception:
        pass

    query_vec = embed_batch([query])[0]

    fts_builder = table.search(query, query_type="fts", fts_columns=["raw_text", "caption"])
    if where_clause is not None:
        fts_builder = fts_builder.where(where_clause, prefilter=True)
    fts_df = fts_builder.limit(_FTS_LIMIT).to_pandas()

    ann_builder = table.search(query_vec, query_type="vector", vector_column_name="vector")
    if where_clause is not None:
        ann_builder = ann_builder.where(where_clause, prefilter=True)
    ann_df = ann_builder.limit(_ANN_LIMIT).to_pandas()

    fts_ids = [str(x) for x in fts_df["doc_id"].tolist()] if not fts_df.empty else []
    ann_ids = [str(x) for x in ann_df["doc_id"].tolist()] if not ann_df.empty else []

    ranked = _reciprocal_rank_fusion(fts_ids, ann_ids)
    top_ids = [str(doc_id) for doc_id, _ in ranked[:k]]
    row_map = _rows_by_doc_id(fts_df, ann_df)

    results: list[dict] = []
    for doc_id in top_ids:
        row = row_map.get(doc_id)
        if row is not None:
            results.append(row)

    if not results and table.count_rows() > 0:
        return _recent_memories_fallback(table, k)
    return results
