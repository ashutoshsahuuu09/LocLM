"""SQLite-backed offline Local Memory Store for LocLM V5.

Provides fast, persistent, zero-dependency storage and search
for project knowledge, chat context, and tool histories.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from loclm.memory.base import BaseMemoryStore, MemoryEntry, MemorySearchResult


class SQLiteMemoryStore(BaseMemoryStore):
    """SQLite implementation of offline local memory store."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._shared_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                entry_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON memory_entries(category);
        """)
        conn.commit()


    async def add(
        self,
        content: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Add a memory entry to SQLite store."""
        entry_id = str(uuid.uuid4())
        metadata_dict = metadata or {}
        now = time.time()

        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            category=category,
            metadata=metadata_dict,
            timestamp=now,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_entries (entry_id, content, category, metadata_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (entry_id, content, category, json.dumps(metadata_dict), now),
            )
            conn.commit()

        return entry

    async def search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
    ) -> list[MemorySearchResult]:
        """Search memory entries using term frequency / similarity scoring."""
        entries = await self._fetch_candidates(category)
        if not entries:
            return []

        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return [MemorySearchResult(entry=e, score=0.5) for e in entries[:limit]]

        scored_results: list[MemorySearchResult] = []
        for entry in entries:
            content_terms = re.findall(r"\w+", entry.content.lower())
            if not content_terms:
                continue

            matches = sum(1 for term in query_terms if term in content_terms)
            if matches == 0:
                continue

            # TF score normalized by content length
            tf_score = matches / (len(set(content_terms)) + 1)
            # Recency boost (up to 0.2 score boost for newer entries)
            recency_boost = min(0.2, 1.0 / (1.0 + (time.time() - entry.timestamp) / 86400))
            final_score = round(min(1.0, tf_score + recency_boost), 4)

            scored_results.append(MemorySearchResult(entry=entry, score=final_score))

        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:limit]

    async def _fetch_candidates(self, category: str | None = None) -> list[MemoryEntry]:
        with self._get_connection() as conn:
            if category:
                cursor = conn.execute(
                    "SELECT entry_id, content, category, metadata_json, timestamp FROM memory_entries WHERE category = ?",
                    (category,),
                )
            else:
                cursor = conn.execute(
                    "SELECT entry_id, content, category, metadata_json, timestamp FROM memory_entries"
                )

            rows = cursor.fetchall()
            return [
                MemoryEntry(
                    entry_id=row["entry_id"],
                    content=row["content"],
                    category=row["category"],
                    metadata=json.loads(row["metadata_json"]),
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]

    async def clear(self, category: str | None = None) -> None:
        """Clear entries from store."""
        with self._get_connection() as conn:
            if category:
                conn.execute("DELETE FROM memory_entries WHERE category = ?", (category,))
            else:
                conn.execute("DELETE FROM memory_entries")
            conn.commit()

    async def get_all(self, limit: int = 100) -> list[MemoryEntry]:
        """Get stored entries."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT entry_id, content, category, metadata_json, timestamp FROM memory_entries ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                MemoryEntry(
                    entry_id=row["entry_id"],
                    content=row["content"],
                    category=row["category"],
                    metadata=json.loads(row["metadata_json"]),
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]
