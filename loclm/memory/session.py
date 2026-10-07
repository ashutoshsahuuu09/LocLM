"""Session Memory for LocLM V7.

Provides persistent, cross-turn conversation thread storage backed by SQLite.
Unlike V5 task-scoped memory (which stores only task results), V7 SessionMemory
maintains a full ordered conversation thread: every user message, agent response,
and tool event is appended — enabling true multi-turn context recall.

Key features
------------
- **Named Sessions**: each chat session has a unique session_id.
- **Full Thread Recall**: retrieve the N most recent turns as ChatMessage objects.
- **Keyword Search**: find past turns containing specific terms across all sessions.
- **Compact Summarisation Hint**: returns a compact summary prefix for prompt injection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from loclm.models.base import ChatMessage

logger = logging.getLogger(__name__)


class SessionTurn:
    """A single recorded turn in a conversation session."""

    __slots__ = ("turn_id", "session_id", "role", "content", "metadata", "timestamp")

    def __init__(
        self,
        turn_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any],
        timestamp: float,
    ) -> None:
        self.turn_id = turn_id
        self.session_id = session_id
        self.role = role
        self.content = content
        self.metadata = metadata
        self.timestamp = timestamp

    def to_chat_message(self) -> ChatMessage:
        """Convert this turn to a ChatMessage for prompt injection."""
        return ChatMessage(role=self.role, content=self.content)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"<SessionTurn {self.role} @ {self.session_id[:8]}: {preview!r}>"


class SessionMemory:
    """SQLite-backed persistent conversation session memory for V7.

    Stores an ordered sequence of conversation turns per session and provides
    retrieval methods for context injection into agent prompts.

    Args:
        db_path: Path to SQLite database file (or ``:memory:`` for tests).
        session_id: Session identifier; auto-generated UUID if not supplied.
        max_context_turns: Maximum number of turns to inject as context.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        session_id: str | None = None,
        max_context_turns: int = 10,
    ) -> None:
        self.db_path = str(db_path)
        self.session_id = session_id or str(uuid.uuid4())
        self.max_context_turns = max_context_turns

        self._shared_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.row_factory = sqlite3.Row

        self._init_db()
        logger.debug("SessionMemory initialised: session_id=%s", self.session_id[:8])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_user_message(self, content: str, metadata: dict[str, Any] | None = None) -> SessionTurn:
        """Append a user message to the current session."""
        return self._append_turn(role="user", content=content, metadata=metadata or {})

    def record_assistant_message(self, content: str, metadata: dict[str, Any] | None = None) -> SessionTurn:
        """Append an assistant (agent) response to the current session."""
        return self._append_turn(role="assistant", content=content, metadata=metadata or {})

    def record_tool_event(
        self,
        tool_name: str,
        result_summary: str,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> SessionTurn:
        """Append a tool execution event as a system-role turn."""
        content = f"[Tool: {tool_name}] {'OK' if success else 'FAILED'} — {result_summary}"
        meta = {"tool_name": tool_name, "success": success, **(metadata or {})}
        return self._append_turn(role="tool", content=content, metadata=meta)

    def get_recent_turns(self, n: int | None = None) -> list[SessionTurn]:
        """Retrieve the N most recent turns for the current session.

        Args:
            n: Number of turns to retrieve. Defaults to ``max_context_turns``.

        Returns:
            List of SessionTurn in chronological order (oldest first).
        """
        limit = n or self.max_context_turns
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT turn_id, session_id, role, content, metadata_json, timestamp
                FROM session_turns
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (self.session_id, limit),
            )
            rows = cursor.fetchall()

        turns = [self._row_to_turn(r) for r in rows]
        turns.reverse()  # chronological order
        return turns

    def get_context_messages(self, n: int | None = None) -> list[ChatMessage]:
        """Return recent turns as ChatMessage list for LLM prompt injection.

        Tool-role turns are skipped (they are internal events only).

        Args:
            n: Number of recent turns to include.

        Returns:
            List of ChatMessage (user/assistant roles only).
        """
        turns = self.get_recent_turns(n)
        return [t.to_chat_message() for t in turns if t.role in ("user", "assistant")]

    def search_turns(self, keyword: str, limit: int = 5) -> list[SessionTurn]:
        """Keyword-based search across all sessions.

        Args:
            keyword: Term to search for in turn content.
            limit: Maximum results to return.

        Returns:
            Matching SessionTurn objects ordered by recency.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT turn_id, session_id, role, content, metadata_json, timestamp
                FROM session_turns
                WHERE content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (f"%{keyword}%", limit),
            )
            rows = cursor.fetchall()
        return [self._row_to_turn(r) for r in rows]

    def get_session_summary_prefix(self) -> str:
        """Return a compact context prefix summarising the current session.

        Useful for injecting a brief conversation history note at the start
        of a new agent prompt without flooding it with full message content.

        Returns:
            A string like ``"[Session Context — 12 turn(s) stored]"`` or empty.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM session_turns WHERE session_id = ?",
                (self.session_id,),
            )
            count: int = cursor.fetchone()[0]

        if count == 0:
            return ""
        return f"[Session Context — {count} turn(s) stored | session: {self.session_id[:8]}]"

    def list_sessions(self) -> list[str]:
        """Return all distinct session IDs present in the database."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT session_id
                FROM session_turns
                GROUP BY session_id
                ORDER BY MIN(timestamp)
                """
            )
            return [row[0] for row in cursor.fetchall()]

    def clear_session(self, session_id: str | None = None) -> None:
        """Delete all turns for the specified (or current) session."""
        target = session_id or self.session_id
        with self._get_conn() as conn:
            conn.execute("DELETE FROM session_turns WHERE session_id = ?", (target,))
            conn.commit()
        logger.info("Cleared session: %s", target[:8])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any],
    ) -> SessionTurn:
        turn_id = str(uuid.uuid4())
        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO session_turns (turn_id, session_id, role, content, metadata_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (turn_id, self.session_id, role, content, json.dumps(metadata), now),
            )
            conn.commit()
        return SessionTurn(
            turn_id=turn_id,
            session_id=self.session_id,
            role=role,
            content=content,
            metadata=metadata,
            timestamp=now,
        )

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_turns (
                    turn_id      TEXT PRIMARY KEY,
                    session_id   TEXT NOT NULL,
                    role         TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    timestamp    REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON session_turns(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts ON session_turns(timestamp)"
            )
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> SessionTurn:
        return SessionTurn(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"]),
            timestamp=row["timestamp"],
        )
