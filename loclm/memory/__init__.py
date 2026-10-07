"""LocLM Memory Module (V7 Architecture).

Provides persistent offline memory, vector/keyword indexing,
contextual retrieval for multi-agent workflows, and V7 session-persistent
conversation thread memory.
"""

from loclm.memory.base import BaseMemoryStore, MemoryEntry, MemorySearchResult
from loclm.memory.local_store import SQLiteMemoryStore
from loclm.memory.manager import MemoryManager
from loclm.memory.session import SessionMemory, SessionTurn

__all__ = [
    "BaseMemoryStore",
    "MemoryEntry",
    "MemorySearchResult",
    "SQLiteMemoryStore",
    "MemoryManager",
    # V7
    "SessionMemory",
    "SessionTurn",
]
