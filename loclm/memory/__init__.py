"""LocLM Memory Module (V5 Architecture).

Provides persistent offline memory, vector/keyword indexing,
and contextual retrieval for multi-agent workflows.
"""

from loclm.memory.base import BaseMemoryStore, MemoryEntry, MemorySearchResult
from loclm.memory.local_store import SQLiteMemoryStore
from loclm.memory.manager import MemoryManager

__all__ = [
    "BaseMemoryStore",
    "MemoryEntry",
    "MemorySearchResult",
    "SQLiteMemoryStore",
    "MemoryManager",
]
