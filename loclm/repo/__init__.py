"""LocLM Multi-Repository & Codebase Intelligence Module (V6 Architecture).

Provides multi-workspace indexing, cross-repository symbol resolution,
and dependency graph tracking.
"""

from loclm.repo.manager import MultiRepoManager, RepoMetadata, SymbolDefinition

__all__ = [
    "MultiRepoManager",
    "RepoMetadata",
    "SymbolDefinition",
]
