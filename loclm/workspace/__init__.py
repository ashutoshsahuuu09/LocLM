"""LocLM Workspace Package (V9 Architecture).

Provides Multi-Directory Workspace Management, Registry Persistence,
Permission Boundaries (READ, WRITE, FULL), Directory Routing, and Context Tracking.
"""

from loclm.workspace.context import WorkspaceContext
from loclm.workspace.manager import WorkspaceManager
from loclm.workspace.permissions import WorkspacePermission
from loclm.workspace.registry import WorkspaceEntry, WorkspaceRegistry
from loclm.workspace.router import DirectoryRouter, WorkspaceMatch
from loclm.workspace.scanner import WorkspaceScanner

__all__ = [
    "WorkspaceContext",
    "WorkspaceManager",
    "WorkspacePermission",
    "WorkspaceEntry",
    "WorkspaceRegistry",
    "DirectoryRouter",
    "WorkspaceMatch",
    "WorkspaceScanner",
]
