"""Workspace Registry Module — LocLM V9.

Manages persistent local workspace metadata storage (~/.loclm/workspaces.json).
100% offline, zero network data egress.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loclm.workspace.permissions import WorkspacePermission

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path.home() / ".loclm" / "workspaces.json"


class WorkspaceEntry(BaseModel):
    """Metadata record for an approved local workspace directory."""

    name: str = Field(description="Friendly workspace name, e.g. CodeV")
    path: str = Field(description="Absolute target directory path")
    permission: WorkspacePermission = Field(default=WorkspacePermission.WRITE)
    approved: bool = Field(default=True, description="Explicit user approval flag")
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved_path(self) -> Path:
        """Resolved absolute Path object."""
        return Path(self.path).resolve()


class WorkspaceRegistry:
    """Persistent registry tracking approved user workspaces."""

    def __init__(self, registry_file: Path | None = None) -> None:
        self.registry_file = registry_file or DEFAULT_REGISTRY_PATH
        self._entries: dict[str, WorkspaceEntry] = {}
        self._load()

    def _derive_default_name(self, target_path: Path) -> str:
        """Derive a friendly non-empty workspace name from path."""
        if target_path.name:
            return target_path.name
        if target_path.drive:
            return f"{target_path.drive.rstrip(':').upper()}_Drive"
        return "Root_Workspace"

    def _load(self) -> None:
        """Load entries from workspaces.json."""
        if not self.registry_file.exists():
            self._entries = {}
            return

        try:
            content = self.registry_file.read_text(encoding="utf-8")
            data = json.loads(content)
            self._entries = {}
            for key, item in data.items():
                target_p = Path(item.get("path", "")).resolve()
                name = item.get("name", "").strip() or key.strip() or self._derive_default_name(target_p)
                item["name"] = name
                entry = WorkspaceEntry(**item)
                self._entries[name] = entry
        except Exception as exc:
            logger.warning("Failed to load workspace registry from %s: %s", self.registry_file, exc)
            self._entries = {}

    def _save(self) -> None:
        """Save entries to workspaces.json."""
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            data = {name: entry.model_dump() for name, entry in self._entries.items()}
            self.registry_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save workspace registry to %s: %s", self.registry_file, exc)

    def add_workspace(
        self,
        name: str | None,
        path: str | Path,
        permission: WorkspacePermission | str = WorkspacePermission.WRITE,
    ) -> WorkspaceEntry:
        """Add an approved workspace entry.

        Args:
            name: Optional workspace name. Derived from path if empty.
            path: Target directory path.
            permission: Granted permission level (READ, WRITE, FULL).

        Returns:
            The created WorkspaceEntry.
        """
        target_path = Path(path).resolve()
        perm = WorkspacePermission(permission) if isinstance(permission, str) else permission

        ws_name = name.strip() if name and name.strip() else self._derive_default_name(target_path)

        entry = WorkspaceEntry(
            name=ws_name,
            path=str(target_path),
            permission=perm,
            approved=True,
        )
        self._entries[ws_name] = entry
        self._save()
        logger.info("Added workspace '%s' at '%s' [permission=%s]", ws_name, target_path, perm.value)
        return entry

    def remove_workspace(self, name_or_path: str) -> bool:
        """Remove workspace access permission. (Does NOT delete user files!).

        Returns:
            True if removed, False if not found.
        """
        entry = self.get_workspace(name_or_path)
        if not entry:
            return False

        del self._entries[entry.name]
        self._save()
        logger.info("Removed workspace access permission for '%s' (files remain untouched)", entry.name)
        return True

    def list_workspaces(self) -> list[WorkspaceEntry]:
        """List all approved workspace entries."""
        return list(self._entries.values())

    def get_workspace(self, name_or_path: str | None = None) -> WorkspaceEntry | None:
        """Find a workspace entry by name or exact/resolved path."""
        if not name_or_path or not name_or_path.strip():
            return list(self._entries.values())[0] if self._entries else None

        query = name_or_path.strip()

        # 1. Match by name
        if query in self._entries:
            return self._entries[query]

        # 2. Case-insensitive name match
        lower_query = query.lower()
        for name, entry in self._entries.items():
            if name.lower() == lower_query:
                return entry

        # 3. Match by path
        try:
            target_path = Path(query).resolve()
            for entry in self._entries.values():
                if entry.resolved_path == target_path:
                    return entry
        except Exception:
            pass

        return None

    def update_permission(
        self,
        name_or_path: str,
        new_permission: WorkspacePermission | str,
    ) -> WorkspaceEntry:
        """Update permission level for a workspace.

        Raises:
            KeyError: If workspace is not found.
        """
        entry = self.get_workspace(name_or_path)
        if not entry:
            raise KeyError(f"Workspace '{name_or_path}' not found in registry.")

        perm = WorkspacePermission(new_permission) if isinstance(new_permission, str) else new_permission
        entry.permission = perm
        self._entries[entry.name] = entry
        self._save()
        logger.info("Updated workspace '%s' permission → %s", entry.name, perm.value)
        return entry

    def find_containing_workspace(self, target_path: str | Path) -> WorkspaceEntry | None:
        """Find the approved workspace that contains target_path."""
        resolved = Path(target_path).resolve()
        matched_entry: WorkspaceEntry | None = None
        best_len = -1

        for entry in self._entries.values():
            w_path = entry.resolved_path
            if resolved == w_path or w_path in resolved.parents:
                # Pick the most specific (longest) matching workspace path
                path_len = len(str(w_path))
                if path_len > best_len:
                    best_len = path_len
                    matched_entry = entry

        return matched_entry
