"""Filesystem tools for LocLM (V2).

Provides safe local filesystem operations:
- ReadFileTool
- WriteFileTool
- ListDirTool
- SearchFilesTool
- FileInfoTool
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loclm.tools.base import BaseTool, ToolCategory, ToolParameter, ToolPermissionLevel, ToolResult
from loclm.tools.security import SecurityGuard


class ReadFileTool(BaseTool):
    """Tool to read text content from a file."""

    name = "read_file"
    description = "Read text content from a specified file path."
    category = ToolCategory.FILESYSTEM
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Absolute or relative path to the file to read",
                required=True,
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="1-indexed line number to start reading from (optional)",
                required=False,
                default=1,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="1-indexed line number to stop reading at (optional)",
                required=False,
                default=None,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        file_path_str = kwargs.get("path")
        if not file_path_str:
            return ToolResult(success=False, error="Parameter 'path' is required")

        start_line = kwargs.get("start_line", 1) or 1
        end_line = kwargs.get("end_line")

        target = Path(file_path_str).resolve()
        is_valid, reason = self._guard.validate_path(target)
        if not is_valid:
            return ToolResult(success=False, error=reason)

        if not target.exists():
            return ToolResult(success=False, error=f"File not found: {target}")

        if not target.is_file():
            return ToolResult(success=False, error=f"Path is not a file: {target}")

        try:
            with open(target, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            s_idx = max(0, start_line - 1)
            e_idx = end_line if end_line else total_lines

            selected_lines = lines[s_idx:e_idx]
            content = "".join(selected_lines)

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "path": str(target),
                    "total_lines": total_lines,
                    "returned_lines": len(selected_lines),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}")


class WriteFileTool(BaseTool):
    """Tool to create or overwrite a file with given content."""

    name = "write_file"
    description = "Create or overwrite a file with the specified content."
    category = ToolCategory.FILESYSTEM
    permission_level = ToolPermissionLevel.CONFIRM

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Target file path to write to",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Text content to write to the file",
                required=True,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        file_path_str = kwargs.get("path")
        content = kwargs.get("content", "")

        if not file_path_str:
            return ToolResult(success=False, error="Parameter 'path' is required")

        target = Path(file_path_str).resolve()
        is_valid, reason = self._guard.validate_path(target)
        if not is_valid:
            return ToolResult(success=False, error=reason)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)

            bytes_written = len(content.encode("utf-8"))
            return ToolResult(
                success=True,
                output=f"Successfully wrote {bytes_written} bytes to '{target}'",
                metadata={"path": str(target), "bytes_written": bytes_written},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to write file: {e}")


class ListDirTool(BaseTool):
    """Tool to list directory contents."""

    name = "list_dir"
    description = "List all files and subdirectories in a given directory path."
    category = ToolCategory.FILESYSTEM
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Directory path to list (default: current directory)",
                required=False,
                default=".",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        dir_path_str = kwargs.get("path", ".") or "."
        target = Path(dir_path_str).resolve()

        is_valid, reason = self._guard.validate_path(target)
        if not is_valid:
            return ToolResult(success=False, error=reason)

        if not target.exists():
            return ToolResult(success=False, error=f"Directory not found: {target}")

        if not target.is_dir():
            return ToolResult(success=False, error=f"Path is not a directory: {target}")

        try:
            entries = []
            for item in sorted(target.iterdir()):
                kind = "DIR" if item.is_dir() else "FILE"
                size = item.stat().st_size if item.is_file() else 0
                entries.append(f"{kind:<5} | {item.name:<30} | {size} bytes")

            output_str = "\n".join(entries) if entries else "(empty directory)"
            return ToolResult(
                success=True,
                output=output_str,
                metadata={"path": str(target), "count": len(entries)},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to list directory: {e}")


class SearchFilesTool(BaseTool):
    """Tool to search for pattern inside files or search filenames."""

    name = "search_files"
    description = "Search for a text query inside files or find filenames matching a pattern."
    category = ToolCategory.FILESYSTEM
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Text pattern or query to search for",
                required=True,
            ),
            ToolParameter(
                name="directory",
                type="string",
                description="Directory to search in (default: current directory)",
                required=False,
                default=".",
            ),
            ToolParameter(
                name="glob_pattern",
                type="string",
                description="File glob pattern to match (e.g. '*.py', '*.md')",
                required=False,
                default="*",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        dir_str = kwargs.get("directory", ".") or "."
        glob_pat = kwargs.get("glob_pattern", "*") or "*"

        target = Path(dir_str).resolve()
        is_valid, reason = self._guard.validate_path(target)
        if not is_valid:
            return ToolResult(success=False, error=reason)

        matches = []
        try:
            for file_path in target.rglob(glob_pat):
                if file_path.is_file() and not any(part.startswith(".") for part in file_path.parts):
                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            for idx, line in enumerate(f, 1):
                                if query.lower() in line.lower():
                                    rel = file_path.relative_to(target)
                                    matches.append(f"{rel}:{idx}: {line.strip()}")
                                    if len(matches) >= 50:
                                        break
                    except Exception:
                        continue
                if len(matches) >= 50:
                    break

            output = "\n".join(matches) if matches else f"No matches found for '{query}'"
            return ToolResult(
                success=True,
                output=output,
                metadata={"matches_count": len(matches)},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed: {e}")


class FileInfoTool(BaseTool):
    """Tool to inspect file or directory metadata."""

    name = "file_info"
    description = "Get detailed file metadata (size, permissions, type, timestamps)."
    category = ToolCategory.FILESYSTEM
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to inspect",
                required=True,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        file_path_str = kwargs.get("path")
        if not file_path_str:
            return ToolResult(success=False, error="Parameter 'path' is required")

        target = Path(file_path_str).resolve()
        is_valid, reason = self._guard.validate_path(target)
        if not is_valid:
            return ToolResult(success=False, error=reason)

        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {target}")

        stat = target.stat()
        info = (
            f"Path: {target}\n"
            f"Type: {'Directory' if target.is_dir() else 'File'}\n"
            f"Size: {stat.st_size} bytes ({stat.st_size / 1024:.2f} KB)\n"
            f"Modified: {stat.st_mtime}\n"
            f"Created: {stat.st_ctime}"
        )
        return ToolResult(
            success=True,
            output=info,
            metadata={"size_bytes": stat.st_size, "is_dir": target.is_dir()},
        )


class CreateProjectScaffoldTool(BaseTool):
    """Tool to create complete directory structure and files for a project scaffold."""

    name = "create_project_scaffold"
    description = "Create full directory structure and initial source files for a new or designed project."
    category = ToolCategory.FILESYSTEM
    permission_level = ToolPermissionLevel.CONFIRM

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="root_path",
                type="string",
                description="Root directory path for the project scaffold (default: current directory)",
                required=False,
                default=".",
            ),
            ToolParameter(
                name="directories",
                type="array",
                description="List of relative directory paths to create",
                required=False,
                default=[],
            ),
            ToolParameter(
                name="files",
                type="object",
                description="Map of relative file paths to their content strings",
                required=False,
                default={},
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        root_path_str = kwargs.get("root_path", ".") or "."
        directories = kwargs.get("directories", []) or []
        files = kwargs.get("files", {}) or {}

        root = Path(root_path_str).resolve()
        is_valid, reason = self._guard.validate_path(root)
        if not is_valid:
            return ToolResult(success=False, error=reason)

        created_dirs: list[str] = []
        created_files: list[str] = []

        try:
            # Create root directory if it doesn't exist
            root.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(root))

            # Create requested directories
            for d in directories:
                d_path = (root / d).resolve()
                valid, err = self._guard.validate_path(d_path)
                if valid:
                    d_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(d_path))

            # Create requested files with contents
            for rel_file, content in files.items():
                f_path = (root / rel_file).resolve()
                valid, err = self._guard.validate_path(f_path)
                if valid:
                    f_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(f_path, "w", encoding="utf-8") as f_out:
                        f_out.write(content)
                    created_files.append(str(rel_file))

            summary = (
                f"Successfully created project scaffold at: {root}\n"
                f"Directories created ({len(created_dirs)}): {', '.join(directories) if directories else 'root'}\n"
                f"Files written ({len(created_files)}): {', '.join(created_files)}"
            )

            return ToolResult(
                success=True,
                output=summary,
                metadata={"root": str(root), "files_created": created_files, "dirs_created": created_dirs},
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Failed to create project scaffold: {e}")

