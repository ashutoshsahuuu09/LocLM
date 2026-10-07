"""Workspace Scanner & Tree Inspector Module — LocLM V9.

Provides file tree generation, directory structure analysis, and project inspection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


class WorkspaceScanner:
    """Scans and inspects local workspace directory trees and project indicators."""

    @staticmethod
    def generate_tree(
        root_path: str | Path,
        max_depth: int = 3,
        max_files: int = 150,
    ) -> str:
        """Generate a formatted ASCII directory tree representation.

        Args:
            root_path: Directory path to scan.
            max_depth: Maximum directory recursion depth.
            max_files: Maximum total files to include.

        Returns:
            ASCII directory tree string.
        """
        root = Path(root_path).resolve()
        if not root.exists() or not root.is_dir():
            return f"[Error: Directory '{root}' does not exist]"

        lines: list[str] = [f"{root.name}/"]
        file_count = 0

        def _walk(directory: Path, prefix: str = "", depth: int = 1):
            nonlocal file_count
            if depth > max_depth or file_count >= max_files:
                return

            try:
                entries = sorted(
                    [e for e in directory.iterdir() if e.name not in IGNORED_DIRS],
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
            except PermissionError:
                lines.append(f"{prefix}└── [Permission Denied]")
                return

            count = len(entries)
            for idx, entry in enumerate(entries):
                if file_count >= max_files:
                    lines.append(f"{prefix}└── ... (max files limit reached)")
                    break

                is_last = (idx == count - 1)
                connector = "└── " if is_last else "├── "
                child_prefix = "    " if is_last else "│   "

                if entry.is_dir():
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    _walk(entry, prefix + child_prefix, depth + 1)
                else:
                    file_count += 1
                    lines.append(f"{prefix}{connector}{entry.name}")

        _walk(root)
        return "\n".join(lines)

    @staticmethod
    def inspect_project_indicators(root_path: str | Path) -> dict[str, Any]:
        """Detect project technologies, language signatures, git status, and framework files.

        Returns:
            Dict containing detected frameworks, language, git presence, and test setup.
        """
        root = Path(root_path).resolve()
        if not root.exists():
            return {"error": f"Path '{root}' does not exist"}

        has_git = (root / ".git").exists()
        has_docker = (root / "Dockerfile").exists() or (root / "docker-compose.yml").exists()

        languages: list[str] = []
        frameworks: list[str] = []
        tests_available = False

        if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists() or (root / "setup.py").exists():
            languages.append("Python")

        if (root / "package.json").exists():
            languages.append("JavaScript/TypeScript")
            try:
                pkg = (root / "package.json").read_text(encoding="utf-8").lower()
                if "react" in pkg:
                    frameworks.append("React")
                if "vite" in pkg:
                    frameworks.append("Vite")
                if "express" in pkg:
                    frameworks.append("Express")
                if "next" in pkg:
                    frameworks.append("Next.js")
            except Exception:
                pass

        if "Python" in languages:
            # Check for FastAPI / Django / Flask
            for p_file in ("pyproject.toml", "requirements.txt"):
                target = root / p_file
                if target.exists():
                    try:
                        text = target.read_text(encoding="utf-8").lower()
                        if "fastapi" in text:
                            frameworks.append("FastAPI")
                        if "django" in text:
                            frameworks.append("Django")
                        if "flask" in text:
                            frameworks.append("Flask")
                    except Exception:
                        pass

        if (root / "tests").exists() or (root / "test").exists():
            tests_available = True

        return {
            "name": root.name,
            "path": str(root),
            "languages": languages or ["General"],
            "frameworks": frameworks or ["None"],
            "has_git": has_git,
            "has_docker": has_docker,
            "tests_available": tests_available,
        }
