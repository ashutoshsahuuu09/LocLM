"""Multi-Repository Manager & Symbol Graph for LocLM V6.

Indexes workspace repositories, extracts AST symbols (functions, classes, variables),
and resolves dependencies across multi-folder projects.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SymbolDefinition(BaseModel):
    """Represents a code symbol (class, function, variable)."""

    name: str = Field(description="Symbol name")
    kind: str = Field(description="Kind: function, class, variable, import")
    filepath: str = Field(description="Relative filepath containing the symbol")
    line_number: int = Field(description="1-indexed line number of definition")
    docstring: str = Field(default="", description="Associated docstring if present")


class RepoMetadata(BaseModel):
    """Metadata describing a registered repository."""

    repo_id: str = Field(description="Unique repository identifier / folder name")
    root_path: str = Field(description="Absolute path to repository root")
    file_count: int = Field(default=0, description="Total source files indexed")
    symbols: list[SymbolDefinition] = Field(default_factory=list, description="Extracted symbols")


class MultiRepoManager:
    """Manages multi-repository workspaces and cross-repo symbol indexing."""

    def __init__(self) -> None:
        self._repos: dict[str, RepoMetadata] = {}

    def register_repo(self, root_path: str | Path, repo_id: str | None = None) -> RepoMetadata:
        """Register and index a repository root path.

        Args:
            root_path: Path to the repository root directory.
            repo_id: Optional custom ID for the repository.

        Returns:
            RepoMetadata object for the indexed repository.
        """
        root = Path(root_path).resolve()
        r_id = repo_id or root.name

        symbols: list[SymbolDefinition] = []
        file_count = 0

        if root.exists() and root.is_dir():
            for p in root.rglob("*.py"):
                # Exclude virtual environments and hidden dirs
                if any(part.startswith(".") or part in ("venv", "env", "__pycache__") for part in p.parts):
                    continue

                file_count += 1
                symbols.extend(self._extract_python_symbols(p, root))

        metadata = RepoMetadata(
            repo_id=r_id,
            root_path=str(root),
            file_count=file_count,
            symbols=symbols,
        )
        self._repos[r_id] = metadata
        logger.info("Indexed repository '%s': %d files, %d symbols", r_id, file_count, len(symbols))
        return metadata

    def find_symbol(self, symbol_name: str, repo_id: str | None = None) -> list[SymbolDefinition]:
        """Search for symbol definitions across indexed repositories.

        Args:
            symbol_name: Name of symbol to look up.
            repo_id: Optional filter by repository ID.

        Returns:
            List of matching SymbolDefinition objects.
        """
        matches: list[SymbolDefinition] = []
        target_repos = [self._repos[repo_id]] if repo_id and repo_id in self._repos else self._repos.values()

        for repo in target_repos:
            for sym in repo.symbols:
                if sym.name == symbol_name:
                    matches.append(sym)

        return matches

    def _extract_python_symbols(self, filepath: Path, root: Path) -> list[SymbolDefinition]:
        """Extract Python class and function definitions using AST."""
        symbols: list[SymbolDefinition] = []
        rel_path = str(filepath.relative_to(root))

        try:
            code = filepath.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(code, filename=str(filepath))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    doc = ast.get_docstring(node) or ""
                    symbols.append(
                        SymbolDefinition(
                            name=node.name,
                            kind="function",
                            filepath=rel_path,
                            line_number=node.lineno,
                            docstring=doc[:150],
                        )
                    )
                elif isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node) or ""
                    symbols.append(
                        SymbolDefinition(
                            name=node.name,
                            kind="class",
                            filepath=rel_path,
                            line_number=node.lineno,
                            docstring=doc[:150],
                        )
                    )
        except Exception as e:
            logger.debug("Failed to parse AST for %s: %s", filepath, e)

        return symbols
