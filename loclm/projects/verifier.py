"""Project Verifier Module — LocLM V9.

Validates created or modified project file structures, syntax, and build readiness.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VerificationReport(BaseModel):
    """Result of project file structure and syntax verification."""

    project_name: str
    total_files_checked: int = Field(default=0)
    missing_files: list[str] = Field(default_factory=list)
    syntax_errors: list[str] = Field(default_factory=list)
    is_valid: bool = Field(default=True)
    details: list[str] = Field(default_factory=list)


class ProjectVerifier:
    """Verifies project files and checks for missing entries or syntax errors."""

    @staticmethod
    def verify_project(
        project_dir: str | Path,
        expected_files: list[str] | None = None,
    ) -> VerificationReport:
        """Verify project files and check syntax.

        Args:
            project_dir: Root directory of the project.
            expected_files: List of relative file paths expected to exist.

        Returns:
            VerificationReport.
        """
        root = Path(project_dir).resolve()
        missing: list[str] = []
        syntax_errs: list[str] = []
        details: list[str] = []
        checked = 0

        # 1. Check expected files exist
        if expected_files:
            for rel_path in expected_files:
                target = root / rel_path
                checked += 1
                if not target.exists():
                    missing.append(rel_path)
                    details.append(f"✗ Missing: {rel_path}")
                else:
                    details.append(f"✓ {rel_path}")

        # 2. Syntax validation for Python and JSON files
        if root.exists():
            for file_path in root.rglob("*"):
                if file_path.is_file() and not any(part.startswith(".") for part in file_path.parts):
                    if file_path.suffix == ".py":
                        try:
                            ast.parse(file_path.read_text(encoding="utf-8"))
                        except Exception as exc:
                            rel = str(file_path.relative_to(root))
                            syntax_errs.append(f"{rel}: {exc}")
                    elif file_path.suffix == ".json":
                        try:
                            json.loads(file_path.read_text(encoding="utf-8"))
                        except Exception as exc:
                            rel = str(file_path.relative_to(root))
                            syntax_errs.append(f"{rel}: {exc}")

        is_valid = len(missing) == 0 and len(syntax_errs) == 0

        return VerificationReport(
            project_name=root.name,
            total_files_checked=checked,
            missing_files=missing,
            syntax_errors=syntax_errs,
            is_valid=is_valid,
            details=details,
        )
