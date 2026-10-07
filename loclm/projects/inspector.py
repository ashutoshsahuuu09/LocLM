"""Project Inspector Module — LocLM V9.

Analyzes an existing project directory inside an approved workspace and generates
a structured project profile (Type, Backend, Frontend, Database, Git, Tests, Path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loclm.workspace.scanner import WorkspaceScanner


class ProjectInspector:
    """Inspects and profiles existing projects within approved workspaces."""

    @staticmethod
    def inspect(project_path: str | Path) -> dict[str, Any]:
        """Perform deep inspection of a project directory.

        Returns:
            Dict profile summary with keys: name, type, backend, frontend, database, git, tests, path.
        """
        root = Path(project_path).resolve()
        indicators = WorkspaceScanner.inspect_project_indicators(root)

        backend = "None"
        frontend = "None"
        database = "None"
        project_type = "General"

        frameworks = indicators.get("frameworks", [])
        languages = indicators.get("languages", [])

        # Backend detection
        for fw in ("FastAPI", "Django", "Flask", "Express"):
            if fw in frameworks:
                backend = fw

        # Frontend detection
        for fw in ("React", "Vite", "Next.js"):
            if fw in frameworks:
                frontend = fw

        # Database detection
        for f_name in ("pyproject.toml", "requirements.txt", "package.json"):
            target = root / f_name
            if target.exists():
                try:
                    text = target.read_text(encoding="utf-8").lower()
                    if "postgresql" in text or "psycopg" in text or "asyncpg" in text:
                        database = "PostgreSQL"
                    elif "sqlite" in text:
                        database = "SQLite"
                    elif "mongodb" in text or "mongoose" in text or "pymongo" in text:
                        database = "MongoDB"
                    elif "redis" in text:
                        database = "Redis"
                except Exception:
                    pass

        # Project type synthesis
        if backend != "None" and frontend != "None":
            project_type = "Full Stack"
        elif backend != "None":
            project_type = f"Backend ({backend})"
        elif frontend != "None":
            project_type = f"Frontend ({frontend})"
        elif "Python" in languages:
            project_type = "Python Application"
        elif "JavaScript/TypeScript" in languages:
            project_type = "Node.js / Web Application"

        return {
            "name": root.name,
            "type": project_type,
            "languages": languages or ["General"],
            "frameworks": frameworks or ["None"],
            "backend": backend,
            "frontend": frontend,
            "database": database,
            "git": "Yes" if indicators.get("has_git") else "No",
            "tests": "Available" if indicators.get("tests_available") else "None",
            "docker": "Available" if indicators.get("has_docker") else "None",
            "workspace_path": str(root),
        }

    @staticmethod
    def format_inspection_report(profile: dict[str, Any]) -> str:
        """Format inspection profile into clean Markdown/text block."""
        return (
            f"PROJECT\n\n"
            f"Name:\n{profile.get('name')}\n\n"
            f"Type:\n{profile.get('type')}\n\n"
            f"Backend:\n{profile.get('backend')}\n\n"
            f"Frontend:\n{profile.get('frontend')}\n\n"
            f"Database:\n{profile.get('database')}\n\n"
            f"Git:\n{profile.get('git')}\n\n"
            f"Tests:\n{profile.get('tests')}\n\n"
            f"Workspace:\n{profile.get('workspace_path')}"
        )
