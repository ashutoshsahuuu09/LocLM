"""Project Creation Engine & Agent — LocLM V9.

Parses natural language project creation requirements, designs architectural plans,
protects existing files against silent overwrite, creates directory structures,
and validates project generation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loclm.models.base import ChatMessage
from loclm.models.manager import ModelManager
from loclm.projects.templates import TemplateRegistry
from loclm.projects.verifier import ProjectVerifier, VerificationReport

logger = logging.getLogger(__name__)


class ProjectPlan(BaseModel):
    """Architectural plan for creating a new project."""

    project_name: str = Field(description="Name of the project to create")
    target_workspace_path: str = Field(description="Parent workspace directory path")
    template_name: str | None = Field(default=None, description="Matched template name if any")
    architecture_summary: str = Field(description="Summary of tech stack & architecture")
    planned_files: list[str] = Field(default_factory=list, description="List of relative file paths to create")
    dependencies: list[str] = Field(default_factory=list, description="Project dependencies")
    requires_user_confirmation: bool = Field(default=True)

    @property
    def project_dir(self) -> Path:
        """Full target project directory path."""
        return Path(self.target_workspace_path) / self.project_name

    def format_plan_report(self) -> str:
        """Format the project creation plan for user confirmation."""
        files_str = "\n".join(f"  {f}" for f in self.planned_files)
        return (
            f"PROJECT PLAN\n\n"
            f"Name:\n{self.project_name}\n\n"
            f"Location:\n{self.project_dir}\n\n"
            f"Architecture:\n{self.architecture_summary}\n\n"
            f"Files to create:\n{files_str}"
        )


class ProjectCreationAgent:
    """Agent responsible for natural language requirements parsing and project creation."""

    def __init__(self, model_manager: ModelManager | None = None) -> None:
        self.model_manager = model_manager

    def parse_requirements_to_plan(
        self,
        prompt: str,
        workspace_path: str | Path,
    ) -> ProjectPlan:
        """Parse user requirements and create a structured ProjectPlan.

        Args:
            prompt: User request prompt (e.g. "Create FastAPI app TaskManager").
            workspace_path: Target workspace path.

        Returns:
            ProjectPlan blueprint.
        """
        p_lower = prompt.lower()
        ws_path = str(Path(workspace_path).resolve())

        # Extract project name heuristic or default
        project_name = "NewProject"
        for keyword in ("called", "named", "project"):
            if keyword in p_lower:
                parts = prompt.split(keyword, 1)[1].strip().split()
                if parts:
                    clean_name = parts[0].strip("'\":;,. ")
                    if clean_name and clean_name.isidentifier():
                        project_name = clean_name
                        break

        # Check matched template
        matched_template = None
        for t in TemplateRegistry.list_templates():
            if t.name in p_lower or t.display_name.lower() in p_lower:
                matched_template = t
                break

        if matched_template:
            planned_files = list(matched_template.default_files)
            dependencies = list(matched_template.dependencies)
            summary = matched_template.description
            t_name = matched_template.name
        else:
            # Default generic Python/Web structure
            t_name = None
            summary = "Custom multi-file project architecture"
            planned_files = [
                "app/__init__.py",
                "app/main.py",
                "app/config.py",
                "tests/__init__.py",
                "tests/test_main.py",
                "requirements.txt",
                "README.md",
                ".gitignore",
            ]
            dependencies = []

        return ProjectPlan(
            project_name=project_name,
            target_workspace_path=ws_path,
            template_name=t_name,
            architecture_summary=summary,
            planned_files=planned_files,
            dependencies=dependencies,
        )

    def check_existing_collisions(
        self,
        project_plan: ProjectPlan,
    ) -> list[str]:
        """Check if any planned file already exists in target directory.

        Returns:
            List of existing relative file paths.
        """
        colliding: list[str] = []
        target_dir = project_plan.project_dir

        for rel_file in project_plan.planned_files:
            full_p = target_dir / rel_file
            if full_p.exists():
                colliding.append(rel_file)

        return colliding

    def execute_plan(
        self,
        plan: ProjectPlan,
        overwrite_strategy: str = "keep",
    ) -> tuple[bool, list[str], VerificationReport]:
        """Execute project creation plan on filesystem.

        Args:
            plan: ProjectPlan to execute.
            overwrite_strategy: 'keep' (skip existing), 'replace' (overwrite), 'backup' (backup & replace).

        Returns:
            Tuple of (success, list_of_created_files, VerificationReport).
        """
        target_dir = plan.project_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        created_files: list[str] = []

        for rel_path in plan.planned_files:
            file_p = target_dir / rel_path

            if file_p.exists():
                if overwrite_strategy == "keep":
                    logger.info("Skipping existing file '%s' per strategy 'keep'", rel_path)
                    continue
                elif overwrite_strategy == "backup":
                    backup_p = file_p.with_suffix(file_p.suffix + f".bak.{int(time.time())}")
                    file_p.rename(backup_p)
                    logger.info("Backed up existing file to '%s'", backup_p.name)

            file_p.parent.mkdir(parents=True, exist_ok=True)
            starter_code = self._generate_starter_content(rel_path, plan)
            file_p.write_text(starter_code, encoding="utf-8")
            created_files.append(rel_path)

        report = ProjectVerifier.verify_project(target_dir, plan.planned_files)
        return True, created_files, report

    def _generate_starter_content(self, rel_path: str, plan: ProjectPlan) -> str:
        """Generate clean starter code content for a created file."""
        f_name = Path(rel_path).name

        if f_name == "__init__.py":
            return f'"""Package initialization for {plan.project_name}."""\n'
        elif f_name == "main.py":
            if plan.template_name == "fastapi":
                return (
                    'from fastapi import FastAPI\n\n'
                    f'app = FastAPI(title="{plan.project_name}")\n\n'
                    '@app.get("/")\n'
                    'def root():\n'
                    '    return {"message": "Welcome to ' + plan.project_name + ' API"}\n'
                )
            return f'"""Main entry point for {plan.project_name}."""\n\ndef main():\n    print("Hello from {plan.project_name}")\n\nif __name__ == "__main__":\n    main()\n'
        elif f_name == "README.md":
            return f"# {plan.project_name}\n\n{plan.architecture_summary}\n\n## Quick Start\n\n- Run application: `python -m app.main`\n"
        elif f_name == "requirements.txt":
            return "\n".join(plan.dependencies) + "\n" if plan.dependencies else "# Add dependencies here\n"
        elif f_name == "Dockerfile":
            return "FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nCMD [\"python\", \"-m\", \"app.main\"]\n"
        elif f_name == "docker-compose.yml":
            return f"version: '3.8'\nservices:\n  app:\n    build: .\n    ports:\n      - \"8000:8000\"\n"
        elif f_name == "package.json":
            return f'{{\n  "name": "{plan.project_name.lower()}",\n  "version": "1.0.0",\n  "private": true\n}}\n'
        elif f_name == "index.html":
            return f"<!DOCTYPE html>\n<html>\n<head>\n  <title>{plan.project_name}</title>\n</head>\n<body>\n  <div id=\"root\"></div>\n</body>\n</html>\n"

        return f"# {rel_path} created by LocLM\n"
