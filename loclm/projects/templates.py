"""Project Templates Module — LocLM V9.

Provides template definitions and registry for scaffolding projects (FastAPI, React, Vite,
Python CLI, Machine Learning, Data Science, Node.js, Express, Full Stack, Docker).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectTemplate(BaseModel):
    """Definition of a project template blueprint."""

    name: str = Field(description="Template identifier, e.g. fastapi")
    display_name: str = Field(description="Human-readable title")
    category: str = Field(default="general")
    description: str = Field(description="Summary of stack & setup")
    default_files: list[str] = Field(default_factory=list, description="Default file paths")
    dependencies: list[str] = Field(default_factory=list, description="Default dependencies")


class TemplateRegistry:
    """Registry of built-in project templates."""

    TEMPLATES: dict[str, ProjectTemplate] = {
        "fastapi": ProjectTemplate(
            name="fastapi",
            display_name="FastAPI Backend",
            category="python",
            description="FastAPI web application with SQLite/PostgreSQL, JWT, Docker, and Pytest",
            default_files=[
                "app/__init__.py",
                "app/main.py",
                "app/config.py",
                "app/database.py",
                "app/models.py",
                "app/auth.py",
                "app/routes.py",
                "tests/__init__.py",
                "tests/test_main.py",
                "Dockerfile",
                "docker-compose.yml",
                "requirements.txt",
                "README.md",
                ".gitignore",
            ],
            dependencies=["fastapi", "uvicorn", "pydantic", "sqlalchemy", "pytest", "httpx"],
        ),
        "react": ProjectTemplate(
            name="react",
            display_name="React Frontend",
            category="frontend",
            description="React frontend project with Vite, components, and pages scaffolding",
            default_files=[
                "src/App.jsx",
                "src/main.jsx",
                "src/components/Navbar.jsx",
                "src/components/Sidebar.jsx",
                "src/pages/Dashboard.jsx",
                "src/pages/Login.jsx",
                "package.json",
                "vite.config.js",
                "index.html",
                "README.md",
            ],
            dependencies=["react", "react-dom", "vite"],
        ),
        "vite": ProjectTemplate(
            name="vite",
            display_name="Vite Web App",
            category="frontend",
            description="Minimal Vite web application bundle",
            default_files=["src/main.js", "index.html", "package.json", "vite.config.js", "README.md"],
            dependencies=["vite"],
        ),
        "python_cli": ProjectTemplate(
            name="python_cli",
            display_name="Python CLI Application",
            category="python",
            description="Python command line application with Typer/Rich and pytest",
            default_files=[
                "src/__init__.py",
                "src/main.py",
                "src/cli.py",
                "tests/__init__.py",
                "tests/test_cli.py",
                "pyproject.toml",
                "README.md",
            ],
            dependencies=["typer", "rich", "pytest"],
        ),
        "machine_learning": ProjectTemplate(
            name="machine_learning",
            display_name="Machine Learning Project",
            category="ml",
            description="ML workspace with data, notebook, train, predict scripts",
            default_files=[
                "data/.gitkeep",
                "models/.gitkeep",
                "notebooks/exploration.ipynb",
                "src/__init__.py",
                "src/train.py",
                "src/predict.py",
                "requirements.txt",
                "README.md",
            ],
            dependencies=["numpy", "pandas", "scikit-learn", "torch", "matplotlib"],
        ),
        "data_science": ProjectTemplate(
            name="data_science",
            display_name="Data Science Project",
            category="ds",
            description="Data analysis workspace with pandas, Jupyter, and visualization",
            default_files=[
                "data/raw/.gitkeep",
                "data/processed/.gitkeep",
                "notebooks/eda.ipynb",
                "src/preprocess.py",
                "requirements.txt",
                "README.md",
            ],
            dependencies=["pandas", "numpy", "matplotlib", "seaborn", "jupyter"],
        ),
        "nodejs": ProjectTemplate(
            name="nodejs",
            display_name="Node.js Application",
            category="node",
            description="Standard Node.js application scaffold",
            default_files=["src/index.js", "package.json", "README.md", ".gitignore"],
            dependencies=[],
        ),
        "express": ProjectTemplate(
            name="express",
            display_name="Express API Server",
            category="node",
            description="Node.js Express REST API server with routes and middleware",
            default_files=[
                "src/app.js",
                "src/routes/index.js",
                "package.json",
                "README.md",
                ".gitignore",
            ],
            dependencies=["express", "dotenv", "cors"],
        ),
        "fullstack": ProjectTemplate(
            name="fullstack",
            display_name="Full Stack Application",
            category="fullstack",
            description="Full-stack repository with backend and frontend services",
            default_files=[
                "backend/app/main.py",
                "backend/requirements.txt",
                "frontend/src/App.jsx",
                "frontend/package.json",
                "docker-compose.yml",
                "README.md",
            ],
            dependencies=[],
        ),
        "docker": ProjectTemplate(
            name="docker",
            display_name="Docker Containerized Project",
            category="devops",
            description="Dockerized service environment with Dockerfile and compose configuration",
            default_files=["Dockerfile", "docker-compose.yml", ".dockerignore", "README.md"],
            dependencies=[],
        ),
    }

    @classmethod
    def get(cls, name: str) -> ProjectTemplate | None:
        """Get template by name (case-insensitive)."""
        key = name.lower().replace("-", "_").replace(" ", "_")
        return cls.TEMPLATES.get(key)

    @classmethod
    def list_templates(cls) -> list[ProjectTemplate]:
        """Return list of all built-in project templates."""
        return list(cls.TEMPLATES.values())
