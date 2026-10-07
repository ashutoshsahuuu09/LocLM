"""LocLM Projects Package (V9 Architecture).

Provides Project Creation Engine (ProjectCreationAgent), Built-in Project Templates,
Project Inspection, Verification, and Existing File Collision Handling.
"""

from loclm.projects.creator import ProjectCreationAgent, ProjectPlan
from loclm.projects.inspector import ProjectInspector
from loclm.projects.manager import ProjectManager
from loclm.projects.templates import ProjectTemplate, TemplateRegistry
from loclm.projects.verifier import ProjectVerifier

__all__ = [
    "ProjectCreationAgent",
    "ProjectPlan",
    "ProjectInspector",
    "ProjectManager",
    "ProjectTemplate",
    "TemplateRegistry",
    "ProjectVerifier",
]
