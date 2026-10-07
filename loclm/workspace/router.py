"""Directory Router Module — LocLM V9.

Determines which approved workspace is relevant to a user request,
identifies ambiguities when multiple projects match, and avoids guessing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from loclm.workspace.registry import WorkspaceEntry, WorkspaceRegistry

logger = logging.getLogger(__name__)


class WorkspaceMatch(BaseModel):
    """Result of directory routing analysis on a user prompt."""

    matched_workspace: WorkspaceEntry | None = Field(default=None)
    all_matches: list[WorkspaceEntry] = Field(default_factory=list)
    is_ambiguous: bool = Field(default=False)
    match_reason: str = Field(default="")


class DirectoryRouter:
    """Routes natural language user requests to approved local workspaces."""

    def route_prompt(
        self,
        prompt: str,
        registry: WorkspaceRegistry,
    ) -> WorkspaceMatch:
        """Analyze prompt and find matching approved workspace(s).

        Args:
            prompt: User request text.
            registry: Active WorkspaceRegistry containing approved workspaces.

        Returns:
            WorkspaceMatch object indicating matched workspace or ambiguous candidates.
        """
        approved = registry.list_workspaces()
        if not approved:
            return WorkspaceMatch(
                matched_workspace=None,
                match_reason="No approved workspaces registered.",
            )

        prompt_lower = prompt.lower()

        # ── Strategy 1: Direct name match in prompt ─────────────────────────
        exact_matches: list[WorkspaceEntry] = []
        for ws in approved:
            pattern = r"\b" + re.escape(ws.name.lower()) + r"\b"
            if re.search(pattern, prompt_lower):
                exact_matches.append(ws)

        if len(exact_matches) == 1:
            logger.info("DirectoryRouter matched workspace by name: '%s'", exact_matches[0].name)
            return WorkspaceMatch(
                matched_workspace=exact_matches[0],
                all_matches=exact_matches,
                is_ambiguous=False,
                match_reason=f"Matched workspace name '{exact_matches[0].name}'",
            )
        elif len(exact_matches) > 1:
            logger.info("DirectoryRouter found ambiguous name matches: %s", [w.name for w in exact_matches])
            return WorkspaceMatch(
                matched_workspace=None,
                all_matches=exact_matches,
                is_ambiguous=True,
                match_reason=f"Multiple workspace names matched: {', '.join(w.name for w in exact_matches)}",
            )

        # ── Strategy 2: Keyword / Path component match ─────────────────────
        keyword_matches: list[WorkspaceEntry] = []
        for ws in approved:
            # Check directory folder name (e.g. CodeV or TradingBot)
            folder_name = ws.resolved_path.name.lower()
            pattern = r"\b" + re.escape(folder_name) + r"\b"
            if re.search(pattern, prompt_lower):
                keyword_matches.append(ws)

        if len(keyword_matches) == 1:
            return WorkspaceMatch(
                matched_workspace=keyword_matches[0],
                all_matches=keyword_matches,
                is_ambiguous=False,
                match_reason=f"Matched workspace folder '{keyword_matches[0].name}'",
            )
        elif len(keyword_matches) > 1:
            return WorkspaceMatch(
                matched_workspace=None,
                all_matches=keyword_matches,
                is_ambiguous=True,
                match_reason=f"Multiple workspace folders matched: {', '.join(w.name for w in keyword_matches)}",
            )

        # ── Strategy 3: General project category heuristic ─────────────────
        # Check if generic project terms ("python project", "app", "workspace") match multiple
        generic_terms = ["project", "codebase", "repo", "app", "workspace"]
        if any(term in prompt_lower for term in generic_terms) and len(approved) > 1:
            logger.info("DirectoryRouter: ambiguous request matching %d approved workspaces", len(approved))
            return WorkspaceMatch(
                matched_workspace=None,
                all_matches=approved,
                is_ambiguous=True,
                match_reason="Generic project request matching multiple approved workspaces.",
            )

        # If only 1 workspace exists in total, default to it
        if len(approved) == 1:
            return WorkspaceMatch(
                matched_workspace=approved[0],
                all_matches=approved,
                is_ambiguous=False,
                match_reason=f"Defaulted to sole approved workspace '{approved[0].name}'",
            )

        return WorkspaceMatch(
            matched_workspace=None,
            all_matches=[],
            is_ambiguous=False,
            match_reason="No workspace matched user request.",
        )
