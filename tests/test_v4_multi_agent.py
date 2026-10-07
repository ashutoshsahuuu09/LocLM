"""Tests for LocLM V4 Multi-Agent Architecture (Router, Planner, Orchestrator)."""

import pytest
from loclm.agents.base import AgentRole
from loclm.agents.router import RouterAgent


@pytest.mark.asyncio
async def test_router_fast_heuristics():
    # Router initialized with None model manager to test fast rule matching
    router = RouterAgent(model_manager=None)  # type: ignore

    assert await router.route("Write a python function to compute fibonacci") == AgentRole.CODING
    assert await router.route("Run git status in terminal") == AgentRole.TERMINAL
    assert await router.route("Search files for config settings") == AgentRole.KNOWLEDGE
