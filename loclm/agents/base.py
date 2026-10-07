"""Base classes and data structures for LocLM Agents (V3).

All specialized agents (CodingAgent, KnowledgeAgent, RouterAgent, etc.)
inherit from BaseAgent.
"""

from __future__ import annotations

import abc
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loclm.models.base import ChatMessage
from loclm.tools.base import ToolResult


class AgentRole(StrEnum):
    """Role/specialization of an agent."""

    ROUTER = "router"
    CODING = "coding"
    REASONING = "reasoning"
    GENERAL = "general"
    TERMINAL = "terminal"
    KNOWLEDGE = "knowledge"
    PROJECT = "project"
    REFACTORING = "refactoring"




class ToolCall(BaseModel):
    """Structured call specification for a tool execution."""

    tool_name: str = Field(description="Name of the tool to execute")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Key-value arguments for tool")
    call_id: str = Field(default="", description="Unique identifier for the tool call")


class ExecutionStep(BaseModel):
    """A single reasoning + tool action step in agent execution loop."""

    step_number: int = Field(description="1-indexed step number")
    thought: str = Field(default="", description="Agent reasoning/thought before acting")
    tool_call: ToolCall | None = Field(default=None, description="Tool requested by agent")
    tool_result: ToolResult | None = Field(default=None, description="Output returned by tool")


class AgentContext(BaseModel):
    """Execution context supplied to an agent."""

    task: str = Field(description="Primary task or goal statement for agent")
    codebase_path: Path = Field(default_factory=Path.cwd, description="Working directory / workspace root")
    history: list[ChatMessage] = Field(default_factory=list, description="Prior conversation messages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context variables")


class AgentResponse(BaseModel):
    """Final output response produced by an agent."""

    content: str = Field(description="Final text response / synthesis")
    execution_steps: list[ExecutionStep] = Field(default_factory=list, description="Sequence of steps taken")
    is_complete: bool = Field(default=True, description="True if task goal was achieved")
    total_tokens: int = Field(default=0, description="Total tokens generated during task")


class BaseAgent(abc.ABC):
    """Abstract base class for all LocLM autonomous agents."""

    name: str
    role: AgentRole
    description: str

    @abc.abstractmethod
    async def run(self, context: AgentContext) -> AgentResponse:
        """Run the agent autonomously on the given task context.

        Args:
            context: AgentContext containing task and environment info.

        Returns:
            AgentResponse containing final output and step history.
        """
        ...
