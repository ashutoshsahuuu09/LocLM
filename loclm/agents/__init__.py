"""LocLM Agents Framework (V4 Multi-Agent Architecture).

Provides base agent abstractions, multi-agent router, task planner,
multi-agent orchestrator, and specialized agents (Coding, Terminal, Knowledge, General).
"""

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.agents.coding import CodingAgent
from loclm.agents.general_agent import GeneralAgent
from loclm.agents.knowledge_agent import KnowledgeAgent
from loclm.agents.orchestrator import AgentOrchestrator
from loclm.agents.planner import PlannerAgent
from loclm.agents.router import RouterAgent
from loclm.agents.runner import AgentLoop
from loclm.agents.terminal_agent import TerminalAgent

__all__ = [
    "AgentContext",
    "AgentResponse",
    "AgentRole",
    "BaseAgent",
    "CodingAgent",
    "TerminalAgent",
    "KnowledgeAgent",
    "GeneralAgent",
    "RouterAgent",
    "PlannerAgent",
    "AgentOrchestrator",
    "AgentLoop",
]
