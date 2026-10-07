"""LocLM Agents Framework (V5 Multi-Agent Architecture).

Provides base agent abstractions, multi-agent router, task planner,
V4 & V5 multi-agent orchestrators, memory integration, self-correction,
and specialized agents (Coding, Terminal, Knowledge, General).
"""

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.agents.coding import CodingAgent
from loclm.agents.general_agent import GeneralAgent
from loclm.agents.knowledge_agent import KnowledgeAgent
from loclm.agents.orchestrator import AgentOrchestrator
from loclm.agents.planner import PlannerAgent
from loclm.agents.project_agent import ProjectAgent
from loclm.agents.refactor_agent import RefactorAgent
from loclm.agents.router import RouterAgent
from loclm.agents.runner import AgentLoop
from loclm.agents.terminal_agent import TerminalAgent
from loclm.agents.v5_orchestrator import SelfCorrectionTrace, V5AgentResponse, V5Orchestrator

__all__ = [
    "AgentContext",
    "AgentResponse",
    "AgentRole",
    "BaseAgent",
    "CodingAgent",
    "ProjectAgent",
    "RefactorAgent",
    "TerminalAgent",
    "KnowledgeAgent",
    "GeneralAgent",
    "RouterAgent",
    "PlannerAgent",
    "AgentOrchestrator",
    "AgentLoop",
    "V5Orchestrator",
    "V5AgentResponse",
    "SelfCorrectionTrace",
]

