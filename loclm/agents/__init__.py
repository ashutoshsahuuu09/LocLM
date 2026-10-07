"""LocLM Agents Framework (V8 Architecture).

Provides base agent abstractions, multi-agent router, task planner,
V5/V7/V8 orchestrators, memory integration, self-correction,
specialised agents (Coding, Terminal, Knowledge, General, Project, Refactor),
EvaluatorAgent, Multi-Agent Swarm Orchestrator, Consensus Manager,
and the V8 Autonomous Workflow DAG Orchestrator with Policy Security Guard.
"""

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.agents.coding import CodingAgent
from loclm.agents.consensus import AgentProposal, ConsensusManager, ConsensusResult, ConsensusStatus
from loclm.agents.evaluator_agent import EvaluationResult, EvaluatorAgent
from loclm.agents.general_agent import GeneralAgent
from loclm.agents.knowledge_agent import KnowledgeAgent
from loclm.agents.orchestrator import AgentOrchestrator
from loclm.agents.planner import PlannerAgent
from loclm.agents.project_agent import ProjectAgent
from loclm.agents.refactor_agent import RefactorAgent
from loclm.agents.router import RouterAgent
from loclm.agents.runner import AgentLoop
from loclm.agents.swarm import SwarmMessage, SwarmOrchestrator, SwarmResult, SwarmRole
from loclm.agents.terminal_agent import TerminalAgent
from loclm.agents.v5_orchestrator import SelfCorrectionTrace, V5AgentResponse, V5Orchestrator
from loclm.agents.v7_orchestrator import V7AgentResponse, V7Orchestrator
from loclm.agents.v8_orchestrator import V8AgentResponse, V8Orchestrator

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
    # V5
    "V5Orchestrator",
    "V5AgentResponse",
    "SelfCorrectionTrace",
    # V7
    "EvaluatorAgent",
    "EvaluationResult",
    "V7Orchestrator",
    "V7AgentResponse",
    # V8
    "V8Orchestrator",
    "V8AgentResponse",
    "ConsensusManager",
    "ConsensusResult",
    "ConsensusStatus",
    "AgentProposal",
    "SwarmOrchestrator",
    "SwarmResult",
    "SwarmRole",
    "SwarmMessage",
]
