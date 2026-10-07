"""Comprehensive Pytest Suite for LocLM V8 — Swarm, Workflow DAG, Consensus & Policy Guard."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent, ExecutionStep
from loclm.agents.consensus import AgentProposal, ConsensusManager, ConsensusResult, ConsensusStatus
from loclm.agents.swarm import SwarmMessage, SwarmOrchestrator, SwarmResult, SwarmRole
from loclm.agents.v8_orchestrator import V8AgentResponse, V8Orchestrator
from loclm.models.manager import ModelManager
from loclm.security.guard import AuditRecord, PolicyGuard, PolicyViolationError, RiskLevel, SecurityPolicy
from loclm.workflow.dag import StepStatus, WorkflowDAG, WorkflowEngine, WorkflowState, WorkflowStep


# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------


class DummyAgent(BaseAgent):
    """Dummy agent implementation for testing DAG engine."""

    def __init__(self, role: AgentRole = AgentRole.GENERAL) -> None:
        self.role = role
        self.name = f"dummy_{role.value}"
        self.description = "Dummy agent for testing"

    async def run(self, context: AgentContext) -> AgentResponse:
        return AgentResponse(
            content=f"Executed step task: {context.task[:40]}",
            execution_steps=[
                ExecutionStep(
                    step_number=1,
                    thought=f"Dummy execution for {self.role.value}",
                    action_summary="Completed dummy action",
                )
            ],
            is_complete=True,
        )


def make_mock_model_manager(response_text: str = "Mocked LLM Response") -> MagicMock:
    """Create a mock ModelManager with canned async chat response."""
    mm = MagicMock(spec=ModelManager)
    mm.chat = AsyncMock(return_value=response_text)
    return mm


# ---------------------------------------------------------------------------
# 1. Workflow DAG Tests
# ---------------------------------------------------------------------------


def test_workflow_dag_topological_levels():
    dag = WorkflowDAG(workflow_id="test_dag")

    s1 = WorkflowStep(step_id="step_1", instruction="Fetch data", assigned_role=AgentRole.KNOWLEDGE)
    s2 = WorkflowStep(step_id="step_2", instruction="Process data A", depends_on=["step_1"])
    s3 = WorkflowStep(step_id="step_3", instruction="Process data B", depends_on=["step_1"])
    s4 = WorkflowStep(step_id="step_4", instruction="Combine output", depends_on=["step_2", "step_3"])

    dag.add_step(s1)
    dag.add_step(s2)
    dag.add_step(s3)
    dag.add_step(s4)

    levels = dag.get_topological_levels()
    assert len(levels) == 3
    assert levels[0] == ["step_1"]
    assert set(levels[1]) == {"step_2", "step_3"}  # Steps 2 & 3 run concurrently
    assert levels[2] == ["step_4"]


def test_workflow_dag_cycle_detection():
    dag = WorkflowDAG(workflow_id="cycle_dag")

    s1 = WorkflowStep(step_id="step_1", instruction="Task 1", depends_on=["step_2"])
    s2 = WorkflowStep(step_id="step_2", instruction="Task 2", depends_on=["step_1"])

    dag.add_step(s1)
    dag.add_step(s2)

    with pytest.raises(ValueError, match="Cycle detected"):
        dag.get_topological_levels()


def test_workflow_dag_missing_dependency():
    dag = WorkflowDAG(workflow_id="missing_dep")

    s1 = WorkflowStep(step_id="step_1", instruction="Task 1", depends_on=["non_existent"])
    dag.add_step(s1)

    with pytest.raises(ValueError, match="non-existent dependency"):
        dag.validate()


@pytest.mark.asyncio
async def test_workflow_engine_execution():
    dag = WorkflowDAG(workflow_id="exec_dag")

    s1 = WorkflowStep(step_id="step_1", instruction="Gather facts", assigned_role=AgentRole.KNOWLEDGE)
    s2 = WorkflowStep(step_id="step_2", instruction="Write code", assigned_role=AgentRole.CODING, depends_on=["step_1"])
    dag.add_step(s1)
    dag.add_step(s2)

    agent_map = {
        AgentRole.KNOWLEDGE: DummyAgent(AgentRole.KNOWLEDGE),
        AgentRole.CODING: DummyAgent(AgentRole.CODING),
        AgentRole.GENERAL: DummyAgent(AgentRole.GENERAL),
    }

    engine = WorkflowEngine(agent_map)
    state = await engine.execute_dag(dag, user_context="Global User Goal")

    assert state.is_completed is True
    assert state.has_errors is False
    assert state.steps["step_1"].status == StepStatus.COMPLETED
    assert state.steps["step_2"].status == StepStatus.COMPLETED
    assert "Executed step task" in state.steps["step_1"].output


@pytest.mark.asyncio
async def test_workflow_engine_conditional_skipped():
    dag = WorkflowDAG(workflow_id="cond_dag")

    s1 = WorkflowStep(step_id="step_1", instruction="Run test", assigned_role=AgentRole.CODING)
    s2 = WorkflowStep(
        step_id="step_2",
        instruction="Deploy app",
        assigned_role=AgentRole.TERMINAL,
        depends_on=["step_1"],
        condition="contains(step_1, 'SUCCESS')",
    )
    dag.add_step(s1)
    dag.add_step(s2)

    agent_map = {
        AgentRole.CODING: DummyAgent(AgentRole.CODING),
        AgentRole.TERMINAL: DummyAgent(AgentRole.TERMINAL),
        AgentRole.GENERAL: DummyAgent(AgentRole.GENERAL),
    }

    engine = WorkflowEngine(agent_map)
    state = await engine.execute_dag(dag, user_context="Deploy task")

    # Step 1 output from DummyAgent doesn't contain 'SUCCESS', so step 2 should be SKIPPED
    assert state.steps["step_1"].status == StepStatus.COMPLETED
    assert state.steps["step_2"].status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# 2. Multi-Agent Consensus Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consensus_manager_run():
    mm = make_mock_model_manager("Approach Title: Clean Refactoring\nDetailed Solution: Modular architecture.")
    cm = ConsensusManager(mm)

    res = await cm.run_consensus("Refactor database access layer", roles=[AgentRole.CODING, AgentRole.KNOWLEDGE])

    assert isinstance(res, ConsensusResult)
    assert res.consensus_status in (ConsensusStatus.AGREEMENT, ConsensusStatus.MAJORITY)
    assert res.agreement_score > 0.0
    assert len(res.proposals) == 2
    assert "CONSENSUS EXECUTION PLAN" in res.synthesized_plan


def test_consensus_score_calculation():
    p1 = AgentProposal(
        agent_role=AgentRole.CODING,
        proposal_title="Approach A",
        solution_summary="Use SQLite database with indexing",
        confidence_score=0.9,
    )
    p2 = AgentProposal(
        agent_role=AgentRole.KNOWLEDGE,
        proposal_title="Approach B",
        solution_summary="Use SQLite database with persistent indexing",
        confidence_score=0.85,
    )

    score, status = ConsensusManager._evaluate_consensus_score([p1, p2])
    assert score >= 0.7
    assert status == ConsensusStatus.AGREEMENT


# ---------------------------------------------------------------------------
# 3. Multi-Agent Swarm Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_swarm_orchestrator_execution():
    mm = make_mock_model_manager("APPROVED: Solution looks complete and correct.")
    swarm = SwarmOrchestrator(mm, max_rounds=2)

    res = await swarm.execute_swarm("Implement a thread-safe LRU Cache in Python")

    assert isinstance(res, SwarmResult)
    assert res.swarm_success is True
    assert res.rounds_executed >= 1
    assert len(res.message_trace) >= 4  # Leader, Worker, Critic, Verifier
    assert res.message_trace[0].sender == SwarmRole.LEADER
    assert res.message_trace[-1].sender == SwarmRole.VERIFIER


# ---------------------------------------------------------------------------
# 4. Security Policy Guard Tests
# ---------------------------------------------------------------------------


def test_policy_guard_shell_command_blocking():
    guard = PolicyGuard()

    # Safe command
    assert guard.audit_shell_command("pytest tests/") is True

    # Dangerous command should be blocked
    with pytest.raises(PolicyViolationError, match="dangerous pattern"):
        guard.audit_shell_command("rm -rf /")

    with pytest.raises(PolicyViolationError, match="dangerous pattern"):
        guard.audit_shell_command("format C:")


def test_policy_guard_path_access_blocking():
    guard = PolicyGuard()

    # Safe path
    assert guard.audit_path_access("loclm/main.py") is True

    # Forbidden Windows system path
    with pytest.raises(PolicyViolationError, match="restricted system location"):
        guard.audit_path_access("C:\\Windows\\System32\\cmd.exe")

    # Forbidden Linux path
    with pytest.raises(PolicyViolationError, match="restricted system location"):
        guard.audit_path_access("/etc/shadow")


def test_policy_guard_audit_trail_record():
    guard = PolicyGuard()
    guard.audit_tool_call("write_to_file", {"TargetFile": "test.py"}, role=AgentRole.CODING)

    trail = guard.get_audit_trail()
    assert len(trail) >= 1
    assert trail[0].action_type in ("file_access", "tool_call")
    assert trail[0].agent_role == AgentRole.CODING


# ---------------------------------------------------------------------------
# 5. V8 Universal Orchestrator Integration Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v8_orchestrator_pipeline():
    mm = make_mock_model_manager("Final V8 Output: Task successfully completed.")
    policy_guard = PolicyGuard()

    orchestrator = V8Orchestrator(
        model_manager=mm,
        policy_guard=policy_guard,
        plugin_dir=None,
        enable_consensus=True,
        enable_swarm=True,
        enable_dag_workflow=True,
    )

    response = await orchestrator.execute_task("Analyze code quality and optimize memory layout")

    assert isinstance(response, V8AgentResponse)
    assert response.task == "Analyze code quality and optimize memory layout"
    assert "Task successfully completed" in response.final_output
    assert response.consensus_result is not None
    assert response.swarm_result is not None
    assert response.workflow_state is not None
    assert len(response.audit_trail) > 0
    assert response.total_execution_time_sec >= 0.0
