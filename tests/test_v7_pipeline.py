"""Tests for LocLM V7 — Plugin System, Session Memory, Parallel Execution & EvaluatorAgent."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from loclm.agents.base import AgentContext, AgentResponse, AgentRole
from loclm.agents.evaluator_agent import (
    EvaluationDimension,
    EvaluationResult,
    EvaluatorAgent,
)
from loclm.agents.router import RouterAgent
from loclm.agents.v7_orchestrator import V7AgentResponse, V7Orchestrator
from loclm.memory.session import SessionMemory, SessionTurn
from loclm.plugins.loader import PluginLoader, PluginManifest
from loclm.plugins.registry import PluginRegistry
from loclm.tools.base import BaseTool, ToolCategory, ToolPermissionLevel, ToolResult
from loclm.tools.registry import ToolRegistry
from loclm.tools.security import SecurityGuard


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_mock_model_manager(response_text: str = "OK") -> MagicMock:
    """Create a ModelManager mock that always returns ``response_text``."""
    mm = MagicMock()
    mm.chat = AsyncMock(return_value=response_text)
    mm.chat_stream = AsyncMock()
    return mm


# ===========================================================================
# 1. SessionMemory Tests
# ===========================================================================


class TestSessionMemory:
    """Tests for loclm.memory.session.SessionMemory."""

    def setup_method(self):
        self.session = SessionMemory(db_path=":memory:", session_id="test-session-abc")

    def test_record_and_retrieve_user_message(self):
        self.session.record_user_message("Hello, LocLM!")
        turns = self.session.get_recent_turns()
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert "Hello" in turns[0].content

    def test_record_assistant_message(self):
        self.session.record_user_message("Write a sort function")
        self.session.record_assistant_message("Here is a sort function: ...")
        turns = self.session.get_recent_turns()
        assert len(turns) == 2
        assert turns[-1].role == "assistant"

    def test_record_tool_event(self):
        turn = self.session.record_tool_event(
            tool_name="read_file",
            result_summary="Read 200 lines",
            success=True,
        )
        assert turn.role == "tool"
        assert "read_file" in turn.content
        assert "OK" in turn.content

    def test_get_context_messages_excludes_tool_turns(self):
        self.session.record_user_message("Ask something")
        self.session.record_tool_event("run_terminal", "Exit 0", success=True)
        self.session.record_assistant_message("Done!")
        messages = self.session.get_context_messages()
        # Tool turns should be excluded
        roles = [m.role for m in messages]
        assert "tool" not in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_search_turns_keyword(self):
        self.session.record_user_message("Explain async/await in Python")
        self.session.record_user_message("What is a class decorator?")
        results = self.session.search_turns("async")
        assert len(results) >= 1
        assert "async" in results[0].content.lower()

    def test_session_summary_prefix_empty(self):
        fresh = SessionMemory(db_path=":memory:")
        assert fresh.get_session_summary_prefix() == ""

    def test_session_summary_prefix_with_turns(self):
        self.session.record_user_message("Hello")
        prefix = self.session.get_session_summary_prefix()
        assert "turn" in prefix.lower()
        # session_id is truncated to 8 chars in the prefix
        assert self.session.session_id[:8] in prefix

    def test_max_context_turns_limiting(self):
        session = SessionMemory(db_path=":memory:", max_context_turns=3)
        for i in range(10):
            session.record_user_message(f"Message {i}")
        turns = session.get_recent_turns()
        assert len(turns) == 3

    def test_list_sessions(self):
        s1 = SessionMemory(db_path=":memory:", session_id="session-1")
        s2 = SessionMemory(db_path=s1.db_path, session_id="session-2")
        # Both share the same in-memory conn only if same object — test isolation:
        self.session.record_user_message("turn A")
        sessions = self.session.list_sessions()
        assert "test-session-abc" in sessions

    def test_clear_session(self):
        self.session.record_user_message("Will be deleted")
        self.session.clear_session()
        turns = self.session.get_recent_turns()
        assert len(turns) == 0

    def test_session_turn_to_chat_message(self):
        self.session.record_user_message("Convert me")
        turns = self.session.get_recent_turns()
        cm = turns[0].to_chat_message()
        assert cm.role == "user"
        assert "Convert me" in cm.content


# ===========================================================================
# 2. EvaluatorAgent Tests
# ===========================================================================


class TestEvaluatorAgent:
    """Tests for loclm.agents.evaluator_agent.EvaluatorAgent."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_result(self):
        json_response = """{
            "correctness": {"score": 0.9, "rationale": "Factually accurate"},
            "completeness": {"score": 0.8, "rationale": "Covers main points"},
            "safety": {"score": 1.0, "rationale": "No safety concerns"},
            "relevance": {"score": 0.85, "rationale": "Directly on-topic"}
        }"""
        mm = _make_mock_model_manager(json_response)
        evaluator = EvaluatorAgent(mm, min_quality_threshold=0.55)

        result = await evaluator.evaluate(
            task="Explain Python list comprehensions",
            response="List comprehensions provide a concise way to create lists...",
        )

        assert isinstance(result, EvaluationResult)
        assert result.correctness.score == pytest.approx(0.9)
        assert result.completeness.score == pytest.approx(0.8)
        assert result.safety.score == pytest.approx(1.0)
        assert result.quality_score > 0.5
        assert result.needs_retry is False

    @pytest.mark.asyncio
    async def test_evaluate_low_quality_triggers_retry_flag(self):
        # Scores that produce quality < 0.55
        json_response = """{
            "correctness": {"score": 0.1, "rationale": "Incorrect answer"},
            "completeness": {"score": 0.2, "rationale": "Very incomplete"},
            "safety": {"score": 0.9, "rationale": "Safe"},
            "relevance": {"score": 0.3, "rationale": "Barely relevant"}
        }"""
        mm = _make_mock_model_manager(json_response)
        evaluator = EvaluatorAgent(mm, min_quality_threshold=0.55)

        result = await evaluator.evaluate(
            task="Explain recursion",
            response="Recursion is when something recurses.",
        )

        assert result.needs_retry is True
        assert result.improvement_hint != ""

    @pytest.mark.asyncio
    async def test_evaluate_model_failure_returns_default(self):
        mm = MagicMock()
        mm.chat = AsyncMock(side_effect=RuntimeError("Model offline"))
        evaluator = EvaluatorAgent(mm)

        result = await evaluator.evaluate(task="Test task", response="Test response")

        # Should return a default result, not raise
        assert isinstance(result, EvaluationResult)
        assert result.quality_score == pytest.approx(0.5, abs=0.05)

    @pytest.mark.asyncio
    async def test_evaluate_malformed_json_returns_default(self):
        mm = _make_mock_model_manager("This is not JSON at all!!!")
        evaluator = EvaluatorAgent(mm)

        result = await evaluator.evaluate(task="Task", response="Response")
        assert isinstance(result, EvaluationResult)

    def test_evaluation_result_build_weighted_score(self):
        """Verify the weighted scoring formula: 0.35·C + 0.30·Co + 0.20·S + 0.15·R."""
        c = EvaluationDimension(score=1.0, rationale="")
        co = EvaluationDimension(score=1.0, rationale="")
        s = EvaluationDimension(score=1.0, rationale="")
        r = EvaluationDimension(score=1.0, rationale="")
        result = EvaluationResult.build("task", c, co, s, r)
        assert result.quality_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_run_method_delegates_to_evaluate(self):
        json_response = """{
            "correctness": {"score": 0.8, "rationale": "Good"},
            "completeness": {"score": 0.8, "rationale": "Good"},
            "safety": {"score": 0.8, "rationale": "Good"},
            "relevance": {"score": 0.8, "rationale": "Good"}
        }"""
        mm = _make_mock_model_manager(json_response)
        evaluator = EvaluatorAgent(mm)

        ctx = AgentContext(
            task="Test task",
            metadata={"response": "Test agent output"},
        )
        response = await evaluator.run(ctx)
        assert response.is_complete is True
        assert "quality_score" in response.content


# ===========================================================================
# 3. Plugin System Tests
# ===========================================================================


class _DummyTool(BaseTool):
    """A simple dummy tool for plugin testing."""

    name = "dummy_plugin_tool"
    description = "Dummy tool from test plugin"
    category = ToolCategory.FILESYSTEM
    permission_level = ToolPermissionLevel.ALLOW
    parameters_schema: dict = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="dummy result")


class TestPluginLoader:
    """Tests for loclm.plugins.loader.PluginLoader."""

    def test_discover_nonexistent_dir_returns_empty(self, tmp_path):
        loader = PluginLoader(plugin_dir=tmp_path / "nonexistent")
        manifests = loader.discover()
        assert manifests == []

    def test_discover_empty_dir_returns_empty(self, tmp_path):
        loader = PluginLoader(plugin_dir=tmp_path)
        manifests = loader.discover()
        assert manifests == []

    def test_discover_valid_plugin(self, tmp_path):
        plugin_code = textwrap.dedent("""
            from loclm.tools.base import BaseTool, ToolCategory, ToolPermissionLevel, ToolResult

            PLUGIN_METADATA = {
                "name": "Calculator Plugin",
                "version": "1.2.0",
                "description": "Adds a basic calculator tool",
                "author": "tester",
            }

            class CalculatorTool(BaseTool):
                name = "calculator"
                description = "Performs basic math"
                category = ToolCategory.FILESYSTEM
                permission_level = ToolPermissionLevel.ALLOW

                def get_parameters(self):
                    return []

                async def execute(self, **kwargs) -> ToolResult:
                    return ToolResult(success=True, output="42")
        """)
        (tmp_path / "calculator_plugin.py").write_text(plugin_code, encoding="utf-8")

        loader = PluginLoader(plugin_dir=tmp_path)
        manifests = loader.discover()

        assert len(manifests) == 1
        m = manifests[0]
        assert m.is_valid
        assert m.name == "Calculator Plugin"
        assert m.version == "1.2.0"
        assert m.author == "tester"
        assert len(m.tools) == 1
        assert m.tools[0].__name__ == "CalculatorTool"

    def test_discover_broken_plugin_logs_error(self, tmp_path):
        bad_code = "this is not valid python at all !!!@#$"
        (tmp_path / "bad_plugin.py").write_text(bad_code, encoding="utf-8")

        loader = PluginLoader(plugin_dir=tmp_path)
        manifests = loader.discover()

        assert len(manifests) == 1
        assert manifests[0].is_valid is False
        assert manifests[0].load_error is not None

    def test_plugin_summary_format(self, tmp_path):
        loader = PluginLoader(plugin_dir=tmp_path)
        m = PluginManifest(
            plugin_id="my_plugin",
            file_path=tmp_path / "my_plugin.py",
            name="My Plugin",
            version="2.0.0",
        )
        summary = m.summary()
        assert "my_plugin" in summary
        assert "My Plugin" in summary
        assert "2.0.0" in summary


class TestPluginRegistry:
    """Tests for loclm.plugins.registry.PluginRegistry."""

    def test_load_and_register_tools(self, tmp_path):
        plugin_code = textwrap.dedent("""
            from loclm.tools.base import BaseTool, ToolCategory, ToolPermissionLevel, ToolResult

            class EchoTool(BaseTool):
                name = "echo_tool"
                description = "Echoes input"
                category = ToolCategory.FILESYSTEM
                permission_level = ToolPermissionLevel.ALLOW

                def get_parameters(self):
                    return []

                async def execute(self, **kwargs) -> ToolResult:
                    return ToolResult(success=True, output="echo")
        """)
        (tmp_path / "echo_plugin.py").write_text(plugin_code, encoding="utf-8")

        registry = PluginRegistry(plugin_dir=str(tmp_path))
        tool_reg = ToolRegistry()
        mm = _make_mock_model_manager()

        stats = registry.load_and_register(
            tool_registry=tool_reg,
            model_manager=mm,
            agent_map={},
        )

        assert stats["plugins"] == 1
        assert stats["tools"] == 1
        assert tool_reg.get_tool("echo_tool") is not None

    def test_no_plugins_returns_zero_counts(self, tmp_path):
        registry = PluginRegistry(plugin_dir=str(tmp_path))
        stats = registry.load_and_register(
            tool_registry=ToolRegistry(),
            model_manager=_make_mock_model_manager(),
            agent_map={},
        )
        assert stats["plugins"] == 0
        assert stats["tools"] == 0
        assert stats["agents"] == 0


# ===========================================================================
# 4. V7 Router Fast-Path Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_router_coding_fast_path():
    router = RouterAgent(model_manager=None)  # type: ignore[arg-type]
    role = await router.route("write a Python function to parse JSON")
    assert role == AgentRole.CODING


@pytest.mark.asyncio
async def test_router_terminal_fast_path():
    router = RouterAgent(model_manager=None)  # type: ignore[arg-type]
    role = await router.route("run the test suite with pytest")
    assert role == AgentRole.TERMINAL


@pytest.mark.asyncio
async def test_router_refactoring_fast_path():
    router = RouterAgent(model_manager=None)  # type: ignore[arg-type]
    role = await router.route("refactor the main module")
    assert role == AgentRole.REFACTORING


# ===========================================================================
# 5. V7Orchestrator Integration Tests (mocked model)
# ===========================================================================


def _make_v7_orchestrator(
    response_text: str = "The answer is 42.",
    eval_text: str | None = None,
    plugin_dir: str | None = None,
    tmp_path: Path | None = None,
) -> V7Orchestrator:
    """Build a V7Orchestrator with mocked ModelManager."""
    if eval_text is None:
        eval_text = """{
            "correctness": {"score": 0.8, "rationale": "Good"},
            "completeness": {"score": 0.75, "rationale": "Complete"},
            "safety": {"score": 1.0, "rationale": "Safe"},
            "relevance": {"score": 0.9, "rationale": "Relevant"}
        }"""

    mm = MagicMock()

    async def _chat(messages, **kwargs):
        # Evaluator always gets eval_text, agents get response_text
        content = "\n".join(m.content for m in messages)
        if "quality evaluator" in content.lower() or "correctness" in content.lower():
            return eval_text
        return response_text

    mm.chat = AsyncMock(side_effect=_chat)

    session = SessionMemory(db_path=":memory:")
    return V7Orchestrator(
        model_manager=mm,
        session_memory=session,
        plugin_dir=plugin_dir or (str(tmp_path) if tmp_path else None),
        enable_evaluation=True,
        enable_parallel=False,  # sequential for unit tests
    )


@pytest.mark.asyncio
async def test_v7_orchestrator_basic_response(tmp_path):
    orch = _make_v7_orchestrator(
        response_text="Python is a high-level programming language.",
        tmp_path=tmp_path,
    )
    result = await orch.execute_task("What is Python?")

    assert isinstance(result, V7AgentResponse)
    assert "Python" in result.final_output
    assert result.session_id != ""
    assert result.is_complete


@pytest.mark.asyncio
async def test_v7_session_memory_records_turns(tmp_path):
    orch = _make_v7_orchestrator(tmp_path=tmp_path)

    await orch.execute_task("First question")
    await orch.execute_task("Second question")

    history = orch.get_session_history(n=20)
    user_turns = [t for t in history if t["role"] == "user"]
    assert len(user_turns) >= 2


@pytest.mark.asyncio
async def test_v7_evaluation_included_in_response(tmp_path):
    orch = _make_v7_orchestrator(tmp_path=tmp_path)
    result = await orch.execute_task("Explain list comprehensions in Python")

    assert result.evaluation is not None
    assert 0.0 <= result.evaluation.quality_score <= 1.0


@pytest.mark.asyncio
async def test_v7_no_retry_when_quality_good(tmp_path):
    high_quality_eval = """{
        "correctness": {"score": 0.95, "rationale": "Accurate"},
        "completeness": {"score": 0.90, "rationale": "Complete"},
        "safety": {"score": 1.0, "rationale": "Safe"},
        "relevance": {"score": 0.95, "rationale": "Relevant"}
    }"""
    orch = _make_v7_orchestrator(eval_text=high_quality_eval, tmp_path=tmp_path)
    result = await orch.execute_task("What is a generator in Python?")

    assert result.retry_triggered is False


@pytest.mark.asyncio
async def test_v7_retry_triggered_on_low_quality(tmp_path):
    low_quality_eval = """{
        "correctness": {"score": 0.1, "rationale": "Wrong"},
        "completeness": {"score": 0.1, "rationale": "Incomplete"},
        "safety": {"score": 0.5, "rationale": "Borderline"},
        "relevance": {"score": 0.2, "rationale": "Off-topic"}
    }"""
    orch = _make_v7_orchestrator(
        response_text="I don't know.",
        eval_text=low_quality_eval,
        tmp_path=tmp_path,
    )
    orch.min_quality_threshold = 0.55
    result = await orch.execute_task("Explain Python closures")

    assert result.retry_triggered is True


@pytest.mark.asyncio
async def test_v7_plugins_loaded_count_reported(tmp_path):
    # Plugin dir is empty → 0 plugins
    orch = _make_v7_orchestrator(tmp_path=tmp_path)
    result = await orch.execute_task("Hello!")
    assert result.plugins_loaded == 0


@pytest.mark.asyncio
async def test_v7_parallel_execution(tmp_path):
    """With enable_parallel=True and a complex plan, steps run concurrently."""
    mm = MagicMock()
    mm.chat = AsyncMock(return_value="Step result")

    session = SessionMemory(db_path=":memory:")
    orch = V7Orchestrator(
        model_manager=mm,
        session_memory=session,
        plugin_dir=str(tmp_path),
        enable_evaluation=False,  # skip evaluator to isolate parallel logic
        enable_parallel=True,
    )

    # Force a complex plan with 2 steps by mocking the planner
    from loclm.agents.planner import PlanStep, TaskPlan

    complex_plan = TaskPlan(
        goal="Two-step task",
        is_complex=True,
        steps=[
            PlanStep(
                step_number=1,
                title="Step A",
                assigned_role=AgentRole.GENERAL,
                instruction="Do step A",
            ),
            PlanStep(
                step_number=2,
                title="Step B",
                assigned_role=AgentRole.GENERAL,
                instruction="Do step B",
            ),
        ],
    )
    orch.planner.create_plan = AsyncMock(return_value=complex_plan)

    result = await orch.execute_task("Complex parallel task")

    assert result.parallel_steps_run == 2
    assert result.is_complete


@pytest.mark.asyncio
async def test_v7_get_plugin_summary_empty(tmp_path):
    orch = _make_v7_orchestrator(tmp_path=tmp_path)
    summary = orch.get_plugin_summary()
    assert isinstance(summary, list)
