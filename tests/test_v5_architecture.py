"""Tests for LocLM V5 Autonomous Multi-Agent System with Local Memory & Self-Correction."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, ExecutionStep, ToolCall, ToolResult
from loclm.agents.v5_orchestrator import SelfCorrectionTrace, V5AgentResponse, V5Orchestrator
from loclm.memory.local_store import SQLiteMemoryStore
from loclm.memory.manager import MemoryManager
from loclm.models.cascade import ModelCascadeManager
from loclm.models.base import ChatMessage, TaskType


@pytest.mark.asyncio
async def test_sqlite_memory_store_crud():
    store = SQLiteMemoryStore(":memory:")
    
    # Add entry
    entry = await store.add(
        content="LocLM uses offline Ollama engine for local inference",
        category="fact",
        metadata={"source": "docs"},
    )
    assert entry.entry_id is not None
    assert entry.content == "LocLM uses offline Ollama engine for local inference"
    assert entry.category == "fact"
    assert entry.metadata == {"source": "docs"}

    # Get all entries
    all_entries = await store.get_all()
    assert len(all_entries) == 1
    assert all_entries[0].entry_id == entry.entry_id

    # Clear entries
    await store.clear(category="fact")
    all_after_clear = await store.get_all()
    assert len(all_after_clear) == 0


@pytest.mark.asyncio
async def test_sqlite_memory_store_search():
    store = SQLiteMemoryStore(":memory:")
    await store.add("Configure database connection string in config.yaml", category="config")
    await store.add("FastAPI server routing setup", category="backend")

    results = await store.search("config database settings")
    assert len(results) > 0
    assert "database" in results[0].entry.content
    assert results[0].score > 0.0


@pytest.mark.asyncio
async def test_memory_manager_context_retrieval():
    store = SQLiteMemoryStore(":memory:")
    manager = MemoryManager(store=store)

    await manager.add_fact("Use Python 3.12 for async compatibility")
    await manager.record_task_run("Create main entry", "Successfully created main.py")

    context = await manager.get_relevant_context("Python async version")
    assert "[Relevant Offline Memory Context]:" in context
    assert "Python 3.12" in context


@pytest.mark.asyncio
async def test_model_cascade_manager():
    mock_model_manager = MagicMock()
    mock_model_manager.chat = AsyncMock(return_value="Cascade response success")

    cascade = ModelCascadeManager(mock_model_manager)
    res = await cascade.chat_with_cascade(
        messages=[ChatMessage(role="user", content="Hello")],
        task_type=TaskType.CODING,
    )

    assert res == "Cascade response success"
    mock_model_manager.chat.assert_called_once()


@pytest.mark.asyncio
async def test_v5_orchestrator_execution():
    mock_model_manager = MagicMock()
    mock_model_manager.chat = AsyncMock(return_value="V5 Agent completed request successfully")

    memory_store = SQLiteMemoryStore(":memory:")
    memory_manager = MemoryManager(store=memory_store)

    orchestrator = V5Orchestrator(
        model_manager=mock_model_manager,
        memory_manager=memory_manager,
    )

    response = await orchestrator.execute_task("Write a python script to test API endpoint")

    assert isinstance(response, V5AgentResponse)
    assert response.task == "Write a python script to test API endpoint"
    assert response.final_output != ""
    assert response.is_complete is True
    assert response.total_execution_time_sec >= 0.0

    # Verify task run was recorded into memory
    all_memories = await memory_store.get_all()
    assert len(all_memories) == 1
    assert "Write a python script" in all_memories[0].content


@pytest.mark.asyncio
async def test_v5_self_correction_trace():
    failed_step = ExecutionStep(
        step_number=1,
        thought="Attempting tool execution",
        tool_call=ToolCall(tool_name="run_terminal_command", arguments={"command": "invalid_cmd"}),
        tool_result=ToolResult(success=False, output="", error="Command not found: invalid_cmd"),
    )

    orchestrator = V5Orchestrator(model_manager=MagicMock())
    trace = await orchestrator._attempt_self_correction(
        agent=MagicMock(),
        failed_step=failed_step,
        error_msg="Command not found: invalid_cmd",
    )

    assert isinstance(trace, SelfCorrectionTrace)
    assert trace.original_error == "Command not found: invalid_cmd"
    assert trace.resolved is True
    assert "Analyzing failure" in trace.reflection


@pytest.mark.asyncio
async def test_create_project_scaffold_tool(tmp_path):
    from loclm.tools.filesystem import CreateProjectScaffoldTool
    from loclm.tools.security import SecurityGuard

    guard = SecurityGuard(workspace_root=tmp_path)
    tool = CreateProjectScaffoldTool(guard)

    res = await tool.execute(
        root_path=str(tmp_path / "my_new_app"),
        directories=["src", "config"],
        files={
            "README.md": "# My New App",
            "src/main.py": "print('App started')",
        },
    )

    assert res.success is True
    assert (tmp_path / "my_new_app" / "src" / "main.py").exists()
    assert (tmp_path / "my_new_app" / "README.md").read_text() == "# My New App"


@pytest.mark.asyncio
async def test_project_agent_routing():
    from loclm.agents.router import RouterAgent

    router = RouterAgent(model_manager=None)  # type: ignore
    role = await router.route("create project for a fast API backend")
    assert role == AgentRole.PROJECT

