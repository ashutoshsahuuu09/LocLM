"""Tests for LocLM V6 Autonomous Codebase Refactoring & Multi-Repository Coordination."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from loclm.agents.base import AgentRole
from loclm.agents.router import RouterAgent
from loclm.repo.manager import MultiRepoManager
from loclm.testing.runner import RegressionTestRunner
from loclm.tools.ast_tools import ASTRenameSymbolTool, ASTValidateSyntaxTool
from loclm.tools.security import SecurityGuard


@pytest.mark.asyncio
async def test_multi_repo_manager(tmp_path: Path):
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()

    file1 = repo_dir / "calculator.py"
    file1.write_text(
        'class Calculator:\n    """Simple calc class."""\n    def add(self, a, b):\n        return a + b\n',
        encoding="utf-8",
    )

    manager = MultiRepoManager()
    metadata = manager.register_repo(repo_dir, repo_id="sample_repo")

    assert metadata.file_count == 1
    assert len(metadata.symbols) == 2

    # Find Calculator class symbol
    calc_syms = manager.find_symbol("Calculator")
    assert len(calc_syms) == 1
    assert calc_syms[0].kind == "class"
    assert calc_syms[0].line_number == 1

    # Find add method symbol
    add_syms = manager.find_symbol("add")
    assert len(add_syms) == 1
    assert add_syms[0].kind == "function"


@pytest.mark.asyncio
async def test_ast_validate_syntax_tool():
    tool = ASTValidateSyntaxTool()

    # Valid syntax
    res_valid = await tool.execute(code="def foo():\n    return 42\n")
    assert res_valid.success is True
    assert res_valid.metadata["valid"] is True

    # Invalid syntax
    res_invalid = await tool.execute(code="def foo(:\n    return 42\n")
    assert res_invalid.success is False
    assert "SyntaxError" in res_invalid.error


@pytest.mark.asyncio
async def test_ast_rename_symbol_tool(tmp_path: Path):
    guard = SecurityGuard(workspace_root=tmp_path)
    tool = ASTRenameSymbolTool(guard)

    test_file = tmp_path / "service.py"
    test_file.write_text(
        "def process_data(item):\n    return item * 2\n\nval = process_data(10)\n",
        encoding="utf-8",
    )

    res = await tool.execute(
        path=str(test_file),
        old_name="process_data",
        new_name="transform_data",
    )

    assert res.success is True
    new_content = test_file.read_text(encoding="utf-8")
    assert "def transform_data(item):" in new_content
    assert "val = transform_data(10)" in new_content
    assert "process_data" not in new_content


@pytest.mark.asyncio
async def test_regression_test_runner(tmp_path: Path):
    runner = RegressionTestRunner()
    res = await runner.run_tests(cwd=tmp_path, test_command='python -c "print(\'Tests passed\')"')

    assert res.success is True
    assert "Tests passed" in res.output


@pytest.mark.asyncio
async def test_refactor_agent_routing():
    router = RouterAgent(model_manager=None)  # type: ignore
    role = await router.route("refactor process_data function in main.py")
    assert role == AgentRole.REFACTORING
