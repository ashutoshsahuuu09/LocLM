"""Tests for LocLM V2 Local Tools and Security Guard."""

import asyncio
from pathlib import Path

import pytest
from loclm.tools.filesystem import FileInfoTool, ListDirTool, ReadFileTool, SearchFilesTool, WriteFileTool
from loclm.tools.git_tools import GitBranchTool, GitDiffTool, GitLogTool, GitStatusTool
from loclm.tools.python_exec import RunPythonScriptTool
from loclm.tools.registry import ToolRegistry
from loclm.tools.security import SecurityGuard
from loclm.tools.terminal import RunTerminalCommandTool


def test_security_guard_path_validation(tmp_path: Path):
    guard = SecurityGuard(workspace_root=tmp_path)

    # Valid path inside workspace
    test_file = tmp_path / "hello.txt"
    ok, _ = guard.validate_path(test_file)
    assert ok is True

    # Blocked system paths
    win_sys = Path("C:\\Windows\\System32\\cmd.exe")
    ok, reason = guard.validate_path(win_sys)
    assert ok is False
    assert "strictly blocked" in reason.lower()


def test_security_guard_command_validation():
    guard = SecurityGuard()

    # Safe command
    ok, _ = guard.validate_command("python --version")
    assert ok is True

    # Forbidden command
    ok, reason = guard.validate_command("rm -rf /")
    assert ok is False
    assert "forbidden" in reason.lower()


@pytest.mark.asyncio
async def test_filesystem_tools(tmp_path: Path):
    guard = SecurityGuard(workspace_root=tmp_path)
    write_tool = WriteFileTool(guard)
    read_tool = ReadFileTool(guard)
    list_tool = ListDirTool(guard)

    target_file = tmp_path / "sample.txt"

    # Write file
    w_res = await write_tool.execute(path=str(target_file), content="Hello LocLM V2 Tools!")
    assert w_res.success is True
    assert target_file.exists()

    # Read file
    r_res = await read_tool.execute(path=str(target_file))
    assert r_res.success is True
    assert r_res.output == "Hello LocLM V2 Tools!"

    # List dir
    l_res = await list_tool.execute(path=str(tmp_path))
    assert l_res.success is True
    assert "sample.txt" in l_res.output


@pytest.mark.asyncio
async def test_python_exec_tool():
    tool = RunPythonScriptTool()
    res = await tool.execute(code="print(5 * 5)")
    assert res.success is True
    assert res.output.strip() == "25"


@pytest.mark.asyncio
async def test_terminal_command_tool(tmp_path: Path):
    tool = RunTerminalCommandTool()
    res = await tool.execute(command='echo "LocLM Terminal Tool"', cwd=str(tmp_path))
    assert res.success is True
    assert "LocLM Terminal Tool" in res.output


@pytest.mark.asyncio
async def test_github_auth_tool():
    from loclm.tools.github import GitHubAuthTool
    tool = GitHubAuthTool()
    res = await tool.execute(action="status")
    # Execute returns status result regardless of login state
    assert res.output is not None
    assert "GitHub Authentication Status" in res.output


@pytest.mark.asyncio
async def test_tool_registry():
    registry = ToolRegistry()
    tools = registry.list_tools()
    assert len(tools) == 18

    schema = registry.export_tools_schema()
    assert len(schema) == 18

    # Execute read_file through registry
    res = await registry.execute_tool("list_dir", path=".")
    assert res.success is True

