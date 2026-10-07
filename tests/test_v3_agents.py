"""Tests for LocLM V3 Agents and Tool Parser."""

import pytest
from loclm.agents.base import AgentContext, AgentRole, ToolCall
from loclm.agents.tool_parser import parse_tool_call


def test_parse_tool_call_xml():
    response = """
    I need to read the configuration file.
    <tool_call>
    {"name": "read_file", "arguments": {"path": "config/config.yaml"}}
    </tool_call>
    """
    call = parse_tool_call(response)
    assert call is not None
    assert call.tool_name == "read_file"
    assert call.arguments == {"path": "config/config.yaml"}


def test_parse_tool_call_json_block():
    response = """
    ```json
    {
        "tool": "list_dir",
        "arguments": {"path": "."}
    }
    ```
    """
    call = parse_tool_call(response)
    assert call is not None
    assert call.tool_name == "list_dir"
    assert call.arguments == {"path": "."}


def test_parse_tool_call_action_format():
    response = """
    {"action": "git_status", "action_input": {}}
    """
    call = parse_tool_call(response)
    assert call is not None
    assert call.tool_name == "git_status"


def test_parse_tool_call_text_tag():
    response = "<tool_call> search_files query=ModelSelector </tool_call>"
    call = parse_tool_call(response)
    assert call is not None
    assert call.tool_name == "search_files"
    assert call.arguments == {"query": "ModelSelector"}


def test_parse_tool_call_none():
    response = "This is just a regular chat message with no tool calls."
    call = parse_tool_call(response)
    assert call is None


def test_agent_context_defaults():
    ctx = AgentContext(task="Fix bug in main.py")
    assert ctx.task == "Fix bug in main.py"
    assert ctx.history == []
