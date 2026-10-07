"""Robust tool call parser for local LLMs (V3).

Parses structured tool calls from text responses produced by local models.
Supports XML tags (<tool_call>), JSON blocks, and text key-value tool calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from loclm.agents.base import ToolCall

logger = logging.getLogger(__name__)


def parse_tool_call(response_text: str) -> ToolCall | None:
    """Extract a ToolCall from model response text.

    Supported Formats:
    1. XML tag format with JSON:
       <tool_call>
       {"name": "read_file", "arguments": {"path": "README.md"}}
       </tool_call>

    2. XML tag format with text parameters:
       <tool_call> search_files query=ModelSelector </tool_call>

    3. JSON code block format:
       ```json
       {"tool": "read_file", "arguments": {"path": "README.md"}}
       ```

    4. Action format:
       {"action": "read_file", "action_input": {"path": "README.md"}}

    Args:
        response_text: Raw text produced by local LLM.

    Returns:
        ToolCall object if a valid tool call was parsed, or None.
    """
    if not response_text or not response_text.strip():
        return None

    # Format 1: XML tag <tool_call> containing JSON
    xml_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    xml_match = re.search(xml_pattern, response_text, re.DOTALL)
    if xml_match:
        data = _safe_json_loads(xml_match.group(1))
        if data and isinstance(data, dict):
            call = _dict_to_tool_call(data)
            if call:
                return call

    # Format 2: XML tag <tool_call> containing text tool call
    text_tag_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    tag_matches = re.findall(text_tag_pattern, response_text, re.DOTALL)
    for tag_content in tag_matches:
        tag_content = tag_content.strip()
        if not tag_content:
            continue
        # Strip nested tags if any
        tag_content = re.sub(r"</?tool_call>", "", tag_content).strip()
        call = _parse_text_tool_call(tag_content)
        if call:
            return call

    # Format 3: Fenced ```json ... ``` code block containing tool call
    json_block_pattern = r"```(?:json)?\s*({.*?})\s*```"
    json_matches = re.findall(json_block_pattern, response_text, re.DOTALL)
    for block in json_matches:
        data = _safe_json_loads(block)
        if data and isinstance(data, dict):
            call = _dict_to_tool_call(data)
            if call:
                return call

    # Format 4: Raw JSON object in text
    raw_json_pattern = r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})"
    raw_matches = re.findall(raw_json_pattern, response_text, re.DOTALL)
    for candidate in raw_matches:
        data = _safe_json_loads(candidate)
        if data and isinstance(data, dict):
            call = _dict_to_tool_call(data)
            if call:
                return call

    return None


def _parse_text_tool_call(text: str) -> ToolCall | None:
    """Parse a plain text tool call like 'search_files query=ModelSelector'."""
    parts = text.split()
    if not parts:
        return None

    tool_name = parts[0].strip()
    # Check if tool_name is clean identifier
    if not re.match(r"^[a-zA-Z0-9_]+$", tool_name):
        return None

    args: dict[str, Any] = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            args[k.strip()] = v.strip().strip('"').strip("'")
        else:
            if "query" not in args and "path" not in args:
                args["query"] = part.strip()

    return ToolCall(tool_name=tool_name, arguments=args)


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    """Safely parse JSON string into dict."""
    try:
        data = json.loads(text.strip())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _dict_to_tool_call(data: dict[str, Any]) -> ToolCall | None:
    """Convert dictionary data into a ToolCall object."""
    name = data.get("name") or data.get("tool") or data.get("action")
    if not name or not isinstance(name, str):
        return None

    args = data.get("arguments") or data.get("args") or data.get("action_input") or {}
    if not isinstance(args, dict):
        args = {}

    return ToolCall(tool_name=name.strip(), arguments=args)
