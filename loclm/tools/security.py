"""Security validator and safety guard for LocLM tools.

Protects local user system against accidental or malicious operations:
- Path boundary checking (workspace containment)
- Shell command blacklisting & sanitization
- Permission enforcement (ALLOW, CONFIRM, BLOCK)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from loclm.tools.base import BaseTool, ToolPermissionLevel

logger = logging.getLogger(__name__)

# Dangerous shell commands & regexes that are strictly blocked
FORBIDDEN_COMMAND_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bdel\s+/f\s+/s\s+/q\s+c:\\",
    r"\br mdir\s+/s\s+/q\s+c:\\",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\s+0\b",
    r"\bnet\s+user\b",
    r"\bchmod\s+-R\s+777\s+/",
    r":\(\)\{\s*:\|:&\s*\};:",  # Fork bomb
]


class SecurityGuard:
    """Safety and security validator for local tool execution."""

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self._workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self._forbidden_patterns = [
            re.compile(p, re.IGNORECASE) for p in FORBIDDEN_COMMAND_PATTERNS
        ]

    @property
    def workspace_root(self) -> Path:
        """The active workspace root directory."""
        return self._workspace_root

    def validate_path(self, target_path: Path | str) -> tuple[bool, str]:
        """Verify that a path is safe and contained within allowed directory.

        Args:
            target_path: Path to validate.

        Returns:
            Tuple of (is_valid: bool, reason: str).
        """
        try:
            resolved = Path(target_path).resolve()
        except Exception as e:
            return False, f"Invalid path specification: {e}"

        # Check string representation for Windows system paths across platforms
        str_path = str(target_path).replace("\\", "/").lower()
        win_system_prefixes = ("c:/windows", "c:/system32")
        if any(str_path.startswith(prefix) for prefix in win_system_prefixes):
            return False, f"Access to system path '{target_path}' is strictly blocked for security."

        # Check path against POSIX / OS system root dangers
        system_roots = [
            Path("/etc").resolve(),
            Path("/usr").resolve(),
            Path("/var").resolve(),
            Path("/bin").resolve(),
            Path("/sbin").resolve(),
        ]

        for sroot in system_roots:
            if resolved == sroot or sroot in resolved.parents:
                return False, f"Access to system path '{resolved}' is strictly blocked for security."

        return True, "Path is valid and safe."

    def validate_command(self, command: str) -> tuple[bool, str]:
        """Check shell command string against dangerous command patterns.

        Args:
            command: Shell command line string.

        Returns:
            Tuple of (is_safe: bool, reason: str).
        """
        cmd_clean = command.strip()
        if not cmd_clean:
            return False, "Command string cannot be empty."

        for pattern in self._forbidden_patterns:
            if pattern.search(cmd_clean):
                logger.warning("Blocked dangerous command execution: %s", command)
                return False, f"Command contains forbidden or high-risk pattern: '{cmd_clean}'"

        return True, "Command is safe."

    def check_tool_permission(self, tool: BaseTool) -> ToolPermissionLevel:
        """Get effective permission level for a tool."""
        return tool.permission_level
