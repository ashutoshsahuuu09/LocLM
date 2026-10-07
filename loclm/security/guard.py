"""Autonomous Security Policy Guard & Auditing Module — LocLM V8.

Enforces zero-trust policy boundaries, pre-executes security checks on tool calls
and shell commands, blocks destructive patterns, and logs immutable audit records.
"""

from __future__ import annotations

import enum
import logging
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loclm.agents.base import AgentRole

logger = logging.getLogger(__name__)


class RiskLevel(str, enum.Enum):
    """Risk severity level for audited operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditRecord(BaseModel):
    """Immutable audit trail record for an audited system operation."""

    timestamp: float = Field(default_factory=time.time)
    agent_role: AgentRole = Field(default=AgentRole.GENERAL)
    action_type: str = Field(description="Type of action, e.g. tool_call, shell_exec, file_access")
    target_resource: str = Field(description="Target tool, file path, or command")
    allowed: bool = Field(description="True if operation was permitted")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    details: str = Field(default="")


class SecurityPolicy(BaseModel):
    """Configurable enterprise security policy rules for LocLM execution."""

    blocked_command_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\brm\s+-rf\s+[/|\\]",
            r"\bformat\s+[a-z]:",
            r"\bmkfs\b",
            r"\bdd\s+if=",
            r":\(\)\{\s*:\|:&\s*\};:",  # Fork bomb
            r"\bdel\s+/s\s+/q\s+c:\*",
            r"\bchmod\s+-R\s+777\s+/",
        ]
    )
    forbidden_path_prefixes: list[str] = Field(
        default_factory=lambda: [
            r"^[a-zA-Z]:\\Windows\\System32",
            r"^[a-zA-Z]:\\Windows\\SysWOW64",
            r"^/etc/shadow",
            r"^/etc/passwd",
            r"^/root",
            r"^/sys",
            r"^/proc",
        ]
    )
    max_token_budget_per_turn: int = Field(default=16384)
    strict_sandbox: bool = Field(default=True)
    allow_terminal_execution: bool = Field(default=True)


class PolicyViolationError(Exception):
    """Raised when an operation violates security policy rules."""

    pass


class PolicyGuard:
    """Security Policy Auditor & Enforcement Guard for LocLM tools and commands."""

    def __init__(self, policy: SecurityPolicy | None = None) -> None:
        self.policy = policy or SecurityPolicy()
        self._audit_trail: list[AuditRecord] = []

    def audit_tool_call(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        role: AgentRole = AgentRole.GENERAL,
    ) -> bool:
        """Audit a proposed tool call before execution.

        Args:
            tool_name: Name of the tool being called.
            kwargs: Arguments passed to the tool.
            role: Agent role initiating the tool call.

        Returns:
            True if allowed.

        Raises:
            PolicyViolationError: If action violates security policy.
        """
        # If terminal/command tool, audit command string
        if tool_name in ("run_command", "terminal", "bash", "cmd"):
            cmd = str(kwargs.get("command") or kwargs.get("cmd") or "")
            if cmd:
                self.audit_shell_command(cmd, role=role)

        # If filesystem tool, audit target path
        if tool_name in ("write_to_file", "view_file", "replace_file_content", "read_file"):
            target = str(kwargs.get("TargetFile") or kwargs.get("AbsolutePath") or kwargs.get("path") or "")
            if target:
                self.audit_path_access(target, role=role)

        record = AuditRecord(
            agent_role=role,
            action_type="tool_call",
            target_resource=tool_name,
            allowed=True,
            risk_level=RiskLevel.LOW,
            details=f"Tool call '{tool_name}' passed audit",
        )
        self._audit_trail.append(record)
        return True

    def audit_shell_command(self, command: str, role: AgentRole = AgentRole.GENERAL) -> bool:
        """Audit shell command for destructive patterns.

        Raises:
            PolicyViolationError: If command matches a blocked destructive pattern.
        """
        if not self.policy.allow_terminal_execution:
            record = AuditRecord(
                agent_role=role,
                action_type="shell_exec",
                target_resource=command[:100],
                allowed=False,
                risk_level=RiskLevel.CRITICAL,
                details="Terminal execution disabled by policy",
            )
            self._audit_trail.append(record)
            raise PolicyViolationError("Terminal command execution is disabled by security policy.")

        for pattern in self.policy.blocked_command_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                record = AuditRecord(
                    agent_role=role,
                    action_type="shell_exec",
                    target_resource=command[:100],
                    allowed=False,
                    risk_level=RiskLevel.CRITICAL,
                    details=f"Command matched blocked pattern '{pattern}'",
                )
                self._audit_trail.append(record)
                logger.error("[SECURITY VIOLATION] Blocked command matching pattern '%s': %s", pattern, command)
                raise PolicyViolationError(
                    f"Command blocked by LocLM Security Guard: matches dangerous pattern '{pattern}'"
                )

        record = AuditRecord(
            agent_role=role,
            action_type="shell_exec",
            target_resource=command[:100],
            allowed=True,
            risk_level=RiskLevel.MEDIUM,
            details="Shell command passed security policy",
        )
        self._audit_trail.append(record)
        return True

    def audit_path_access(self, path_str: str, role: AgentRole = AgentRole.GENERAL) -> bool:
        """Audit file path access against forbidden path prefixes.

        Raises:
            PolicyViolationError: If path accesses restricted system locations.
        """
        normalized = str(Path(path_str).resolve())

        for pattern in self.policy.forbidden_path_prefixes:
            if re.search(pattern, normalized, re.IGNORECASE) or re.search(pattern, path_str, re.IGNORECASE):
                record = AuditRecord(
                    agent_role=role,
                    action_type="file_access",
                    target_resource=path_str,
                    allowed=False,
                    risk_level=RiskLevel.HIGH,
                    details=f"Path matched forbidden prefix pattern '{pattern}'",
                )
                self._audit_trail.append(record)
                logger.error("[SECURITY VIOLATION] Blocked path access to '%s'", path_str)
                raise PolicyViolationError(
                    f"Path access blocked by LocLM Security Guard: restricted system location '{path_str}'"
                )

        record = AuditRecord(
            agent_role=role,
            action_type="file_access",
            target_resource=path_str,
            allowed=True,
            risk_level=RiskLevel.LOW,
            details="Path access permitted",
        )
        self._audit_trail.append(record)
        return True

    def get_audit_trail(self) -> list[AuditRecord]:
        """Return the immutable audit log list recorded during execution."""
        return list(self._audit_trail)
