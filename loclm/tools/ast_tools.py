"""AST-Guided Safe Refactoring Tools for LocLM V6.

Provides syntax validation, symbol inspecting, and AST code transformations.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from loclm.tools.base import BaseTool, ToolCategory, ToolParameter, ToolPermissionLevel, ToolResult
from loclm.tools.security import SecurityGuard

logger = logging.getLogger(__name__)


class ASTValidateSyntaxTool(BaseTool):
    """Tool to validate code syntax using AST parsing before writing changes."""

    name = "ast_validate_syntax"
    description = "Validate code snippet or file syntax using AST parsing without executing it."
    category = ToolCategory.FILESYSTEM
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type="string",
                description="Python code snippet to validate",
                required=True,
            ),
            ToolParameter(
                name="filename",
                type="string",
                description="Filename hint for error reporting (optional)",
                required=False,
                default="snippet.py",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code", "")
        filename = kwargs.get("filename", "snippet.py") or "snippet.py"

        if not code:
            return ToolResult(success=False, error="Parameter 'code' is required")

        try:
            tree = ast.parse(code, filename=filename)
            node_count = len(list(ast.walk(tree)))
            return ToolResult(
                success=True,
                output=f"Syntax valid! AST contains {node_count} nodes.",
                metadata={"valid": True, "node_count": node_count},
            )
        except SyntaxError as e:
            error_msg = f"SyntaxError in {filename}:{e.lineno}:{e.offset}: {e.msg}\nLine: {e.text}"
            return ToolResult(
                success=False,
                error=error_msg,
                metadata={"valid": False, "lineno": e.lineno, "offset": e.offset},
            )


class ASTRenameSymbolTool(BaseTool):
    """Tool to perform safe AST-assisted symbol renaming within a Python file."""

    name = "ast_rename_symbol"
    description = "Safely rename a function or variable symbol within a target Python file."
    category = ToolCategory.FILESYSTEM
    permission_level = ToolPermissionLevel.CONFIRM

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to target Python file",
                required=True,
            ),
            ToolParameter(
                name="old_name",
                type="string",
                description="Current symbol name to replace",
                required=True,
            ),
            ToolParameter(
                name="new_name",
                type="string",
                description="New symbol name",
                required=True,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        file_path_str = kwargs.get("path")
        old_name = kwargs.get("old_name", "").strip()
        new_name = kwargs.get("new_name", "").strip()

        if not file_path_str or not old_name or not new_name:
            return ToolResult(success=False, error="Parameters 'path', 'old_name', and 'new_name' are required.")

        target = Path(file_path_str).resolve()
        valid, reason = self._guard.validate_path(target)
        if not valid:
            return ToolResult(success=False, error=reason)

        if not target.exists() or not target.is_file():
            return ToolResult(success=False, error=f"File not found: {target}")

        try:
            content = target.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(target))

            # AST transformer to rename Name, FunctionDef, ClassDef, and Attribute nodes
            class Renamer(ast.NodeTransformer):
                def __init__(self) -> None:
                    self.replacements = 0

                def visit_Name(self, node: ast.Name) -> ast.AST:
                    if node.id == old_name:
                        node.id = new_name
                        self.replacements += 1
                    return self.generic_visit(node)

                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                    if node.name == old_name:
                        node.name = new_name
                        self.replacements += 1
                    return self.generic_visit(node)

                def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                    if node.name == old_name:
                        node.name = new_name
                        self.replacements += 1
                    return self.generic_visit(node)

            renamer = Renamer()
            new_tree = renamer.visit(tree)

            if renamer.replacements == 0:
                return ToolResult(
                    success=False,
                    error=f"Symbol '{old_name}' not found in AST of {target.name}",
                )

            ast.fix_missing_locations(new_tree)
            new_code = ast.unparse(new_tree)
            target.write_text(new_code, encoding="utf-8")

            return ToolResult(
                success=True,
                output=f"Successfully renamed symbol '{old_name}' to '{new_name}' in {target.name} ({renamer.replacements} occurrences replaced).",
                metadata={"replacements": renamer.replacements},
            )

        except Exception as e:
            return ToolResult(success=False, error=f"AST symbol rename failed: {e}")
