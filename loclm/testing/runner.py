"""Automated Test Runner & Regression Sandbox for LocLM V6.

Executes local pytest suites and test runners, verifying that refactoring
operations produce zero regression failures.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TestRunResult(BaseModel):
    """Structured result of a test suite execution."""

    success: bool = Field(description="True if all tests passed")
    total_tests: int = Field(default=0, description="Total tests run")
    passed_tests: int = Field(default=0, description="Number of passing tests")
    failed_tests: int = Field(default=0, description="Number of failing tests")
    output: str = Field(default="", description="Console stdout/stderr output")
    error: str = Field(default="", description="Error description if execution failed")


class RegressionTestRunner:
    """Runs test suites in isolated workspace context to verify zero regression."""

    async def run_tests(
        self,
        cwd: str | Path = ".",
        test_command: str = "pytest",
    ) -> TestRunResult:
        """Run tests using specified command.

        Args:
            cwd: Working directory root for test execution.
            test_command: CLI command (default: pytest).

        Returns:
            TestRunResult object detailing test outcomes.
        """
        cwd_path = Path(cwd).resolve()
        logger.info("RegressionTestRunner running tests in %s via '%s'", cwd_path, test_command)

        try:
            proc = await asyncio.create_subprocess_shell(
                test_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd_path),
            )
            stdout, stderr = await proc.communicate()
            out_str = (stdout.decode("utf-8", errors="replace") + "\n" + stderr.decode("utf-8", errors="replace")).strip()

            is_success = (proc.returncode == 0)

            return TestRunResult(
                success=is_success,
                output=out_str,
                error="" if is_success else f"Test run exited with non-zero exit code ({proc.returncode})",
            )

        except Exception as e:
            logger.error("Failed to execute test runner command: %s", e)
            return TestRunResult(
                success=False,
                output="",
                error=f"Test runner execution error: {e}",
            )
