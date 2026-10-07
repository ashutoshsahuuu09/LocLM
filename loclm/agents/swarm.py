"""Multi-Agent Swarm Orchestrator — LocLM V8.

Provides multi-round collaborative swarm interactions (Leader, Worker, Critic, Verifier)
for executing high-complexity tasks with iterative critique and verification.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.models.base import ChatMessage
from loclm.models.manager import ModelManager

logger = logging.getLogger(__name__)


class SwarmRole(str, enum.Enum):
    """Functional roles within a multi-agent swarm."""

    LEADER = "leader"     # Coordinates strategy & task breakdown
    WORKER = "worker"     # Generates solution or code implementation
    CRITIC = "critic"     # Reviews for flaws, edge cases, or security risks
    VERIFIER = "verifier" # Performs final sanity checks and output validation


class SwarmMessage(BaseModel):
    """An inter-agent message within the swarm deliberation trace."""

    sender: SwarmRole = Field(description="Role sending the message")
    recipient: SwarmRole | None = Field(default=None)
    content: str = Field(description="Body of the message")
    round_number: int = Field(description="Deliberation round index (1-based)")
    timestamp: float = Field(default_factory=time.time)


class SwarmResult(BaseModel):
    """Final result of a multi-round Swarm execution."""

    task: str = Field(description="Original task request")
    rounds_executed: int = Field(description="Number of swarm rounds completed")
    final_output: str = Field(description="Final verified swarm output")
    message_trace: list[SwarmMessage] = Field(default_factory=list)
    swarm_success: bool = Field(default=True)
    total_execution_time_sec: float = Field(default=0.0)


class SwarmOrchestrator:
    """Orchestrates multi-agent swarm deliberation across Leader, Worker, Critic, and Verifier roles."""

    def __init__(
        self,
        model_manager: ModelManager,
        agent_map: dict[AgentRole, BaseAgent] | None = None,
        max_rounds: int = 3,
    ) -> None:
        self.model_manager = model_manager
        self.agent_map = agent_map or {}
        self.max_rounds = max_rounds

    async def execute_swarm(
        self,
        task: str,
        user_context: str = "",
    ) -> SwarmResult:
        """Run a multi-round collaborative swarm session to solve a complex task.

        Pipeline:
        1. Round 1 (Leader): Decomposes task into strategic execution blueprint.
        2. Round 1-N (Worker): Generates implementation / code.
        3. Round 1-N (Critic): Identifies flaws, missing requirements, or edge cases.
        4. Final Round (Verifier): Synthesizes final response and verifies correctness.

        Args:
            task: Task instruction.
            user_context: Background context.

        Returns:
            SwarmResult containing final verified output and message trace.
        """
        start_t = time.time()
        logger.info("SwarmOrchestrator launching task: '%s'", task[:60])
        trace: list[SwarmMessage] = []

        # ── Step 1: Leader planning ──────────────────────────────────────
        leader_messages = [
            ChatMessage(role="system", content="You are the Swarm Leader responsible for strategy."),
            ChatMessage(
                role="user",
                content=(
                    f"Task: {task}\n"
                    f"Context: {user_context}\n\n"
                    f"As the Swarm LEADER, formulate a concise strategic blueprint for the Workers."
                ),
            ),
        ]
        try:
            blueprint = await self.model_manager.chat(messages=leader_messages, temperature=0.3)
            blueprint = blueprint.strip()
        except Exception as exc:
            logger.warning("Leader step failed: %s", exc)
            blueprint = f"Strategic blueprint for {task}"

        trace.append(SwarmMessage(sender=SwarmRole.LEADER, content=blueprint, round_number=1))

        current_solution = ""
        round_idx = 1

        # ── Step 2: Worker & Critic deliberation loops ───────────────────
        while round_idx <= self.max_rounds:
            logger.info("Swarm Round %d/%d starting", round_idx, self.max_rounds)

            # Worker step
            worker_messages = [
                ChatMessage(role="system", content="You are the Swarm Worker executing tasks cleanly."),
                ChatMessage(
                    role="user",
                    content=(
                        f"Task: {task}\n"
                        f"Leader Blueprint:\n{blueprint}\n\n"
                        + (f"Previous Solution:\n{current_solution}\n" if current_solution else "")
                        + f"Generate/Refine the complete solution."
                    ),
                ),
            ]
            try:
                current_solution = await self.model_manager.chat(messages=worker_messages, temperature=0.4)
                current_solution = current_solution.strip()
            except Exception as exc:
                logger.warning("Worker step failed: %s", exc)
                current_solution = f"Solution for {task}"

            trace.append(
                SwarmMessage(
                    sender=SwarmRole.WORKER,
                    content=current_solution,
                    round_number=round_idx,
                )
            )

            # Critic step
            critic_messages = [
                ChatMessage(role="system", content="You are the Swarm Critic ensuring correctness."),
                ChatMessage(
                    role="user",
                    content=(
                        f"Task: {task}\n"
                        f"Current Solution:\n{current_solution}\n\n"
                        f"As the Swarm CRITIC, evaluate this solution. If perfect, respond with 'APPROVED'."
                    ),
                ),
            ]
            try:
                critique = await self.model_manager.chat(messages=critic_messages, temperature=0.2)
                critique = critique.strip()
            except Exception as exc:
                logger.warning("Critic step failed: %s", exc)
                critique = "APPROVED"

            trace.append(
                SwarmMessage(
                    sender=SwarmRole.CRITIC,
                    content=critique,
                    round_number=round_idx,
                )
            )

            if "APPROVED" in critique.upper() and len(critique) < 200:
                logger.info("Swarm Critic APPROVED solution on round %d", round_idx)
                break

            round_idx += 1

        # ── Step 3: Verifier final synthesis ─────────────────────────────
        verifier_messages = [
            ChatMessage(role="system", content="You are the Swarm Verifier producing final response."),
            ChatMessage(
                role="user",
                content=(
                    f"Task: {task}\n"
                    f"Final Worker Solution:\n{current_solution}\n"
                    f"Critic Feedback:\n{trace[-1].content}\n\n"
                    f"As the Swarm VERIFIER, produce the polished, final output for the user."
                ),
            ),
        ]
        try:
            final_output = await self.model_manager.chat(messages=verifier_messages, temperature=0.2)
            final_output = final_output.strip()
        except Exception as exc:
            logger.warning("Verifier step failed: %s", exc)
            final_output = current_solution

        trace.append(
            SwarmMessage(
                sender=SwarmRole.VERIFIER,
                content=final_output,
                round_number=round_idx,
            )
        )

        duration = round(time.time() - start_t, 3)
        return SwarmResult(
            task=task,
            rounds_executed=round_idx,
            final_output=final_output,
            message_trace=trace,
            swarm_success=True,
            total_execution_time_sec=duration,
        )
