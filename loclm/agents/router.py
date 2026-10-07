"""Router Agent for LocLM Multi-Agent Architecture (V4).

Classifies user requests to select the optimal specialized agent:
CODING, TERMINAL, KNOWLEDGE, REASONING, or GENERAL.
"""

from __future__ import annotations

import logging
from typing import Any

from loclm.agents.base import AgentRole
from loclm.models.base import ChatMessage, TaskType
from loclm.models.manager import ModelManager

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the LocLM Request Router.
Your sole job is to classify the user's input into exactly one of these 5 agent roles:

1. coding     - Programming, writing code, refactoring, fixing bugs, code review
2. terminal   - Running shell commands, git commands, system admin, terminal scripts
3. knowledge  - Searching files, reading documents, explaining codebase structure
4. reasoning  - Complex math, logic puzzles, step-by-step analytical reasoning
5. general    - Simple Q&A, greetings, general conversation, non-coding questions

OUTPUT INSTRUCTIONS:
Respond ONLY with a single JSON object:
{"role": "<coding|terminal|knowledge|reasoning|general>", "confidence": <float 0.0-1.0>, "reason": "<short explanation>"}
"""


class RouterAgent:
    """Classifies user intent and routes to specialized agent."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def route(self, user_input: str) -> AgentRole:
        """Route user input to the best AgentRole.

        Args:
            user_input: The text input from the user.

        Returns:
            The selected AgentRole.
        """
        # Rule-based fast heuristics for instant routing
        input_lower = user_input.lower().strip()

        if any(kw in input_lower for kw in ["code", "function", "def ", "class ", "python", "bug", "refactor", "import "]):
            logger.info("Fast-route rule matched: CODING")
            return AgentRole.CODING

        if any(kw in input_lower for kw in ["git ", "run ", "exec ", "terminal", "command", "pip ", "install"]):
            logger.info("Fast-route rule matched: TERMINAL")
            return AgentRole.TERMINAL

        if any(kw in input_lower for kw in ["search ", "find ", "read file", "list dir", "where is"]):
            logger.info("Fast-route rule matched: KNOWLEDGE")
            return AgentRole.KNOWLEDGE

        # Model-assisted classification for ambiguous requests
        try:
            messages = [
                ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_input),
            ]
            response = await self._model_manager.chat(
                messages=messages,
                task_type=TaskType.ROUTER,
                temperature=0.0,
                max_tokens=100,
            )

            response_clean = response.lower()
            if "coding" in response_clean:
                return AgentRole.CODING
            elif "terminal" in response_clean:
                return AgentRole.TERMINAL
            elif "knowledge" in response_clean:
                return AgentRole.KNOWLEDGE
            elif "reasoning" in response_clean:
                return AgentRole.REASONING
            else:
                return AgentRole.GENERAL

        except Exception as e:
            logger.warning("Router model classification failed: %s. Defaulting to GENERAL", e)
            return AgentRole.GENERAL
