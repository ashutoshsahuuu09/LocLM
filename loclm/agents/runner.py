"""Agent execution loop runner for LocLM (V3).

Manages the multi-step loop:
Prompt -> Model -> Parse Tool Call -> Execute Tool -> Feedback -> Loop -> Synthesis
"""

from __future__ import annotations

import logging
import time
from typing import Any

from loclm.agents.base import AgentContext, AgentResponse, ExecutionStep
from loclm.agents.tool_parser import parse_tool_call
from loclm.models.base import ChatMessage, TaskType
from loclm.models.manager import ModelManager
from loclm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_AGENT_PROMPT = """You are LocLM, a specialized autonomous AI Coding Agent running fully offline on the user's computer.

You have access to the following local system tools:
{tools_schema}

HOW TO CALL A TOOL:
If you need information or need to take an action, output a tool call using this exact XML format:

<tool_call>
{{"name": "<tool_name>", "arguments": {{"<param>": "<value>"}}}}
</tool_call>

RULES:
1. Only call one tool at a time.
2. After receiving tool results, analyze them and decide your next step.
3. When you have completed the task, provide your final detailed answer WITHOUT calling any further tools.
"""


class AgentLoop:
    """Multi-step agent execution loop controller."""

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 6,
    ) -> None:
        self._model_manager = model_manager
        self._tool_registry = tool_registry or ToolRegistry()
        self._max_steps = max_steps

    async def execute_task(
        self,
        context: AgentContext,
        task_type: TaskType = TaskType.CODING,
    ) -> AgentResponse:
        """Execute task through multi-step reasoning and tool calling.

        Args:
            context: AgentContext containing task and environment info.
            task_type: Specialized task type for model selection.

        Returns:
            AgentResponse containing final output and step trace.
        """
        start_time = time.time()
        steps: list[ExecutionStep] = []

        # Prepare system prompt with registered tools schema
        schemas = self._tool_registry.export_tools_schema()
        schema_text = "\n".join(
            f"- {s['function']['name']}: {s['function']['description']}" for s in schemas
        )
        system_prompt = SYSTEM_AGENT_PROMPT.format(tools_schema=schema_text)

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=context.task),
        ]

        logger.info("Starting AgentLoop for task: '%s'", context.task[:60])

        for step_num in range(1, self._max_steps + 1):
            logger.debug("AgentLoop Step %d/%d", step_num, self._max_steps)

            try:
                # Query LLM model
                response_text = await self._model_manager.chat(
                    messages=messages,
                    task_type=task_type,
                    temperature=0.2,  # Low temperature for deterministic tool calling
                    max_tokens=1024,
                )
            except Exception as e:
                logger.error("LLM generation error at step %d: %s", step_num, e)
                return AgentResponse(
                    content=f"Agent execution encountered an error: {e}",
                    execution_steps=steps,
                    is_complete=False,
                )

            # Check if model requested a tool call
            tool_call = parse_tool_call(response_text)

            if tool_call:
                logger.info(
                    "Step %d: Agent requested tool '%s' with args %s",
                    step_num,
                    tool_call.tool_name,
                    tool_call.arguments,
                )

                # Execute requested tool
                tool_res = await self._tool_registry.execute_tool(
                    tool_call.tool_name, **tool_call.arguments
                )

                step = ExecutionStep(
                    step_number=step_num,
                    thought=response_text,
                    tool_call=tool_call,
                    tool_result=tool_res,
                )
                steps.append(step)

                # Append tool call and result to context history
                messages.append(ChatMessage(role="assistant", content=response_text))
                tool_msg_content = (
                    f"TOOL RESULT for '{tool_call.tool_name}':\n"
                    f"Success: {tool_res.success}\n"
                    f"Output:\n{tool_res.output}\n"
                    f"Error: {tool_res.error or 'None'}"
                )
                messages.append(ChatMessage(role="user", content=tool_msg_content))

            else:
                # No tool call requested -- final answer reached!
                logger.info("Step %d: Agent reached final answer", step_num)
                step = ExecutionStep(
                    step_number=step_num,
                    thought=response_text,
                )
                steps.append(step)

                return AgentResponse(
                    content=response_text,
                    execution_steps=steps,
                    is_complete=True,
                )

        # Reached max steps limit
        logger.warning("AgentLoop reached maximum step limit (%d)", self._max_steps)
        last_thought = steps[-1].thought if steps else "Max steps reached."
        return AgentResponse(
            content=f"Task reached maximum execution steps ({self._max_steps}).\n\nLatest Output:\n{last_thought}",
            execution_steps=steps,
            is_complete=False,
        )
