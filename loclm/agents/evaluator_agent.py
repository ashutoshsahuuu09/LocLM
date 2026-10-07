"""Evaluator Agent for LocLM V7.

After the primary agent produces a response, the EvaluatorAgent performs
a lightweight quality assessment pass using the LLM to score and flag
the output on four dimensions:

  - **correctness**   : Is the answer factually / logically correct?
  - **completeness**  : Does it fully address what was asked?
  - **safety**        : Does it avoid harmful, dangerous, or policy-violating content?
  - **relevance**     : Is the response on-topic and well-focused?

Each dimension is scored 0.0 – 1.0 and an overall ``quality_score`` is
produced as their weighted mean. If quality is below the configured
``min_quality_threshold``, the evaluator sets ``needs_retry=True`` and
supplies a structured ``improvement_hint`` that the V7Orchestrator can use
to trigger a corrective re-run.

The evaluator uses the **smallest** available model (TaskType.ROUTER) to
minimise latency and resource overhead, since evaluation prompts are short.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.models.base import ChatMessage, TaskType
from loclm.models.manager import ModelManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation data models
# ---------------------------------------------------------------------------


class EvaluationDimension(BaseModel):
    """Score and rationale for a single quality dimension."""

    score: float = Field(ge=0.0, le=1.0, description="0.0 (poor) to 1.0 (excellent)")
    rationale: str = Field(description="One-sentence explanation of the score")


class EvaluationResult(BaseModel):
    """Structured output of a single EvaluatorAgent evaluation pass."""

    task: str = Field(description="Original user task evaluated")
    correctness: EvaluationDimension
    completeness: EvaluationDimension
    safety: EvaluationDimension
    relevance: EvaluationDimension

    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted overall quality score (0.0–1.0)",
    )
    needs_retry: bool = Field(
        default=False,
        description="True if quality is below acceptable threshold",
    )
    improvement_hint: str = Field(
        default="",
        description="Actionable improvement suggestion for retry",
    )

    @classmethod
    def build(
        cls,
        task: str,
        correctness: EvaluationDimension,
        completeness: EvaluationDimension,
        safety: EvaluationDimension,
        relevance: EvaluationDimension,
        min_quality_threshold: float = 0.55,
    ) -> "EvaluationResult":
        """Compute weighted quality_score and set needs_retry flag."""
        quality = round(
            0.35 * correctness.score
            + 0.30 * completeness.score
            + 0.20 * safety.score
            + 0.15 * relevance.score,
            4,
        )
        needs_retry = quality < min_quality_threshold
        hint = ""
        if needs_retry:
            weak_dims = [
                (name, dim)
                for name, dim in [
                    ("correctness", correctness),
                    ("completeness", completeness),
                    ("safety", safety),
                    ("relevance", relevance),
                ]
                if dim.score < 0.5
            ]
            if weak_dims:
                hint = "Improve: " + "; ".join(f"{n} ({d.rationale})" for n, d in weak_dims)
            else:
                hint = "Overall quality below threshold — please refine the response."

        return cls(
            task=task,
            correctness=correctness,
            completeness=completeness,
            safety=safety,
            relevance=relevance,
            quality_score=quality,
            needs_retry=needs_retry,
            improvement_hint=hint,
        )


# ---------------------------------------------------------------------------
# Evaluator prompt
# ---------------------------------------------------------------------------

_EVALUATOR_SYSTEM_PROMPT = """\
You are a strict quality evaluator for AI assistant responses.
Given a user task and an AI response, rate the response on four dimensions.

Respond ONLY with a valid JSON object — no markdown fencing, no extra text:
{
  "correctness": {"score": <0.0-1.0>, "rationale": "<one sentence>"},
  "completeness": {"score": <0.0-1.0>, "rationale": "<one sentence>"},
  "safety": {"score": <0.0-1.0>, "rationale": "<one sentence>"},
  "relevance": {"score": <0.0-1.0>, "rationale": "<one sentence>"}
}

Scoring guide:
  1.0 = perfect  |  0.75 = good  |  0.5 = acceptable  |  0.25 = poor  |  0.0 = completely wrong/unsafe
"""


# ---------------------------------------------------------------------------
# EvaluatorAgent
# ---------------------------------------------------------------------------


class EvaluatorAgent(BaseAgent):
    """Post-execution quality evaluator agent for V7.

    This agent does NOT call any tools; it sends a structured evaluation
    prompt to the LLM and parses the four-dimensional quality scores.

    Args:
        model_manager: Active ModelManager instance.
        min_quality_threshold: Quality score below which ``needs_retry`` is set.
            Default 0.55 is intentionally lenient to avoid excessive re-tries.
    """

    name = "evaluator_agent"
    role = AgentRole.GENERAL  # evaluator is not routed directly
    description = "Post-execution output quality evaluator producing structured dimension scores."

    def __init__(
        self,
        model_manager: ModelManager,
        min_quality_threshold: float = 0.55,
    ) -> None:
        self._model_manager = model_manager
        self._min_quality_threshold = min_quality_threshold

    async def run(self, context: AgentContext) -> AgentResponse:
        """Evaluate the response stored in ``context.metadata["response"]``.

        The caller (V7Orchestrator) must populate:
          ``context.metadata["response"]`` — the agent output text to evaluate.

        Returns:
            AgentResponse whose ``content`` is the JSON-serialised EvaluationResult.
        """
        response_text: str = context.metadata.get("response", "")
        if not response_text:
            logger.warning("EvaluatorAgent: no response provided in metadata, skipping evaluation")
            return AgentResponse(content="{}", is_complete=False)

        evaluation = await self.evaluate(task=context.task, response=response_text)
        return AgentResponse(
            content=evaluation.model_dump_json(indent=2),
            is_complete=True,
        )

    async def evaluate(self, task: str, response: str) -> EvaluationResult:
        """Run evaluation against a task + response pair.

        Args:
            task: Original user task description.
            response: Agent response text to evaluate.

        Returns:
            EvaluationResult with dimension scores and retry flag.
        """
        user_content = (
            f"USER TASK:\n{task}\n\n"
            f"AI RESPONSE:\n{response[:1500]}"  # cap to avoid context overflow
        )
        messages = [
            ChatMessage(role="system", content=_EVALUATOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ]

        try:
            raw = await self._model_manager.chat(
                messages=messages,
                task_type=TaskType.ROUTER,  # lightweight model
                temperature=0.0,
                max_tokens=300,
            )
            result = self._parse_evaluation(task=task, raw=raw)
        except Exception as exc:
            logger.warning("EvaluatorAgent model call failed: %s — returning default scores", exc)
            result = self._default_result(task)

        logger.info(
            "EvaluatorAgent scored task=%.40r | quality=%.2f | needs_retry=%s",
            task,
            result.quality_score,
            result.needs_retry,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_evaluation(self, task: str, raw: str) -> EvaluationResult:
        """Parse LLM JSON output into an EvaluationResult."""
        # Extract JSON from the model output (handles trailing text / markdown)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logger.debug("EvaluatorAgent: no JSON found in model output, using defaults")
            return self._default_result(task)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.debug("EvaluatorAgent: JSON parse error, using defaults")
            return self._default_result(task)

        def _dim(key: str) -> EvaluationDimension:
            d = data.get(key, {})
            return EvaluationDimension(
                score=float(d.get("score", 0.5)),
                rationale=str(d.get("rationale", "No rationale provided")),
            )

        return EvaluationResult.build(
            task=task,
            correctness=_dim("correctness"),
            completeness=_dim("completeness"),
            safety=_dim("safety"),
            relevance=_dim("relevance"),
            min_quality_threshold=self._min_quality_threshold,
        )

    def _default_result(self, task: str) -> EvaluationResult:
        """Return a neutral default evaluation when parsing fails."""
        neutral = EvaluationDimension(score=0.5, rationale="Evaluation unavailable")
        return EvaluationResult.build(
            task=task,
            correctness=neutral,
            completeness=neutral,
            safety=neutral,
            relevance=neutral,
            min_quality_threshold=self._min_quality_threshold,
        )
