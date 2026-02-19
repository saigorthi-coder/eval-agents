"""Shared utilities for all trace-level evaluators.

This module provides:
- ``JudgeResponse`` — Pydantic model for structured LLM judge output.
- ``call_llm_judge`` — async helper that calls the configured judge model
  with full retry logic, using the framework's ``run_structured_parse_call``.
- ``score_to_float`` — maps a 1–5 Likert score to a [0.0, 1.0] float.
- Trace-data extraction helpers used by every trace evaluator.
"""

import json
import logging
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails
from langfuse.experiment import ExperimentItemResult
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model for the judge's structured response
# ---------------------------------------------------------------------------


class JudgeResponse(BaseModel):
    """Structured output from the LLM judge.

    The judge is instructed to reason step-by-step before scoring so that
    the ``reasoning`` field always explains the ``score``.
    """

    reasoning: str
    """Chain-of-thought explanation before the final score."""

    score: int
    """Likert score from 1 (worst) to 5 (best)."""


# ---------------------------------------------------------------------------
# Core judge call
# ---------------------------------------------------------------------------


async def call_llm_judge(
    *,
    system_prompt: str,
    user_prompt: str,
    model_config: LLMRequestConfig,
) -> JudgeResponse:
    """Call the LLM judge and return a structured ``JudgeResponse``.

    Uses ``run_structured_parse_call`` from the framework so retry logic,
    temperature, and structured-output parsing are all handled for us.

    Parameters
    ----------
    system_prompt : str
        System message containing the rubric and instructions.
    user_prompt : str
        User message with the content to evaluate.
    model_config : LLMRequestConfig
        Model name, temperature, retry settings.

    Returns
    -------
    JudgeResponse
        Parsed judge response with ``reasoning`` and ``score``.

    Raises
    ------
    Exception
        Re-raises after retries are exhausted (callers must handle).
    """
    client_manager = AsyncClientManager.get_instance()
    completion = await run_structured_parse_call(
        openai_client=client_manager.openai_client,
        default_model=client_manager.configs.default_evaluator_model,
        model_config=model_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=JudgeResponse,
    )
    parsed: JudgeResponse | None = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Judge returned no structured response.")
    return parsed


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


def score_to_float(score: int) -> float:
    """Map a 1–5 Likert score to a [0.0, 1.0] float.

    >>> score_to_float(1)
    0.0
    >>> score_to_float(5)
    1.0
    >>> score_to_float(3)
    0.5
    """
    clamped = max(1, min(5, score))
    return (clamped - 1) / 4.0


# ---------------------------------------------------------------------------
# Trace data extraction helpers
# ---------------------------------------------------------------------------


def get_question(trace: TraceWithFullDetails, item_result: ExperimentItemResult) -> str | None:
    """Extract the user question from available sources.

    Tries ``trace.input`` first (Langfuse stores the dataset item's input
    there), then falls back to ``item_result.input``.
    """
    if trace.input is not None:
        if isinstance(trace.input, str):
            return trace.input
        if isinstance(trace.input, dict):
            for key in ("question", "input", "query", "message"):
                if key in trace.input:
                    return str(trace.input[key])
            return json.dumps(trace.input)

    if hasattr(item_result, "input") and item_result.input is not None:
        return str(item_result.input)

    return None


def get_actual_answer(trace: TraceWithFullDetails, item_result: ExperimentItemResult) -> str | None:
    """Extract the agent's final answer from the trace output.

    Handles the health chatbot task output shape: ``{"answer": str}``.
    Falls back to other known keys, then to the raw string or JSON.
    """
    output = trace.output
    if output is None and hasattr(item_result, "output"):
        output = item_result.output

    if isinstance(output, dict):
        # Health chatbot task returns {"answer": str}
        if output.get("answer") is not None:
            return str(output["answer"])
        # Fallback: other keys used by different agent versions
        for key in ("final_report", "response", "result", "text"):
            if output.get(key) is not None:
                val = output[key]
                return json.dumps(val, ensure_ascii=False, indent=2) if isinstance(val, (dict, list)) else str(val)

    if isinstance(output, str):
        return output

    return json.dumps(output, ensure_ascii=False, default=str) if output is not None else None


def is_plot_response(trace: TraceWithFullDetails, item_result: ExperimentItemResult) -> bool:
    """Return True when the task output indicates the agent produced a plot.

    Evaluators (especially answer_correctness) should skip or use a
    relaxed rubric when the response is a chart rather than a text answer.
    """
    output = trace.output
    if output is None and hasattr(item_result, "output"):
        output = item_result.output

    if isinstance(output, dict):
        # Task function sets has_plot=True when it detects plot references
        if output.get("has_plot"):
            return True
        answer = output.get("answer", "") or ""
    else:
        answer = str(output or "")

    lower = answer.lower()
    return any(
        m in lower
        for m in (".png", ".jpg", ".svg", ".html", "![", "plot", "chart", "graph", "figure")
    )


def get_expected_answer(item_result: ExperimentItemResult) -> str | None:
    """Extract the expected answer from the dataset item result.

    Returns ``None`` if no expected output is recorded (evaluators should
    skip gracefully in that case).
    """
    expected = getattr(item_result, "expected_output", None)
    if expected is None:
        return None
    if isinstance(expected, dict):
        return json.dumps(expected, ensure_ascii=False, indent=2)
    return str(expected)


def extract_sql_from_trace(trace: TraceWithFullDetails) -> list[dict[str, Any]]:
    """Extract SQL query observations from the trace.

    Returns a list of dicts with ``{"sql": str, "result": str | None}``
    for each observation whose name or input content suggests SQL.
    """
    results = []
    for obs in trace.observations or []:
        obs_name = (obs.name or "").lower()
        # Look for the database execute tool call
        if "execute" not in obs_name and "sql" not in obs_name and "query" not in obs_name:
            continue

        sql_text: str | None = None
        if isinstance(obs.input, str) and _looks_like_sql(obs.input):
            sql_text = obs.input
        elif isinstance(obs.input, dict):
            for val in obs.input.values():
                if isinstance(val, str) and _looks_like_sql(val):
                    sql_text = val
                    break

        if sql_text:
            result_str: str | None = None
            if obs.output is not None:
                result_str = (
                    json.dumps(obs.output, ensure_ascii=False, default=str)
                    if not isinstance(obs.output, str)
                    else obs.output
                )
            results.append({"sql": sql_text, "result": result_str})

    return results


def extract_tool_calls_from_trace(trace: TraceWithFullDetails) -> list[dict[str, Any]]:
    """Extract all tool-call observations from the trace in order.

    Returns a list of dicts: ``{"name": str, "input": Any, "output": Any}``.
    Uses the same heuristic as the framework's ``_default_tool_call_predicate``.
    """
    tool_calls = []
    for obs in trace.observations or []:
        obs_type = (obs.type or "").lower()
        obs_name = (obs.name or "").lower()
        metadata = obs.metadata or {}

        is_tool = (
            "tool" in obs_type
            or "tool" in obs_name
            or any(k in metadata for k in ("tool_name", "tool", "function", "function_name"))
        )

        if is_tool:
            tool_calls.append(
                {
                    "name": obs.name or "unknown",
                    "input": obs.input,
                    "output": obs.output,
                }
            )

    return tool_calls


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _looks_like_sql(text: str) -> bool:
    """Heuristic check: does this string look like a SQL query?"""
    upper = text.strip().upper()
    return any(upper.startswith(kw) for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE"))
