"""Tool-quality trace evaluator.

Assesses whether the agent called the right tools, in the right order,
with correct arguments for the user's question.

Langfuse score names:
- ``tool_quality``     (0.0 – 1.0, LLM judge)
"""

import json
import logging
from typing import Any

from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails
from langfuse.experiment import ExperimentItemResult

from langfuse_experiments.config.experiment_config import ExperimentConfig
from langfuse_experiments.evaluators.base_evaluator import (
    JudgeResponse,
    call_llm_judge,
    extract_tool_calls_from_trace,
    get_question,
    score_to_float,
)

logger = logging.getLogger(__name__)

_CFG = ExperimentConfig()

_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the tool usage of a data analytics AI agent.

The agent has access to:
- ``execute`` — runs SQL queries against the database
- ``get_schema_info`` — retrieves database schema information
- ``write_xlsx`` — writes the final report as an Excel file

Evaluate step by step:
1. Were the appropriate tools selected for this question?
2. Were the tool arguments/parameters correct and well-formed?
3. Was the tool calling sequence logical (schema → query → report)?
4. Were any necessary tools skipped?

Score the tool quality from 1–5:
5 = Perfect tool selection and usage, correct sequence
4 = Correct tools, minor argument issues or redundant calls
3 = Mostly correct, one wrong or missing tool
2 = Significant tool misuse or wrong tools
1 = Completely wrong tool usage or no tools when needed

Respond with valid JSON only (no markdown fences):
{{"reasoning": "step-by-step analysis", "score": <1-5>}}
"""


def _make_user_prompt(question: str, tool_calls_json: str) -> str:
    return f"USER QUESTION:\n{question}\n\nTOOLS CALLED (in order):\n{tool_calls_json}"


async def tool_quality_evaluator(
    *,
    trace: TraceWithFullDetails,
    item_result: ExperimentItemResult,
    **kwargs: Any,
) -> list[Evaluation]:
    """Judge whether the agent's tool usage was correct and appropriate.

    Returns ``[]`` if no tool calls are found in the trace.

    Parameters
    ----------
    trace : TraceWithFullDetails
        Full trace with observations.
    item_result : ExperimentItemResult
        Experiment item result (for question extraction).

    Returns
    -------
    list[Evaluation]
        ``[Evaluation(name="tool_quality", value=0.0–1.0)]`` or ``[]``.
    """
    question = get_question(trace, item_result)
    tool_calls = extract_tool_calls_from_trace(trace)

    if not tool_calls:
        logger.info("tool_quality_evaluator: no tool calls in trace %s — skipping", trace.id)
        return []

    # Build a clean summary of tool calls for the judge (strip large outputs)
    tool_call_summary = []
    for tc in tool_calls:
        summary = {"name": tc["name"], "input": tc["input"]}
        # Truncate large outputs to avoid blowing up the prompt
        if tc["output"] is not None:
            raw = json.dumps(tc["output"], ensure_ascii=False, default=str)
            summary["output_preview"] = raw[:500] + ("…" if len(raw) > 500 else "")
        tool_call_summary.append(summary)

    tool_calls_json = json.dumps(tool_call_summary, ensure_ascii=False, indent=2)

    model_config = LLMRequestConfig(
        model=_CFG.judge_model,
        temperature=_CFG.judge_temperature,
        max_completion_tokens=_CFG.judge_max_tokens,
        retry_max_attempts=_CFG.judge_retry_max_attempts,
        retry_initial_wait_sec=_CFG.judge_retry_initial_wait_sec,
        retry_max_wait_sec=_CFG.judge_retry_max_wait_sec,
        retry_backoff_multiplier=_CFG.judge_retry_backoff_multiplier,
    )

    try:
        response: JudgeResponse = await call_llm_judge(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_make_user_prompt(question or "(unknown)", tool_calls_json),
            model_config=model_config,
        )
    except Exception as exc:
        logger.error("tool_quality_evaluator: judge call failed for trace %s: %s", trace.id, exc)
        return []

    score = score_to_float(response.score)
    logger.debug("tool_quality: trace=%s score=%d (%.2f)", trace.id, response.score, score)

    return [
        Evaluation(
            name="tool_quality",
            value=score,
            comment=f"[{response.score}/5] {response.reasoning}",
            metadata={"tool_call_count": len(tool_calls)},
        )
    ]
