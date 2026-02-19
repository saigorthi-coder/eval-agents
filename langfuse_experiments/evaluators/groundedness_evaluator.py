"""Groundedness trace evaluator.

Checks whether the agent's final answer is factually grounded in the data
it actually queried (SQL results), rather than hallucinated.

Langfuse score name: ``groundedness``  (0.0 – 1.0)
Returns empty list if no SQL results are found in the trace (cannot judge).
"""

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
    extract_sql_from_trace,
    get_actual_answer,
    get_question,
    score_to_float,
)

logger = logging.getLogger(__name__)

_CFG = ExperimentConfig()

_SYSTEM_PROMPT = """\
You are an expert evaluator assessing whether an AI assistant's answer is grounded in the data it retrieved.

Evaluate step by step:
1. Identify all factual claims in the AI's answer.
2. Check each claim against the retrieved data.
3. Note any claims not supported by the data.

Score the groundedness from 1–5:
5 = All claims directly supported by retrieved data
4 = Minor unsupported details, core claims grounded
3 = Mixed — some claims grounded, some not
2 = Mostly ungrounded, few claims supported
1 = Answer contradicts or ignores retrieved data

Respond with valid JSON only (no markdown fences):
{{"reasoning": "step-by-step analysis", "score": <1-5>}}
"""


def _make_user_prompt(question: str, retrieved_data: str, answer: str) -> str:
    return (
        f"USER QUESTION:\n{question}\n\n"
        f"DATA RETRIEVED (SQL query results):\n{retrieved_data}\n\n"
        f"AI ASSISTANT'S ANSWER:\n{answer}"
    )


async def groundedness_evaluator(
    *,
    trace: TraceWithFullDetails,
    item_result: ExperimentItemResult,
    **kwargs: Any,
) -> list[Evaluation]:
    """Assess whether the agent's answer is grounded in its SQL query results.

    Skips (returns ``[]``) when no SQL data is found in the trace — we cannot
    meaningfully evaluate groundedness without the retrieved data.

    Parameters
    ----------
    trace : TraceWithFullDetails
        Fully populated trace including observations.
    item_result : ExperimentItemResult
        Experiment item result (for question extraction).

    Returns
    -------
    list[Evaluation]
        Single ``Evaluation(name="groundedness", value=0.0–1.0)`` or ``[]``.
    """
    question = get_question(trace, item_result)
    answer = get_actual_answer(trace, item_result)

    if not answer:
        logger.warning("groundedness_evaluator: no answer found for trace %s — skipping", trace.id)
        return []

    sql_observations = extract_sql_from_trace(trace)
    if not sql_observations:
        logger.info("groundedness_evaluator: no SQL observations in trace %s — skipping", trace.id)
        return []

    # Combine all SQL results into a single context block
    retrieved_data_parts = []
    for i, obs in enumerate(sql_observations, start=1):
        retrieved_data_parts.append(f"Query {i}:\n{obs['sql']}")
        if obs["result"]:
            retrieved_data_parts.append(f"Result {i}:\n{obs['result']}")
    retrieved_data = "\n\n".join(retrieved_data_parts)

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
            user_prompt=_make_user_prompt(question or "(unknown)", retrieved_data, answer),
            model_config=model_config,
        )
    except Exception as exc:
        logger.error("groundedness_evaluator: judge call failed for trace %s: %s", trace.id, exc)
        return []

    score = score_to_float(response.score)
    logger.debug("groundedness: trace=%s score=%d (%.2f) reason='%s'", trace.id, response.score, score, response.reasoning[:80])

    return [
        Evaluation(
            name="groundedness",
            value=score,
            comment=f"[{response.score}/5] {response.reasoning}",
        )
    ]
