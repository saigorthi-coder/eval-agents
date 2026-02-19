"""Answer-correctness trace evaluator.

Compares the agent's answer against the expected output stored in the
Langfuse dataset.  Skips (returns ``[]``) when no expected output is found.

Langfuse score name: ``answer_correctness``  (0.0 – 1.0)
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
    get_actual_answer,
    get_expected_answer,
    get_question,
    is_plot_response,
    score_to_float,
)

logger = logging.getLogger(__name__)

_CFG = ExperimentConfig()

_SYSTEM_PROMPT = """\
You are an expert evaluator assessing a data analytics AI chatbot's answer accuracy.

Evaluate step by step:
1. Are the key data points and values correct?
2. Are trends and comparisons correctly identified?
3. Is the analysis complete (covers what was asked)?
4. Are any claims factually wrong compared to the expected answer?

Score from 1–5:
5 = Fully correct and complete
4 = Mostly correct, minor omissions
3 = Partially correct, some significant gaps
2 = Mostly incorrect or incomplete
1 = Wrong answer or completely off-topic

Respond with valid JSON only (no markdown fences):
{{"reasoning": "step-by-step analysis", "score": <1-5>}}
"""


def _make_user_prompt(question: str, expected_answer: str, actual_answer: str) -> str:
    return (
        f"USER QUESTION:\n{question}\n\n"
        f"EXPECTED ANSWER:\n{expected_answer}\n\n"
        f"ACTUAL ANSWER:\n{actual_answer}"
    )


async def answer_correctness_evaluator(
    *,
    trace: TraceWithFullDetails,
    item_result: ExperimentItemResult,
    **kwargs: Any,
) -> list[Evaluation]:
    """Compare the agent's answer against the dataset's expected output.

    Returns ``[]`` (skips) if:
    - No expected output is recorded for this dataset item.
    - No actual answer can be extracted from the trace.
    - The response is primarily a plot (visual output, not a text answer).

    Parameters
    ----------
    trace : TraceWithFullDetails
        Full trace (used to extract the actual answer).
    item_result : ExperimentItemResult
        Contains ``expected_output`` from the dataset item.

    Returns
    -------
    list[Evaluation]
        ``[Evaluation(name="answer_correctness", value=0.0–1.0)]`` or ``[]``.
    """
    expected_answer = get_expected_answer(item_result)
    if not expected_answer:
        logger.info("answer_correctness_evaluator: no expected_output for trace %s — skipping", trace.id)
        return []

    # Skip when the agent returned a plot — there's no text answer to compare.
    if is_plot_response(trace, item_result):
        logger.info("answer_correctness_evaluator: plot response for trace %s — skipping text comparison", trace.id)
        return []

    actual_answer = get_actual_answer(trace, item_result)
    if not actual_answer:
        logger.warning("answer_correctness_evaluator: no actual answer found for trace %s — skipping", trace.id)
        return []

    question = get_question(trace, item_result)

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
            user_prompt=_make_user_prompt(question or "(unknown)", expected_answer, actual_answer),
            model_config=model_config,
        )
    except Exception as exc:
        logger.error("answer_correctness_evaluator: judge failed for trace %s: %s", trace.id, exc)
        return []

    score = score_to_float(response.score)
    logger.debug("answer_correctness: trace=%s score=%d (%.2f)", trace.id, response.score, score)

    return [
        Evaluation(
            name="answer_correctness",
            value=score,
            comment=f"[{response.score}/5] {response.reasoning}",
        )
    ]
