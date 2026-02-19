"""Response-relevance trace evaluator.

Reference-free check: does the agent's response actually address what
the user asked, without needing a ground-truth expected answer?

Langfuse score name: ``response_relevance``  (0.0 – 1.0)
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
    get_question,
    score_to_float,
)

logger = logging.getLogger(__name__)

_CFG = ExperimentConfig()

_SYSTEM_PROMPT = """\
You are an expert evaluator assessing whether an AI assistant's response is relevant to the user's question.

This is a REFERENCE-FREE evaluation — you do not have access to a ground-truth answer.
Focus only on whether the response addresses the question as asked.

Evaluate step by step:
1. Does the response directly address the user's question?
2. Does it provide the specific analysis, data, or explanation requested?
3. Is the response on-topic (no off-topic rambling or refusals without reason)?
4. Is it helpful to someone who asked this question?

Score from 1–5:
5 = Directly and fully addresses the question with useful content
4 = Mostly relevant, minor tangents or missing sub-parts
3 = Somewhat relevant, addresses part of the question
2 = Mostly irrelevant or a confused response
1 = Completely off-topic, refuses without reason, or empty

Respond with valid JSON only (no markdown fences):
{{"reasoning": "step-by-step analysis", "score": <1-5>}}
"""


def _make_user_prompt(question: str, answer: str) -> str:
    return f"USER QUESTION:\n{question}\n\nAI RESPONSE:\n{answer}"


async def response_relevance_evaluator(
    *,
    trace: TraceWithFullDetails,
    item_result: ExperimentItemResult,
    **kwargs: Any,
) -> list[Evaluation]:
    """Assess whether the agent's response is relevant to the question.

    This is reference-free — no expected output is needed.
    Returns ``[]`` if no answer can be extracted from the trace.

    Parameters
    ----------
    trace : TraceWithFullDetails
        Full trace with output.
    item_result : ExperimentItemResult
        Experiment item result (for question extraction).

    Returns
    -------
    list[Evaluation]
        ``[Evaluation(name="response_relevance", value=0.0–1.0)]`` or ``[]``.
    """
    question = get_question(trace, item_result)
    answer = get_actual_answer(trace, item_result)

    if not answer:
        logger.warning("response_relevance_evaluator: no answer found for trace %s — skipping", trace.id)
        return []

    if not question:
        logger.warning("response_relevance_evaluator: no question found for trace %s — skipping", trace.id)
        return []

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
            user_prompt=_make_user_prompt(question, answer),
            model_config=model_config,
        )
    except Exception as exc:
        logger.error("response_relevance_evaluator: judge failed for trace %s: %s", trace.id, exc)
        return []

    score = score_to_float(response.score)
    logger.debug("response_relevance: trace=%s score=%d (%.2f)", trace.id, response.score, score)

    return [
        Evaluation(
            name="response_relevance",
            value=score,
            comment=f"[{response.score}/5] {response.reasoning}",
        )
    ]
