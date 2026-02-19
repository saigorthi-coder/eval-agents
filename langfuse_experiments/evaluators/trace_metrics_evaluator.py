"""Deterministic trace-metrics evaluator.

Extracts ``TraceMetrics`` from the full trace and emits a Langfuse score
for each field.  No LLM judge is involved — this is always deterministic.

Langfuse score names written
----------------------------
- ``latency_sec``       — end-to-end trace latency in seconds
- ``tool_call_count``   — number of tool-call observations
- ``turn_count``        — number of model-generation observations
- ``total_tokens``      — total_input_tokens + total_output_tokens
- ``total_cost_usd``    — total trace cost in USD (if available)
- ``observation_count`` — total observations in the trace
"""

import logging
from typing import Any

from aieng.agent_evals.evaluation.trace import extract_trace_metrics
from aieng.agent_evals.evaluation.types import Evaluation
from langfuse.api import ScoreDataType
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails
from langfuse.experiment import ExperimentItemResult

logger = logging.getLogger(__name__)


async def trace_metrics_evaluator(
    *,
    trace: TraceWithFullDetails,
    item_result: ExperimentItemResult,
    **kwargs: Any,
) -> list[Evaluation]:
    """Emit one numeric Langfuse score per TraceMetrics field.

    This evaluator always runs and never calls the LLM judge.

    Parameters
    ----------
    trace : TraceWithFullDetails
        Fully populated trace from Langfuse (Pass 2).
    item_result : ExperimentItemResult
        Experiment item result (not used here, present to satisfy protocol).

    Returns
    -------
    list[Evaluation]
        One ``Evaluation`` per metric; fields absent from the trace are skipped.
    """
    metrics = extract_trace_metrics(trace)
    evaluations: list[Evaluation] = []

    if metrics.latency_sec is not None:
        evaluations.append(
            Evaluation(
                name="latency_sec",
                value=round(metrics.latency_sec, 3),
                data_type=ScoreDataType.NUMERIC,
                comment=f"End-to-end trace latency: {metrics.latency_sec:.2f}s",
            )
        )

    evaluations.append(
        Evaluation(
            name="tool_call_count",
            value=metrics.tool_call_count,
            data_type=ScoreDataType.NUMERIC,
            comment=f"Number of tool-call observations: {metrics.tool_call_count}",
        )
    )

    evaluations.append(
        Evaluation(
            name="turn_count",
            value=metrics.turn_count,
            data_type=ScoreDataType.NUMERIC,
            comment=f"Number of model-generation turns: {metrics.turn_count}",
        )
    )

    total_tokens = metrics.total_input_tokens + metrics.total_output_tokens
    evaluations.append(
        Evaluation(
            name="total_tokens",
            value=total_tokens,
            data_type=ScoreDataType.NUMERIC,
            comment=(
                f"Total tokens: {total_tokens} "
                f"(input={metrics.total_input_tokens}, output={metrics.total_output_tokens})"
            ),
        )
    )

    evaluations.append(
        Evaluation(
            name="observation_count",
            value=metrics.observation_count,
            data_type=ScoreDataType.NUMERIC,
            comment=f"Total observations in trace: {metrics.observation_count}",
        )
    )

    if metrics.total_cost is not None:
        evaluations.append(
            Evaluation(
                name="total_cost_usd",
                value=round(metrics.total_cost, 6),
                data_type=ScoreDataType.NUMERIC,
                comment=f"Estimated cost: ${metrics.total_cost:.6f} USD",
            )
        )

    logger.debug(
        "trace_metrics_evaluator: trace_id=%s latency=%.2fs tools=%d tokens=%d",
        trace.id,
        metrics.latency_sec or 0.0,
        metrics.tool_call_count,
        total_tokens,
    )

    return evaluations
