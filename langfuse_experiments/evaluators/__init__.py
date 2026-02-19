"""Evaluators for the Langfuse experiment pipeline.

All trace-level evaluators follow the ``TraceEvaluatorFunction`` protocol:

    async def evaluator(
        *, trace: TraceWithFullDetails, item_result: ExperimentItemResult, **kwargs
    ) -> list[Evaluation]:
        ...

Pass them to ``run_experiment_with_trace_evals(trace_evaluators=[...])``
in the order you want them to run.
"""

from langfuse_experiments.evaluators.answer_correctness_evaluator import answer_correctness_evaluator
from langfuse_experiments.evaluators.groundedness_evaluator import groundedness_evaluator
from langfuse_experiments.evaluators.response_relevance_evaluator import response_relevance_evaluator
from langfuse_experiments.evaluators.sql_quality_evaluator import sql_quality_evaluator
from langfuse_experiments.evaluators.tool_quality_evaluator import tool_quality_evaluator
from langfuse_experiments.evaluators.trace_metrics_evaluator import trace_metrics_evaluator

__all__ = [
    "trace_metrics_evaluator",
    "groundedness_evaluator",
    "answer_correctness_evaluator",
    "sql_quality_evaluator",
    "tool_quality_evaluator",
    "response_relevance_evaluator",
]

#: Default ordered list of all trace evaluators for a full experiment run.
ALL_TRACE_EVALUATORS = [
    trace_metrics_evaluator,        # 1 — always runs, deterministic
    groundedness_evaluator,         # 2 — LLM judge, requires SQL in trace
    answer_correctness_evaluator,   # 3 — LLM judge, requires expected_output
    sql_quality_evaluator,          # 4 — LLM judge, requires SQL in trace
    tool_quality_evaluator,         # 5 — LLM judge, requires tool calls in trace
    response_relevance_evaluator,   # 6 — LLM judge, reference-free
]

#: Mapping from CLI/config name → evaluator function.
EVALUATOR_REGISTRY: dict = {
    "trace_metrics": trace_metrics_evaluator,
    "groundedness": groundedness_evaluator,
    "answer_correctness": answer_correctness_evaluator,
    "sql_quality": sql_quality_evaluator,
    "tool_quality": tool_quality_evaluator,
    "response_relevance": response_relevance_evaluator,
}
