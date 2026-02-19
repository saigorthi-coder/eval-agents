"""Experiment configuration — loaded from environment variables.

Copy ../.env.example to ../.env and fill in values.
Override any setting with the corresponding env var or CLI flag.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env from project root (one level above this package)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"), verbose=False)
load_dotenv(verbose=False)


@dataclass
class ExperimentConfig:
    """All tunable knobs for an experiment run."""

    # -- Dataset & experiment identity --
    dataset_name: str = field(default_factory=lambda: os.getenv("LANGFUSE_DATASET_NAME", "OnlineRetailReportEval"))
    experiment_name: str = field(default_factory=lambda: os.getenv("EXPERIMENT_NAME", "report-generation-eval"))
    langfuse_project_name: str = field(
        default_factory=lambda: os.getenv("REPORT_GENERATION_LANGFUSE_PROJECT_NAME", "Report Generation")
    )
    reports_output_path: str = field(
        default_factory=lambda: os.getenv(
            "REPORT_GENERATION_OUTPUT_PATH", "implementations/report_generation/reports/"
        )
    )

    # -- Judge model (must be available via OPENAI_BASE_URL endpoint) --
    # Swap this one line to change the judge without touching any evaluator.
    judge_model: str = field(default_factory=lambda: os.getenv("DEFAULT_EVALUATOR_MODEL", "gemini-2.5-pro"))
    judge_temperature: float = 0.0
    judge_max_tokens: int = 2048
    judge_retry_max_attempts: int = 3
    judge_retry_initial_wait_sec: float = 2.0
    judge_retry_max_wait_sec: float = 30.0
    judge_retry_backoff_multiplier: float = 2.0

    # -- Pass/fail thresholds (scores normalized to 0–1) --
    score_thresholds: dict = field(
        default_factory=lambda: {
            "groundedness": 0.6,
            "tool_quality": 0.6,
            "answer_correctness": 0.6,
            "sql_quality": 0.6,
            "response_relevance": 0.6,
        }
    )

    # -- Metric weights for overall health score --
    metric_weights: dict = field(
        default_factory=lambda: {
            "groundedness": 0.25,
            "answer_correctness": 0.30,
            "response_relevance": 0.20,
            "sql_quality": 0.15,
            "tool_quality": 0.10,
        }
    )

    # -- Execution parameters --
    max_items: int | None = None         # None = all items
    max_concurrency: int = 1             # Task concurrency (Pass 1)
    trace_max_concurrency: int = 5       # Trace eval concurrency (Pass 2)
