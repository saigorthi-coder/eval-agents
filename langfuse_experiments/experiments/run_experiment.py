"""Main entry point for running Langfuse evaluation experiments.

Two-pass workflow
-----------------
Pass 1 — Runs the agent against every dataset item and records trace IDs.
Pass 2 — Waits for traces to be ingested, then runs all trace evaluators.

Usage
-----
Run from the repo root (so relative paths in .env resolve correctly):

    # Full experiment — all items, all evaluators
    python -m langfuse_experiments.experiments.run_experiment

    # Quick smoke-test — first 3 items only
    python -m langfuse_experiments.experiments.run_experiment --max-items 3

    # Custom dataset & experiment name
    python -m langfuse_experiments.experiments.run_experiment \\
        --dataset-name "OnlineRetailReportEval" \\
        --experiment-name "baseline-v1"

    # Override the judge model
    python -m langfuse_experiments.experiments.run_experiment \\
        --judge-model "gemini-2.5-flash"

    # Run only specific evaluators
    python -m langfuse_experiments.experiments.run_experiment \\
        --evaluators groundedness,answer_correctness,trace_metrics

    # Dry-run: execute the agent but skip all trace evaluations
    python -m langfuse_experiments.experiments.run_experiment --dry-run
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Env setup — must happen before any framework imports read configs
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", verbose=False)
load_dotenv(verbose=False)

# PYTHONPATH is consumed by the OS at interpreter startup, so setting it in .env
# via load_dotenv() is too late for sys.path.  Manually insert any paths so that
# users can configure PYTHONPATH=<chatbot-repo-root> in their .env file.
_extra_pythonpath = os.environ.get("PYTHONPATH", "")
if _extra_pythonpath:
    for _p in _extra_pythonpath.split(os.pathsep):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

from aieng.agent_evals.async_client_manager import AsyncClientManager  # noqa: E402
from aieng.agent_evals.evaluation import run_experiment_with_trace_evals  # noqa: E402
from aieng.agent_evals.evaluation.types import TraceWaitConfig  # noqa: E402

# Health chatbot tracer — sets up OTEL → Langfuse so agent tool calls
# appear as observations in Pass 2 trace evaluations.
from src.utils import setup_langfuse_tracer  # noqa: E402

from langfuse_experiments.agent.chatbot_runner import chatbot_task  # noqa: E402
from langfuse_experiments.config.experiment_config import ExperimentConfig  # noqa: E402
from langfuse_experiments.evaluators import ALL_TRACE_EVALUATORS, EVALUATOR_REGISTRY  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a two-pass Langfuse evaluation experiment against the health chatbot agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-name",
        default=None,
        help="Langfuse dataset name (overrides LANGFUSE_DATASET_NAME env var).",
    )
    p.add_argument(
        "--experiment-name",
        default=None,
        help="Human-readable experiment name shown in the Langfuse UI.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit the number of dataset items evaluated (useful for quick tests).",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Override the judge model (e.g. 'gemini-2.5-flash').",
    )
    p.add_argument(
        "--evaluators",
        default=None,
        help=(
            "Comma-separated list of evaluators to run. "
            f"Available: {', '.join(EVALUATOR_REGISTRY)}. "
            "Default: all evaluators."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the agent (Pass 1) but skip trace evaluations (Pass 2).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrency for Pass 1 task execution (default: 1).",
    )
    p.add_argument(
        "--trace-concurrency",
        type=int,
        default=None,
        help="Max concurrency for Pass 2 trace evaluations (default: 5).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    # --- Build config, applying CLI overrides ---
    cfg = ExperimentConfig()

    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.experiment_name:
        cfg.experiment_name = args.experiment_name
    if args.max_items is not None:
        cfg.max_items = args.max_items
    if args.judge_model:
        cfg.judge_model = args.judge_model
        # Propagate to the env var read by ExperimentConfig instances in evaluators
        os.environ["DEFAULT_EVALUATOR_MODEL"] = args.judge_model
    if args.concurrency:
        cfg.max_concurrency = args.concurrency
    if args.trace_concurrency:
        cfg.trace_max_concurrency = args.trace_concurrency

    # --- Resolve evaluator list ---
    if args.evaluators:
        requested = [e.strip() for e in args.evaluators.split(",")]
        unknown = [e for e in requested if e not in EVALUATOR_REGISTRY]
        if unknown:
            logger.error("Unknown evaluators: %s. Available: %s", unknown, list(EVALUATOR_REGISTRY))
            sys.exit(1)
        trace_evaluators = [EVALUATOR_REGISTRY[e] for e in requested]
    else:
        trace_evaluators = list(ALL_TRACE_EVALUATORS)

    if args.dry_run:
        logger.info("--dry-run: trace evaluators will be skipped.")
        trace_evaluators = []

    # --- Initialise Langfuse OTEL tracer so agent tool/LLM calls appear
    #     as observations in the trace (needed by Pass 2 evaluators)  ---
    setup_langfuse_tracer()

    # --- Stamp the experiment name with a timestamp to avoid name collisions ---
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_name = f"{cfg.experiment_name}-{timestamp}"

    logger.info("=" * 60)
    logger.info("Dataset      : %s", cfg.dataset_name)
    logger.info("Experiment   : %s", run_name)
    logger.info("Judge model  : %s", cfg.judge_model)
    logger.info("Max items    : %s", cfg.max_items or "all")
    logger.info("Evaluators   : %s", [e.__name__ for e in trace_evaluators])
    logger.info("=" * 60)

    # chatbot_task already accepts (*, input, expected_output, metadata, **kwargs)
    # and builds the agent lazily on first call — no partial needed.
    task = chatbot_task

    # --- Metadata written to the Langfuse run ---
    run_metadata = {
        "judge_model": cfg.judge_model,
        "evaluators": [e.__name__ for e in trace_evaluators],
        "max_items": cfg.max_items,
        "timestamp": timestamp,
    }

    start_time = time.monotonic()

    try:
        result = run_experiment_with_trace_evals(
            cfg.dataset_name,
            name=run_name,
            task=task,
            evaluators=[],                  # No item-level evaluators (all in Pass 2)
            trace_evaluators=trace_evaluators,
            max_concurrency=cfg.max_concurrency,
            trace_max_concurrency=cfg.trace_max_concurrency,
            metadata=run_metadata,
            trace_wait=TraceWaitConfig(
                max_wait_sec=300.0,          # up to 5 min for trace ingestion
                initial_delay_sec=2.0,
                max_delay_sec=15.0,
                backoff_multiplier=2.0,
            ),
            description=f"Automated evaluation run at {timestamp}",
        )
    except Exception as exc:
        logger.exception("Experiment failed: %s", exc)
        sys.exit(1)

    elapsed = time.monotonic() - start_time

    # --- Summarise results to console ---
    _print_summary(result, elapsed, cfg)

    # --- Persist CSV report ---
    _save_csv_report(result, run_name, cfg)


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def _print_summary(result, elapsed: float, cfg: ExperimentConfig) -> None:
    """Print a human-readable summary to stdout."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # --- Per-metric averages from trace evaluations ---
        trace_evals = result.trace_evaluations
        if trace_evals:
            scores_by_metric: dict[str, list[float]] = {}
            for evals in trace_evals.evaluations_by_trace_id.values():
                for ev in evals:
                    if isinstance(ev.value, (int, float)) and not isinstance(ev.value, bool):
                        scores_by_metric.setdefault(ev.name, []).append(float(ev.value))

            table = Table(title="Evaluation Results", show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="bold")
            table.add_column("Avg", justify="right")
            table.add_column("Min", justify="right")
            table.add_column("Max", justify="right")
            table.add_column("N", justify="right")
            table.add_column("Pass?", justify="center")

            quality_scores: list[float] = []
            for metric, values in sorted(scores_by_metric.items()):
                avg = sum(values) / len(values)
                min_v = min(values)
                max_v = max(values)
                threshold = cfg.score_thresholds.get(metric)
                pass_str = ""
                if threshold is not None:
                    pass_str = "✓" if avg >= threshold else "✗"
                    # Accumulate weighted score
                    weight = cfg.metric_weights.get(metric, 0.0)
                    quality_scores.append(avg * weight)
                table.add_row(
                    metric,
                    f"{avg:.3f}",
                    f"{min_v:.3f}",
                    f"{max_v:.3f}",
                    str(len(values)),
                    pass_str,
                )

            console.print(table)

            # Overall health score (weighted average of quality metrics only)
            total_weight = sum(cfg.metric_weights.get(m, 0.0) for m in scores_by_metric if m in cfg.metric_weights)
            if total_weight > 0 and quality_scores:
                health = sum(quality_scores) / total_weight
                console.print(f"\n[bold]Overall Health Score:[/bold] {health:.3f}")

            skipped = len(trace_evals.skipped_trace_ids)
            failed = len(trace_evals.failed_trace_ids)
            total = len(trace_evals.evaluations_by_trace_id) + skipped + failed
            console.print(f"Traces: {total} total  |  {skipped} skipped  |  {failed} failed")

        console.print(f"\nTotal elapsed: {elapsed:.1f}s")

    except ImportError:
        # Fallback without rich
        te = result.trace_evaluations
        if te:
            scores_by_metric: dict[str, list[float]] = {}
            for evals in te.evaluations_by_trace_id.values():
                for ev in evals:
                    if isinstance(ev.value, (int, float)) and not isinstance(ev.value, bool):
                        scores_by_metric.setdefault(ev.name, []).append(float(ev.value))
            print("\n=== Evaluation Results ===")
            for metric, values in sorted(scores_by_metric.items()):
                avg = sum(values) / len(values)
                print(f"  {metric:30s}: avg={avg:.3f}  n={len(values)}")
        print(f"Elapsed: {elapsed:.1f}s")


def _save_csv_report(result, run_name: str, cfg: ExperimentConfig) -> None:
    """Save a CSV report to ./results/ for offline analysis.

    Each row represents one trace (dataset item).  Columns are:
        run_name, dataset_name, judge_model, trace_id, status,
        <metric_1>, <metric_2>, …  (one column per metric, NaN if absent)

    A final SUMMARY row contains the per-metric averages.
    Scores are already pushed to Langfuse by the framework; this file is
    for quick offline inspection in Excel / pandas.
    """
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    te = result.trace_evaluations

    # Collect all metric names across every trace (preserves insertion order)
    all_metrics: list[str] = []
    seen: set[str] = set()
    rows: list[dict] = []

    if te:
        for trace_id, evals in te.evaluations_by_trace_id.items():
            row: dict = {
                "run_name": run_name,
                "dataset_name": cfg.dataset_name,
                "judge_model": cfg.judge_model,
                "trace_id": trace_id,
                "status": "ok",
            }
            for ev in evals:
                if isinstance(ev.value, (int, float)) and not isinstance(ev.value, bool):
                    row[ev.name] = round(float(ev.value), 4)
                    if ev.name not in seen:
                        all_metrics.append(ev.name)
                        seen.add(ev.name)
            rows.append(row)

        for trace_id in te.skipped_trace_ids:
            rows.append({
                "run_name": run_name,
                "dataset_name": cfg.dataset_name,
                "judge_model": cfg.judge_model,
                "trace_id": trace_id,
                "status": "skipped",
            })

        for trace_id in te.failed_trace_ids:
            rows.append({
                "run_name": run_name,
                "dataset_name": cfg.dataset_name,
                "judge_model": cfg.judge_model,
                "trace_id": trace_id,
                "status": "failed",
            })

    # Build summary row (averages over ok traces only)
    if all_metrics:
        summary: dict = {
            "run_name": run_name,
            "dataset_name": cfg.dataset_name,
            "judge_model": cfg.judge_model,
            "trace_id": "SUMMARY",
            "status": f"ok={len(te.evaluations_by_trace_id)} skipped={len(te.skipped_trace_ids)} failed={len(te.failed_trace_ids)}" if te else "no_evals",
        }
        for metric in all_metrics:
            values = [r[metric] for r in rows if metric in r]
            summary[metric] = round(sum(values) / len(values), 4) if values else ""
        rows.append(summary)

    fieldnames = ["run_name", "dataset_name", "judge_model", "trace_id", "status"] + all_metrics

    report_path = results_dir / f"{run_name}.csv"
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
