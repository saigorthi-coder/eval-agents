# Evaluation Module: `experiment.py` and `graders/`

This document provides a detailed walkthrough of the evaluation harness at
`aieng/agent_evals/evaluation/`, covering `experiment.py` and every file in
the `graders/` subpackage, including the test suite. It also documents every
external dependency these files import.

---

## Module Overview

```
evaluation/
├── __init__.py          # Public re-exports for the whole harness
├── experiment.py        # Top-level experiment runner functions
├── trace.py             # Second-pass trace evaluator runner (also documented below)
├── types.py             # Shared type definitions and dataclasses
└── graders/
    ├── __init__.py      # Public re-exports for graders subpackage
    ├── config.py        # LLMRequestConfig dataclass
    ├── llm_judge.py     # LLM-as-a-judge evaluator factory
    └── _utils.py        # Internal shared helpers
```

The harness sits on top of [Langfuse](https://langfuse.com/)'s
`dataset.run_experiment` API. It adds:

1. A shared `AsyncClientManager` so the Langfuse and OpenAI clients are
   instantiated once and reused across all evaluators.
2. A two-pass experiment runner that separates output scoring (Pass 1) from
   trace-level scoring — tool use, latency, cost, etc. (Pass 2).
3. A reusable LLM-as-a-judge evaluator factory that any agent domain can
   import without reimplementing prompt engineering or retry logic.

---

## External Dependencies (Imported Outside This Module)

### `aieng.agent_evals.async_client_manager.AsyncClientManager`

Imported by: `experiment.py`, `graders/llm_judge.py`, `trace.py`

**File:** `aieng/agent_evals/async_client_manager.py`

A singleton that manages the lifecycle of shared async clients. It uses lazy
initialization — clients are only created on first access — and stores a class-level
`_singleton_instance` so the same object is reused across the entire harness.

Key properties and methods:

| Member | Type | Description |
|---|---|---|
| `get_instance()` | `classmethod` | Returns (or creates) the singleton. |
| `configs` | `property → Configs` | Returns or creates a `Configs` instance loaded from environment. |
| `openai_client` | `property → AsyncOpenAI` | Returns or creates an `AsyncOpenAI` client using `configs.openai_api_key` and `configs.openai_base_url`. |
| `langfuse_client` | `property → Langfuse` | Returns or creates a `Langfuse` client using `configs.langfuse_public_key`, `configs.langfuse_secret_key`, and `configs.langfuse_host`. |
| `close()` | `async method` | Closes all initialized clients (OpenAI, Langfuse, database connections) and resets `_initialized`. |
| `is_initialized()` | `method → bool` | Returns whether any client has been initialized. |

The manager also exposes `aml_db()` and `report_generation_db()` for database
connections used by other agents; these are not used by the evaluation harness
directly.

The `configs` property reads from the `Configs` dataclass
(`aieng.agent_evals.configs`), which is populated from environment variables
and holds API keys, base URLs, and model names including
`configs.default_evaluator_model` used by the LLM judge.

---

### `aieng.agent_evals.evaluation.types` — Type Definitions

Imported by: `experiment.py`, `graders/_utils.py`, `graders/llm_judge.py`, `trace.py`

**File:** `aieng/agent_evals/evaluation/types.py`

Centralizes all type definitions so modules depend on a stable internal API
rather than importing Langfuse internals directly.

#### Re-exported from `langfuse`

| Name | Origin | Description |
|---|---|---|
| `Evaluation` | `langfuse.experiment` | A single scored metric with `name`, `value`, optional `comment`, `data_type`, `metadata`, and `config_id`. |
| `EvaluatorFunction` | `langfuse.experiment` | Protocol for item-level evaluator callables. |
| `ExperimentItemResult` | `langfuse.experiment` | Result for a single dataset item, includes `trace_id` and task output. |
| `ExperimentResult` | `langfuse.experiment` | Top-level result from `dataset.run_experiment`; contains `item_results`. |
| `RunEvaluatorFunction` | `langfuse.experiment` | Protocol for run-level (aggregate) evaluators. |
| `TaskFunction` | `langfuse.experiment` | Protocol for the task callable that runs the agent per item. |
| `CompositeEvaluatorFunction` | `langfuse.batch_evaluation` | Evaluator that receives all item-level scores plus item context. |

#### Defined in `types.py`

**`TraceEvalStatus` (Enum)**

Three states for a trace evaluation attempt:
- `OK` — evaluated successfully.
- `SKIPPED` — trace data was incomplete or missing.
- `FAILED` — an error occurred during evaluation.

**`TraceEvaluatorFunction` (Protocol)**

Called in Pass 2. Signature:

```python
def __call__(
    self, *, trace: TraceWithFullDetails, item_result: ExperimentItemResult, **kwargs
) -> Evaluation | list[Evaluation] | Awaitable[Evaluation | list[Evaluation]]
```

**`TraceObservationPredicate`**

A `Callable[[ObservationsView], bool]` used to classify observations (e.g.,
distinguish tool calls from assistant turns).

**`TraceMetrics` (frozen dataclass)**

| Field | Type | Description |
|---|---|---|
| `tool_call_count` | `int` | Estimated number of tool-call observations. |
| `turn_count` | `int` | Estimated number of assistant turn observations. |
| `observation_count` | `int` | Total observations in the trace. |
| `latency_sec` | `float \| None` | End-to-end trace latency. |
| `total_input_tokens` | `int` | Total input tokens across observations. |
| `total_output_tokens` | `int` | Total output tokens across observations. |
| `total_cost` | `float \| None` | Total cost in USD. |

**`TraceWaitConfig` (frozen dataclass)**

Controls polling behaviour while waiting for trace ingestion:

| Field | Default | Description |
|---|---|---|
| `max_wait_sec` | `180.0` | Maximum total wait time. |
| `initial_delay_sec` | `1.0` | Initial poll delay. |
| `max_delay_sec` | `10.0` | Maximum delay between retries. |
| `backoff_multiplier` | `2.0` | Exponential backoff factor. |

**`TraceEvalResult` (dataclass)**

Container returned by `run_trace_evaluations`:

| Field | Type | Description |
|---|---|---|
| `evaluations_by_trace_id` | `dict[str, list[Evaluation]]` | Successful evaluations per trace. |
| `skipped_trace_ids` | `list[str]` | Traces skipped due to incomplete data. |
| `failed_trace_ids` | `list[str]` | Traces that errored during evaluation. |
| `errors_by_trace_id` | `dict[str, str]` | Error messages for skipped/failed traces. |
| `run_evaluations` | `list[Evaluation]` | Aggregate metrics written at the run level. |

**`EvaluationResult` (frozen dataclass)**

Top-level result returned by `run_experiment_with_trace_evals`:

| Field | Type | Description |
|---|---|---|
| `experiment` | `ExperimentResult` | Pass 1 result from Langfuse. |
| `trace_evaluations` | `TraceEvalResult \| None` | Pass 2 result, or `None` if not run. |

---

### Third-Party Libraries

#### `langfuse`

Used throughout. The Langfuse Python SDK provides:
- `Langfuse` — the main client class (dataset fetch, score creation, trace retrieval).
- `langfuse.experiment.Evaluation` — the scored metric type.
- `langfuse.api.ScoreDataType` — enum for `BOOLEAN`, `NUMERIC`, `CATEGORICAL` (used in `_utils.py`).
- `langfuse.api.ObservationsView` — individual observation from a trace.
- `langfuse.api.resources.commons.types.trace_with_full_details.TraceWithFullDetails` — fully hydrated trace object.
- `langfuse.batch_evaluation.CompositeEvaluatorFunction` — composite evaluator protocol.

#### `openai`

Used in `graders/_utils.py` and `async_client_manager.py`:
- `AsyncOpenAI` — async HTTP client for chat completions.
- `openai.types.chat.parsed_chat_completion.ParsedChatCompletion` — typed response from `chat.completions.parse`.
- Exception types for retry logic: `APIConnectionError`, `APIStatusError`, `APITimeoutError`, `InternalServerError`, `RateLimitError`.

#### `pydantic`

Used in `graders/llm_judge.py` and `graders/_utils.py`:
- `BaseModel` — base class for `LLMJudgeMetric` and `LLMJudgeResponse`.
- `Field` — used to add `ge=0.0, le=1.0` validation on `confidence`.
- `ValidationError` — raised when Pydantic model construction fails (tested in `test_llm_judge.py`).

#### `tenacity`

Used in `graders/_utils.py` and `trace.py` for retry logic:
- `AsyncRetrying` — async retry context manager.
- `retry_if_exception(predicate)` — only retries when predicate returns `True`.
- `stop_after_attempt(n)` — limits retries to `n` attempts.
- `stop_after_delay(sec)` — limits retries by total elapsed time (used in trace wait).
- `wait_exponential(multiplier, min, max)` — exponential backoff with configurable bounds.
- `RetryError` — raised when all retry attempts are exhausted.

#### `httpx`

Used in `trace.py`: `httpx.TransportError` is treated as a retryable network
error when fetching traces.

---

## `experiment.py`

**Path:** [experiment.py](experiment.py)

### Purpose

A thin convenience layer over Langfuse's `dataset.run_experiment`. Provides
two public functions that compose the full evaluation workflow.

| Function | Description |
|---|---|
| `run_experiment` | Single-pass: runs the task and scores outputs. |
| `run_experiment_with_trace_evals` | Two-pass: single-pass then a trace-scoring second pass. |

### Imports

```python
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.trace import run_trace_evaluations
from aieng.agent_evals.evaluation.types import (
    CompositeEvaluatorFunction, EvaluationResult, EvaluatorFunction,
    ExperimentResult, RunEvaluatorFunction, TaskFunction,
    TraceEvaluatorFunction, TraceWaitConfig,
)
```

---

### `run_experiment`

```python
def run_experiment(
    dataset_name: str,
    *,
    name: str,
    task: TaskFunction,
    evaluators: list[EvaluatorFunction],
    composite_evaluator: CompositeEvaluatorFunction | None = None,
    run_evaluators: list[RunEvaluatorFunction] | None = None,
    description: str | None = None,
    run_name: str | None = None,
    max_concurrency: int = 10,
    metadata: dict[str, Any] | None = None,
) -> ExperimentResult
```

**What it does:**

1. Calls `AsyncClientManager.get_instance()` to retrieve the shared Langfuse
   client (avoids repeated authentication and re-instantiation).
2. Fetches the named Langfuse dataset with `langfuse_client.get_dataset(dataset_name)`.
3. Delegates entirely to `dataset.run_experiment(...)`, forwarding all parameters.

**Key parameters:**

- `dataset_name` — Name of the Langfuse dataset to evaluate against.
- `name` — Human-readable label for the experiment run shown in the Langfuse UI.
- `task` — A `TaskFunction` called per dataset item; receives `input`,
  `expected_output`, `metadata`, and `**kwargs`; returns the agent output.
- `evaluators` — Item-level evaluators; each receives
  `(input, output, expected_output, metadata)` and returns one or more
  `Evaluation` objects.
- `composite_evaluator` — Optional; sees all item-level scores plus item context;
  useful for weighted pass/fail decisions.
- `run_evaluators` — Optional run-level evaluators for aggregate metrics (e.g.,
  pass rate across all items).
- `max_concurrency` — Thread/task pool size for parallel item execution (default `10`).

**Returns:** `ExperimentResult` — Langfuse result with `item_results` and
`run_evaluations`.

---

### `run_experiment_with_trace_evals`

```python
def run_experiment_with_trace_evals(
    dataset_name: str,
    *,
    name: str,
    task: TaskFunction,
    evaluators: list[EvaluatorFunction],
    trace_evaluators: list[TraceEvaluatorFunction],
    composite_evaluator: CompositeEvaluatorFunction | None = None,
    run_evaluators: list[RunEvaluatorFunction] | None = None,
    description: str | None = None,
    run_name: str | None = None,
    max_concurrency: int = 10,
    metadata: dict[str, Any] | None = None,
    trace_wait: TraceWaitConfig | None = None,
    trace_max_concurrency: int = 10,
) -> EvaluationResult
```

**Two-pass workflow:**

- **Pass 1** — Calls `run_experiment(...)` to produce agent outputs and record
  trace IDs for each item. Trace data may still be ingesting at this point.
- **Pass 2** — Calls `run_trace_evaluations(experiment_result, trace_evaluators,
  wait=trace_wait, ...)`. This polls Langfuse until each trace is fully ingested
  (per `TraceWaitConfig`), then runs `trace_evaluators` against the populated
  `TraceWithFullDetails` objects.

**Additional parameters over `run_experiment`:**

- `trace_evaluators` — `TraceEvaluatorFunction`s receiving `(trace, item_result)`;
  return `Evaluation | list[Evaluation]`.
- `trace_wait` — A `TraceWaitConfig` controlling polling intervals and max wait
  time (defaults to `TraceWaitConfig()`: 180 s max, exponential backoff).
- `trace_max_concurrency` — Concurrency for the second-pass trace evaluations.

**Returns:** `EvaluationResult` — frozen dataclass wrapping both the
`ExperimentResult` and an optional `TraceEvalResult`.

---

## `trace.py` (Referenced by `experiment.py`)

**Path:** [trace.py](trace.py)

Implements the second-pass trace evaluation logic called by
`run_experiment_with_trace_evals`.

### Key Public Functions

**`run_trace_evaluations(experiment_result, trace_evaluators, *, wait, max_concurrency)`**

Synchronous entry point for Pass 2. Runs `_run_trace_evaluations_async` via
`run_coroutine_sync` (from `aieng.agent_evals.async_utils`).

Internally:
1. Filters `experiment_result.item_results` to those with a non-`None` `trace_id`.
2. Creates an `asyncio.Semaphore(max_concurrency)` for bounded concurrency.
3. For each item: calls `_evaluate_trace(langfuse_client, item_result, trace_evaluators, wait)`.
4. Aggregates results into `TraceEvalResult`, separating OK / skipped / failed traces.
5. Calls `flush_traces()` (from `aieng.agent_evals.langfuse`) at the end.

**`extract_trace_metrics(trace, *, tool_call_predicate, turn_predicate)`**

Public utility for trace evaluators to compute common metrics without writing
their own heuristics. Uses `_default_tool_call_predicate` and
`_default_turn_predicate` unless overridden. Returns a `TraceMetrics` dataclass.

### Internal Helpers in `trace.py`

| Function | Description |
|---|---|
| `_run_trace_evaluations_async` | Async implementation of trace evaluation with semaphore-bounded concurrency. |
| `_evaluate_trace` | Fetches and evaluates a single trace; returns `(evaluations, status, error_message)`. |
| `_fetch_trace_with_wait` | Polls `langfuse_client.async_api.trace.get(trace_id)` with tenacity retry until `_trace_ready`. |
| `_trace_ready` | Heuristic: `trace.input is not None and trace.output is not None`. |
| `_is_retryable_trace_fetch_error` | Retries on `_TraceNotReadyError`, `NotFoundError`, `httpx.TransportError`, and `ApiError` 408/429/5xx. |
| `_default_tool_call_predicate` | Identifies tool-call observations by type, name, or metadata keys. |
| `_default_turn_predicate` | Identifies assistant turn observations by type (`generation`), name, or `role` metadata. |
| `_sum_token_usage` | Sums token usage across observations, handling multiple provider naming conventions. |
| `_extract_total_cost` | Reads `trace.total_cost` or sums `cost_details` from observations. |
| `_upload_trace_scores` | Persists `Evaluation` objects to Langfuse via `langfuse_client.create_score`. |
| `_normalize_evaluations` | Normalizes evaluator outputs: unwraps awaitables, wraps single `Evaluation` in a list, handles `dict` returns. |
| `_get_evaluator_name` | Best-effort name extraction for error messages, handles `functools.partial`. |

---

## `graders/` Subpackage

Reusable evaluator factories shared across agent domains. All factories return
Langfuse-compatible evaluator callables.

---

### `graders/__init__.py`

**Path:** [graders/\_\_init\_\_.py](graders/__init__.py)

Re-exports the public API so callers only need one import path:

```python
from aieng.agent_evals.evaluation.graders import (
    DEFAULT_LLM_JUDGE_RUBRIC,
    LLMJudgeMetric,
    LLMJudgeResponse,
    create_llm_as_judge_evaluator,
)
```

All four symbols originate in `llm_judge.py`.

---

### `graders/config.py`

**Path:** [graders/config.py](graders/config.py)

#### Imports

```python
from dataclasses import dataclass, field
from typing import Any
```

No project-level imports; only standard library.

#### `LLMRequestConfig` (frozen dataclass)

Configuration for the underlying judge model call. Frozen so it is safe to
share across evaluator closures without mutation risk.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str \| None` | `None` | Explicit judge model. Falls back to `AsyncClientManager.configs.default_evaluator_model` if `None`. |
| `temperature` | `float` | `0.0` | Sampling temperature. Keep at `0.0` for deterministic grading. |
| `max_completion_tokens` | `int \| None` | `None` | Optional token cap on the judge completion. |
| `timeout_sec` | `float \| None` | `None` | Per-request timeout in seconds. |
| `extra_request_kwargs` | `dict[str, Any]` | `{}` | Any additional kwargs forwarded to `chat.completions.parse`. |
| `retry_max_attempts` | `int` | `5` | Maximum retry attempts on transient failures. Set to `1` to disable. |
| `retry_initial_wait_sec` | `float` | `1.0` | Initial backoff delay. |
| `retry_max_wait_sec` | `float` | `10.0` | Maximum backoff delay. |
| `retry_backoff_multiplier` | `float` | `2.0` | Exponential backoff multiplier. |

---

### `graders/llm_judge.py`

**Path:** [graders/llm\_judge.py](graders/llm_judge.py)

The main grader module. Provides an **LLM-as-a-judge evaluator factory** that
wraps any OpenAI-compatible model to score agent outputs against expected
outputs using a customizable rubric.

#### Imports

```python
# Standard library
from pathlib import Path
from typing import Any

# Internal — same package
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import (
    LLMRequestConfig,
    build_error_evaluation,
    load_markdown,
    render_system_prompt_with_optional_rubric,
    run_structured_parse_call,
    serialize_for_prompt,
)
from aieng.agent_evals.evaluation.types import Evaluation, EvaluatorFunction

# Third-party
from pydantic import BaseModel, Field
```

Note that `LLMRequestConfig` is re-exported through `_utils.py` for convenience
(it originates in `config.py`).

#### Module-Level Constants

**`DEFAULT_SYSTEM_PROMPT_TEMPLATE`**

Default system prompt. Instructs the judge to:
1. Analyze the input's intent and constraints.
2. Verify negative constraints (format, length, prohibited content).
3. Write an explanation before assigning scores (chain-of-thought).
4. Assign scores per rubric.
5. Return strict JSON (no markdown code fences).

Contains a `{rubric_section}` placeholder injected by
`render_system_prompt_with_optional_rubric`.

**`DEFAULT_USER_PROMPT_TEMPLATE`**

Default user message with three labeled sections:
`# Input`, `# Expected Output`, `# Candidate Output (To Evaluate)`.

**`DEFAULT_LLM_JUDGE_RUBRIC`**

Built-in rubric with three binary (0/1) metrics:

| Metric | Passes (1) when… |
|---|---|
| `correctness` | Candidate output is materially consistent with expected output; no material contradictions. |
| `completeness` | Candidate output includes all materially required information from expected output. |
| `constraint_adherence` | Candidate output follows all explicit constraints from input. Passes automatically if input has no constraints. |

#### Pydantic Models

**`LLMJudgeMetric`**

A single scored metric returned by the judge.

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | Yes | Metric name (maps to `Evaluation.name`). |
| `value` | `bool \| int \| float \| str` | Yes | Score (maps to `Evaluation.value`). |
| `comment` | `str \| None` | No | One-sentence explanation for this metric. |
| `confidence` | `float \| None` | No | Judge's confidence in `[0.0, 1.0]`; validated by Pydantic `Field(ge=0.0, le=1.0)`. |
| `metadata` | `dict[str, Any] \| None` | No | Optional extra metadata (e.g., source references). |

**`LLMJudgeResponse`**

Top-level structured output schema the judge model must return.

| Field | Type | Required | Description |
|---|---|---|---|
| `explanation` | `str` | Yes | Global rationale for all scores; fallback `comment` for metrics that omit their own. |
| `metrics` | `list[LLMJudgeMetric]` | Yes | One or more scored metrics. |

#### `create_llm_as_judge_evaluator`

```python
def create_llm_as_judge_evaluator(
    *,
    name: str = "llm_judge",
    model_config: LLMRequestConfig | None = None,
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    rubric_markdown: str | Path | None = None,
    error_metric_name: str | None = None,
) -> EvaluatorFunction
```

**What it does at factory time:**

1. Resolves `model_config` (defaults to `LLMRequestConfig()` if `None`).
2. Loads rubric text via `load_markdown(rubric_markdown)` — accepts inline
   markdown, a `.md` file `Path`, or falls back to `DEFAULT_LLM_JUDGE_RUBRIC`.
3. Renders the system prompt via `render_system_prompt_with_optional_rubric(...)`.
4. Sets `resolved_error_metric_name` to `error_metric_name or f"{name}_error"`.
5. Returns the async inner function `_evaluator` with `__name__` set to `name`.

**What `_evaluator` does at call time:**

1. Formats the user prompt using `serialize_for_prompt` on all three inputs.
2. Retrieves the singleton `AsyncClientManager` and its `openai_client`.
3. Calls `run_structured_parse_call(...)` with the config, prompts, and
   `LLMJudgeResponse` as the response format.
4. Extracts the parsed `LLMJudgeResponse` from `completion.choices[0].message.parsed`.
5. Maps metrics to `Evaluation` objects via `_to_evaluations(judge_response)`.
6. On any exception, returns a single error `Evaluation` via
   `build_error_evaluation(name=resolved_error_metric_name, ...)`.

**Parameter notes:**

- `rubric_markdown` can be raw text or a file path, enabling rubrics to be
  maintained as separate versioned `.md` files.
- `error_metric_name` defaults to `f"{name}_error"` so error metrics are
  namespaced to the evaluator.

#### `_to_evaluations` (internal)

```python
def _to_evaluations(response: LLMJudgeResponse | None) -> list[Evaluation]
```

Maps a parsed `LLMJudgeResponse` to Langfuse `Evaluation` objects:

- Raises `ValueError` if `response` is `None` or has an empty `metrics` list.
- Merges `metric.confidence` into `metric_metadata["confidence"]`.
- Falls back to `response.explanation` for the `comment` of metrics that have none.
- Sets evaluation `metadata` only when the merged dict is non-empty.

---

### `graders/_utils.py`

**Path:** [graders/\_utils.py](graders/_utils.py)

Internal helpers shared by grader modules. Not part of the public API.

#### Imports

```python
# Standard library
import json
from pathlib import Path
from typing import Any, TypeVar, cast

# Internal — same project
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation

# Third-party
from langfuse.api import ScoreDataType
from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, RateLimitError
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential
```

#### `run_structured_parse_call`

```python
async def run_structured_parse_call(
    *,
    openai_client: Any,
    default_model: str,
    model_config: LLMRequestConfig,
    system_prompt: str,
    user_prompt: str,
    response_format: type[T],
) -> ParsedChatCompletion[T]
```

Wraps `openai_client.chat.completions.parse(...)` with tenacity exponential-backoff
retries.

- **Model resolution**: `model_config.model` if set, otherwise `default_model`.
- **Request construction**: merges `extra_request_kwargs`, then sets `model`,
  `messages`, `response_format`, `temperature`, and optionally
  `max_completion_tokens` and `timeout`.
- **Retry policy**: configured from `LLMRequestConfig`; retries on transient
  API errors classified by `is_retryable_api_exception`.
- **Reraise**: propagates the last exception when all attempts are exhausted.
- **Defensive fallback**: raises `RuntimeError` if tenacity exits without
  returning (should not happen in normal operation).

#### `is_retryable_api_exception`

```python
def is_retryable_api_exception(exc: BaseException) -> bool
```

Tenacity predicate returning `True` for:
- `APIConnectionError`, `APITimeoutError`, `RateLimitError`, `InternalServerError`
- `APIStatusError` with status code 408, 429, or ≥ 500

#### `build_error_evaluation`

```python
def build_error_evaluation(*, name: str, error: Exception, prefix: str) -> Evaluation
```

Constructs a deterministic boolean `Evaluation` (`value=True`,
`data_type=ScoreDataType.BOOLEAN`) used when an evaluator fails. Encodes the
error class name and message in `metadata` for observability:

```python
metadata={"error_type": error.__class__.__name__, "error": str(error)}
comment=f"{prefix}: {message}"
```

#### `render_system_prompt_with_optional_rubric`

```python
def render_system_prompt_with_optional_rubric(
    *, system_prompt_template: str, rubric_text: str | None
) -> str
```

Two rendering modes:

1. **Placeholder present** (`{rubric_section}` in template): calls
   `template.format(rubric_section=rubric_section)`.
2. **No placeholder**: appends `# Rubric\n{rubric_text}` at the end of the
   template, keeping custom templates simple.

If `rubric_text` is falsy, the placeholder is replaced with an empty string
and no section is appended.

#### `load_markdown`

```python
def load_markdown(markdown: str | Path | None) -> str | None
```

Loads rubric/prompt text from multiple sources:

- `None` → returns `None`.
- `Path` → reads file content with UTF-8 encoding.
- `str` with `.md` suffix that exists on disk → reads that file.
- Any other `str` → returns it as-is (inline markdown text).

#### `serialize_for_prompt`

```python
def serialize_for_prompt(value: Any) -> str
```

Serializes any Python value to deterministic, readable JSON for prompt
injection. Uses `json.dumps(value, ensure_ascii=False, indent=2, default=str)`.
Non-serializable objects fall back to `str(value)`. Consistent formatting
enables snapshot testing of prompts.

---

## Tests: `tests/graders/test_llm_judge.py`

**Path:** [tests/.../test\_llm\_judge.py](../../tests/aieng/agent_evals/evaluation/graders/test_llm_judge.py)

Unit tests for `create_llm_as_judge_evaluator` and related internals.

#### Imports

```python
# Standard library
from types import SimpleNamespace
from unittest.mock import AsyncMock

# Third-party
import pytest
from pydantic import ValidationError

# Internal — module under test
from aieng.agent_evals.evaluation import graders as graders_package
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.graders.llm_judge import (
    DEFAULT_LLM_JUDGE_RUBRIC,
    LLMJudgeMetric,
    LLMJudgeResponse,
    _to_evaluations,
    create_llm_as_judge_evaluator,
)
```

### Test Helpers

**`_completion(parsed_response)`** — Builds a minimal `SimpleNamespace` mock
that mimics `ParsedChatCompletion` with
`choices[0].message.parsed = parsed_response`.

**`fake_manager` fixture** — `monkeypatch`es `AsyncClientManager.get_instance`
to return a `SimpleNamespace` with a stub `openai_client` and
`default_evaluator_model="gpt-default-evaluator"`. Prevents any real HTTP
calls.

### Test Cases

#### `test_make_evaluator_success_with_custom_rubric_maps_and_wires_calls`

Happy path with a custom rubric and two metrics (one with `confidence`,
one without `comment`).

Verifies:
- `evaluator.__name__` is set to the provided `name`.
- Both metrics are mapped to `Evaluation` objects.
- `confidence` is merged into `metadata["confidence"]`.
- `response.explanation` is used as fallback `comment` when metric has none.
- `run_structured_parse_call` is called with the fake client, the default
  model, the config, `LLMJudgeResponse` as `response_format`, the rubric text
  in the system prompt, and correct user prompt sections.

#### `test_make_evaluator_uses_default_rubric_when_none`

`rubric_markdown=None` path.

Verifies:
- `DEFAULT_LLM_JUDGE_RUBRIC` is injected into the system prompt.
- Metric from judge response is correctly mapped.
- The package re-export `graders_package.DEFAULT_LLM_JUDGE_RUBRIC` equals the
  module-level constant.

#### `test_make_evaluator_error_paths_return_deterministic_error_metric`

Parameterized across two error scenarios:

| Scenario | Error | `error_metric_name` | Expected metric name |
|---|---|---|---|
| `parse_error` | `RuntimeError` from `run_structured_parse_call` | `None` | `"quality_judge_error"` |
| `prompt_template_key_error` | `KeyError` from bad template | `"custom_error_metric"` | `"custom_error_metric"` |

Verifies:
- Exactly one error `Evaluation` is returned.
- `value is True`.
- `comment.startswith("LLM judge error: ")`.
- `metadata["error_type"]` matches the exception class name.
- `run_structured_parse_call` is awaited or not based on where the error occurs.

#### `test_to_evaluations_rejects_missing_metrics`

Parameterized with `None` and `LLMJudgeResponse(explanation="...", metrics=[])`.

Verifies: `ValueError` matching `"must contain at least one metric"` is raised.

#### `test_llm_judge_metric_confidence_validation_bounds`

Verifies `confidence=0.0` and `confidence=1.0` are accepted, and
`confidence=1.1` raises `ValidationError`.

---

## Data Flow Summary

```
run_experiment_with_trace_evals
│
├─ Pass 1: run_experiment
│    ├─ AsyncClientManager.get_instance() → Langfuse client
│    ├─ langfuse.get_dataset(dataset_name)
│    └─ dataset.run_experiment(task, evaluators, ...)
│         └─ For each dataset item (up to max_concurrency parallel):
│              ├─ task(input, ...) → output
│              └─ evaluator(input, output, expected_output, metadata)
│                   └─ [create_llm_as_judge_evaluator]
│                        ├─ serialize_for_prompt(input/output/expected)
│                        ├─ run_structured_parse_call (tenacity retries)
│                        │    └─ openai_client.chat.completions.parse(...)
│                        │         → LLMJudgeResponse
│                        └─ _to_evaluations → list[Evaluation]
│
└─ Pass 2: run_trace_evaluations
     ├─ AsyncClientManager.get_instance() → Langfuse client
     ├─ asyncio.Semaphore(trace_max_concurrency)
     └─ For each trace_id from Pass 1:
          ├─ _fetch_trace_with_wait (tenacity retry until trace ready)
          │    └─ langfuse.async_api.trace.get(trace_id) → TraceWithFullDetails
          └─ trace_evaluator(trace, item_result) → list[Evaluation]
               └─ _upload_trace_scores → langfuse.create_score(...)
```

---

## Quick Reference: Public API

### From `evaluation` package

```python
from aieng.agent_evals.evaluation import (
    run_experiment,
    run_experiment_with_trace_evals,
    run_trace_evaluations,
    extract_trace_metrics,
)
```

### From `graders` subpackage

```python
from aieng.agent_evals.evaluation.graders import (
    create_llm_as_judge_evaluator,   # factory → EvaluatorFunction
    LLMJudgeMetric,                  # Pydantic model for individual metrics
    LLMJudgeResponse,                # Pydantic model for full judge response
    DEFAULT_LLM_JUDGE_RUBRIC,        # Built-in binary rubric string
)
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
```

### Minimal usage example

```python
from aieng.agent_evals.evaluation import run_experiment
from aieng.agent_evals.evaluation.graders import create_llm_as_judge_evaluator
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig

def my_task(*, input, **kwargs):
    return {"answer": my_agent(input["question"])}

judge = create_llm_as_judge_evaluator(
    name="answer_quality",
    model_config=LLMRequestConfig(model="gpt-4o-mini", temperature=0.0),
)

result = run_experiment(
    "my_qa_dataset",
    name="baseline-v1",
    task=my_task,
    evaluators=[judge],
)
```
