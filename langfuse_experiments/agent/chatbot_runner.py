"""Task function that runs the health chatbot orchestrator agent.

Wraps ``agents.Runner.run()`` (openai-agents SDK) so it can be passed to
``run_experiment_with_trace_evals`` as the ``task`` parameter.

Import path assumption
----------------------
The health chatbot lives in a separate repo whose root must be on PYTHONPATH:
    export PYTHONPATH=/path/to/health-chatbot-repo:$PYTHONPATH

Or install it as an editable package:
    pip install -e /path/to/health-chatbot-repo

To point at a different agent version, change the import on the line
marked ← CHANGE AGENT IMPORT and update _get_agent() below.
"""

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Env must be loaded before any src.* imports touch configs
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", verbose=False)
load_dotenv(verbose=False)

import agents  # openai-agents SDK  # noqa: E402

from src.utils.client_manager import AsyncClientManager           # noqa: E402  ← CHANGE AGENT IMPORT
from src.health_chatbot.health_agents.orchestrator import build_orchestrator_agent  # noqa: E402
from src.utils.langfuse.shared_client import langfuse_client      # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build the agent ONCE at module load — expensive to recreate per item.
# The agent is stateless; each Runner.run() call is an isolated single turn.
# ---------------------------------------------------------------------------
_client_manager: AsyncClientManager | None = None
_orchestrator_agent: "agents.Agent | None" = None


def _get_agent() -> "agents.Agent":
    """Lazy singleton: build the orchestrator agent on first call."""
    global _client_manager, _orchestrator_agent
    if _orchestrator_agent is None:
        _client_manager = AsyncClientManager()
        _orchestrator_agent = build_orchestrator_agent(_client_manager)
        logger.info("Orchestrator agent initialised: %s", _orchestrator_agent.name)
    return _orchestrator_agent


# ---------------------------------------------------------------------------
# Task function — called once per dataset item by the experiment runner
# ---------------------------------------------------------------------------

async def chatbot_task(item: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Run the health chatbot for a single evaluation item.

    This is the ``task`` function passed to ``run_experiment_with_trace_evals``.
    Each call is a fresh, single-turn interaction — no conversation history
    from other dataset items bleeds in.

    Parameters
    ----------
    input : Any
        Dataset item input, typically a plain question string.
    expected_output : Any
        Ground-truth answer (not used here; passed through by the framework).
    metadata : dict | None
        Item-level metadata from the dataset.

    Returns
    -------
    dict
        ``{"answer": str | None}`` where ``answer`` is the agent's final
        response. Returns ``None`` on failure so trace evaluators can skip
        gracefully instead of crashing.
    """

    query = item.input
    expected_output = getattr(item, "expected_output", None)
    metadata = getattr(item, "metadata", None)

    if query is None:
        answer = None
        error = "Missing 'input' in item"
        logger.warning("Item missing 'input': %s", item)
        return {"answer": answer,"error": error}


    agent = _get_agent()

    # Wrap in a Langfuse observation so the agent's internal LLM and tool
    # calls are nested under the experiment's trace and visible during
    # Pass 2 trace evaluations.
    with langfuse_client.start_as_current_observation(name="Health Chatbot — Eval Turn",as_type="agent",input=query,) as obs:
        try:
            # Non-streaming run; no session → fresh single turn per item.
            result = await agents.Runner.run(agent,input=query, max_turns=3,)

            # final_output is the agent's last text message.
            # The agent can return text, numbers, or references to plots
            # (e.g. "I've plotted X — see /tmp/chart.png").
            # We capture the full text here; evaluators treat it as the
            # "answer". Visual content embedded in the text (plot URLs,
            # file paths, markdown image links) is preserved as-is so
            # evaluators can still assess relevance and correctness of the
            # surrounding narrative.
            final_text: str = result.final_output or ""

            # Detect whether the response likely contains a plot reference
            # so evaluators can apply appropriate rubrics.
            has_plot = _contains_plot_reference(final_text)

            task_output = {"answer": final_text, "has_plot": has_plot,}
            if expected_output is not None:
                task_output["expected_output"] = expected_output
            else:
                return {"answer":final_text, "skip_judge": True,}

            obs.update(output=final_text)
            logger.info(
                "Agent replied (%d chars, plot=%s) for: '%s...'",
                len(final_text), has_plot, query[:60],
            )
            return task_output

        except Exception as exc:
            logger.error("Agent failed for '%s...': %s", query[:60], exc)
            obs.update(output=f"ERROR: {exc}")
            return {"answer": None, "has_plot": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_plot_reference(text: str) -> bool:
    """Heuristic: does the agent's response reference a generated plot?

    The visualization agent (E2B code interpreter) typically produces output
    that ends up in the final text as a file path, URL, or markdown image.
    This flag lets evaluators adjust their rubric (e.g. skip answer_correctness
    when the expected output is a chart, not a number).
    """
    lower = text.lower()
    return any(
        marker in lower
        for marker in (
            ".png", ".jpg", ".jpeg", ".svg", ".html",
            "![", "plot", "chart", "graph", "figure",
            "/tmp/", "sandbox:", "e2b://",
        )
    )
