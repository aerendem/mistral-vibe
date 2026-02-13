"""Shared fixtures for live LLM memory benchmark tests.

Provides a real_llm_caller fixture that makes actual Mistral API calls,
LLM-as-judge helpers, and MemoryConfig factory for live tests.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import warnings
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path

import pytest

from vibe.core.memory.config import MemoryConfig

# ---------------------------------------------------------------------------
# Capture the real API key at import time, BEFORE the global conftest
# autouse fixture (tests/conftest.py:64-66) monkeypatches it to "mock".
#
# The key may be in os.environ directly (export MISTRAL_API_KEY=...)
# or only in ~/.vibe/.env (set by the onboarding screen). We check both,
# matching the production login method (load_dotenv_values in config.py:34).
# ---------------------------------------------------------------------------


def _resolve_api_key() -> str:
    """Get the real Mistral API key from env or ~/.vibe/.env."""
    key = os.environ.get("MISTRAL_API_KEY", "")
    if key and key != "mock":
        return key
    # Fall back to the dotenv file used by production (same as load_dotenv_values)
    try:
        from dotenv import dotenv_values

        from vibe.core.paths.global_paths import GLOBAL_ENV_FILE

        env_path = GLOBAL_ENV_FILE.path
        if env_path.is_file():
            vals = dotenv_values(env_path)
            return vals.get("MISTRAL_API_KEY", "")
    except Exception:
        pass
    return ""


_REAL_MISTRAL_API_KEY = _resolve_api_key()

SKIP_NO_KEY = pytest.mark.skipif(
    not _REAL_MISTRAL_API_KEY,
    reason="MISTRAL_API_KEY not set",
)
LIVE_LLM = pytest.mark.live_llm


# ---------------------------------------------------------------------------
# Strict env-var scoping for real API key
# ---------------------------------------------------------------------------


@contextmanager
def _restore_api_key():
    """Temporarily restore the real API key in os.environ.

    The global conftest sets MISTRAL_API_KEY=mock.  MistralBackend.__init__
    reads os.getenv(provider.api_key_env_var), so we need the real value
    present during backend construction AND the actual API call.
    """
    old = os.environ.get("MISTRAL_API_KEY")
    os.environ["MISTRAL_API_KEY"] = _REAL_MISTRAL_API_KEY
    try:
        yield
    finally:
        if old is not None:
            os.environ["MISTRAL_API_KEY"] = old
        else:
            os.environ.pop("MISTRAL_API_KEY", None)


# ---------------------------------------------------------------------------
# real_llm_caller fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_llm_caller() -> Callable[[str, str], Awaitable[str]]:
    """Return a Callable[[str, str], Awaitable[str]] that calls the real Mistral API.

    Uses the same resolution chain as AgentLoop._select_backend() (agent_loop.py:279-283):
    VibeConfig -> get_active_model -> get_provider_for_model -> BACKEND_FACTORY.
    """
    from vibe.core.config import VibeConfig
    from vibe.core.llm.backend.factory import BACKEND_FACTORY
    from vibe.core.types import LLMMessage, Role

    # Resolve model/provider from user's config (same as production).
    # Must restore the real API key so VibeConfig reads the correct env.
    with _restore_api_key():
        config = VibeConfig()
        model = config.get_active_model()
        provider = config.get_provider_for_model(model)
        timeout = config.api_timeout

    _log = logging.getLogger(__name__)

    async def _call(system_prompt: str, user_prompt: str) -> str:
        messages = [
            LLMMessage(role=Role.system, content=system_prompt),
            LLMMessage(role=Role.user, content=user_prompt),
        ]
        # Retry with exponential backoff on rate-limit (429) errors.
        max_retries = 4
        for attempt in range(max_retries + 1):
            try:
                with _restore_api_key():
                    backend = BACKEND_FACTORY[provider.backend](
                        provider=provider, timeout=timeout
                    )
                    async with backend as b:
                        result = await b.complete(
                            model=model,
                            messages=messages,
                            temperature=0.1,
                            tools=None,
                            max_tokens=500,
                            tool_choice=None,
                            extra_headers=None,
                        )
                return result.message.content or ""
            except Exception as exc:
                if "rate limit" in str(exc).lower() and attempt < max_retries:
                    wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                    _log.info("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                    await asyncio.sleep(wait)
                    continue
                raise

    return _call


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def make_live_config(tmp_path: Path, **overrides) -> MemoryConfig:
    """Create a MemoryConfig for live LLM tests.

    Uses scoring_mode="llm" and reflection_trigger=30 (lower than production
    default 50) to account for non-deterministic LLM scoring variance.
    """
    defaults = dict(
        enabled=True,
        db_path=str(tmp_path / "live_bench.db"),
        reflection_trigger=30,
        importance_threshold=3,
        scoring_mode="llm",
    )
    defaults.update(overrides)
    return MemoryConfig(**defaults)


# ---------------------------------------------------------------------------
# LLM-as-judge helpers
# ---------------------------------------------------------------------------


async def judge_accuracy(
    llm_caller: Callable[[str, str], Awaitable[str]],
    observations: list[str],
    output: str,
    criterion: str,
) -> int:
    """Ask the LLM to rate output quality on a 1-10 scale.

    Returns parsed int (0 on parse failure).
    """
    system = (
        "You are evaluating a memory system's output. "
        "Given original observations and the system's output, rate accuracy on 1-10.\n"
        "10 = perfectly captures all important information correctly.\n"
        "1 = completely wrong or missing critical information.\n"
        f"Evaluation criterion: {criterion}\n"
        "Respond with ONLY a single integer."
    )
    user = "Original observations:\n"
    user += "\n".join(f"- {o}" for o in observations)
    user += f"\n\nSystem output:\n{output}"
    raw = await llm_caller(system, user)
    return _parse_judge_score(raw)


async def judge_preference_presence(
    llm_caller: Callable[[str, str], Awaitable[str]],
    preference: str,
    memory_block: str,
) -> bool:
    """Ask the LLM whether a memory block captures a specific preference."""
    system = (
        "You are evaluating whether a memory block captures a specific user preference. "
        "Answer ONLY 'yes' or 'no'."
    )
    user = f"Preference: {preference}\n\nMemory block:\n{memory_block}"
    raw = await llm_caller(system, user)
    return raw.strip().lower().startswith("y")


async def judge_hallucination(
    llm_caller: Callable[[str, str], Awaitable[str]],
    observations: list[str],
    output: str,
) -> int:
    """Rate fidelity: does the output contain only information from observations?"""
    system = (
        "You are checking a memory profile for hallucinated information. "
        "Given original observations and the profile, rate fidelity 1-10.\n"
        "10 = every piece of information is directly supported by observations.\n"
        "1 = the profile contains fabricated information not in the observations.\n"
        "Respond with ONLY a single integer."
    )
    user = "Original observations:\n"
    user += "\n".join(f"- {o}" for o in observations)
    user += f"\n\nMemory profile:\n{output}"
    raw = await llm_caller(system, user)
    return _parse_judge_score(raw)


async def judge_usefulness(
    llm_caller: Callable[[str, str], Awaitable[str]],
    memory_block: str,
) -> int:
    """Rate how useful the memory block would be for a coding assistant."""
    system = (
        "You are a coding assistant evaluating a memory block for usefulness. "
        "Rate how helpful this memory block would be for personalizing responses 1-10.\n"
        "10 = extremely useful, covers tech stack, preferences, work patterns.\n"
        "1 = useless or misleading.\n"
        "Respond with ONLY a single integer."
    )
    raw = await llm_caller(system, f"Memory block:\n{memory_block}")
    return _parse_judge_score(raw)


def _parse_judge_score(raw: str) -> int:
    """Extract an integer from LLM judge response."""
    match = re.search(r"\d+", raw.strip())
    if match:
        return max(0, min(10, int(match.group())))
    return 0


def soft_judge_assert(score: int, threshold: int, label: str) -> None:
    """Warn (but don't fail) if a judge score is below threshold."""
    if score < threshold:
        warnings.warn(
            f"[LLM Judge] {label}: scored {score}/10 (below threshold {threshold})",
            stacklevel=2,
        )
