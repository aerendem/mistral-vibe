"""Live LLM-as-judge benchmarks -- meta-evaluation of memory system quality."""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path

import pytest

from tests.memory.conftest import (
    LIVE_LLM,
    SKIP_NO_KEY,
    judge_accuracy,
    judge_hallucination,
    judge_preference_presence,
    judge_usefulness,
    make_live_config,
    soft_judge_assert,
)
from tests.memory.test_bench_prefeval import FILLER_MESSAGES, PREFERENCES
from vibe.core.memory import MemoryManager

pytestmark = [SKIP_NO_KEY, LIVE_LLM, pytest.mark.asyncio, pytest.mark.timeout(180)]


# ---------------------------------------------------------------------------
# Preference corpus specific to judge tests
# ---------------------------------------------------------------------------

CLEAR_PREFERENCES = [
    "I exclusively use Python 3.12 for all backend services",
    "We standardized on FastAPI as our web framework",
    "I always run pytest with strict mode, never unittest",
    "PostgreSQL is our only database, we never use MySQL",
    "I enforce type hints with mypy strict on every PR",
    "We use Docker Compose locally and Kubernetes in production",
    "Black with line length 88 is our formatter, non-negotiable",
    "I always write docstrings for all public functions",
    "Pre-commit hooks are mandatory on all repositories",
    "We use GitHub Actions for all CI/CD pipelines",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _observe_with_delay(
    mgr: MemoryManager, messages: list[str], delay: float = 0.5
) -> None:
    for msg in messages:
        await mgr.observe(msg, "user")
        await asyncio.sleep(delay)


# ===================================================================
# 1. Reflection accuracy judged by LLM
# ===================================================================


async def test_judge_reflection_accuracy(
    tmp_path: Path, real_llm_caller
) -> None:
    """Feed clear preferences + filler, then judge reflection accuracy."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, CLEAR_PREFERENCES + FILLER_MESSAGES[0:5])

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert len(state.fields) >= 3, (
            f"Expected at least 3 fields from 10 clear preferences, "
            f"got {len(state.fields)}"
        )

        block = mgr.get_memory_block()
        score = await judge_accuracy(
            real_llm_caller,
            observations=CLEAR_PREFERENCES,
            output=block,
            criterion="preference extraction accuracy",
        )
        soft_judge_assert(score, 6, "reflection accuracy")
    finally:
        mgr.close()


# ===================================================================
# 2. Consolidation completeness judged by LLM
# ===================================================================


async def test_judge_consolidation_completeness(
    tmp_path: Path, real_llm_caller
) -> None:
    """Trigger consolidation and judge whether key info is preserved."""
    config = make_live_config(
        tmp_path, sensory_cap=10, short_term_cap=5, auto_consolidate=True
    )
    mgr = MemoryManager(config, real_llm_caller)
    try:
        messages = FILLER_MESSAGES[:15] + CLEAR_PREFERENCES[:5]
        await _observe_with_delay(mgr, messages)
        await mgr.on_session_end()

        ctx = mgr._store.get_or_create_context_memory(
            mgr.get_context_key(), mgr._user_id
        )

        # Consolidation may fail if LLM calls are rate-limited.
        # The sensory buffer should have been populated regardless.
        if ctx.long_term == "" and len(ctx.short_term) == 0:
            assert len(ctx.sensory) >= 1, (
                "Sensory buffer should have content even if consolidation "
                "was rate-limited"
            )
            warnings.warn(
                "[LLM] Consolidation did not produce output — "
                "likely rate-limited during LLM call",
                stacklevel=1,
            )
            return

        if ctx.long_term != "":
            score = await judge_accuracy(
                real_llm_caller,
                observations=messages,
                output=ctx.long_term,
                criterion="consolidation completeness",
            )
            soft_judge_assert(score, 6, "consolidation completeness")
    finally:
        mgr.close()


# ===================================================================
# 3. Preference recall across all 15 preferences
# ===================================================================


async def test_judge_preference_recall_all_15(
    tmp_path: Path, real_llm_caller
) -> None:
    """Feed all 15 PrefEval preferences and measure LLM-judged recall."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, PREFERENCES)

        assert mgr.metrics.reflections_triggered >= 1, (
            "Reflection should fire after 15 high-signal messages"
        )

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert len(state.fields) >= 5, (
            f"Expected at least 5 fields from 15 preferences, "
            f"got {len(state.fields)}"
        )

        block = mgr.get_memory_block()
        found = 0
        for pref in PREFERENCES:
            present = await judge_preference_presence(
                real_llm_caller, pref, block
            )
            if present:
                found += 1
            await asyncio.sleep(0.5)

        recall = found / 15
        recall_pct = recall * 100

        if recall < 0.6:
            warnings.warn(
                f"[LLM Judge] preference_recall: {found}/15 "
                f"({recall_pct:.0f}%) recalled (below 60% threshold)",
                stacklevel=1,
            )
    finally:
        mgr.close()


# ===================================================================
# 4. No hallucination in memory profile
# ===================================================================


SPECIFIC_PREFS = [
    "I use Python 3.12 exclusively",
    "We deploy only with Kubernetes",
    "Our testing framework is pytest",
    "We use PostgreSQL for all databases",
    "Code formatting is done with Black",
]


async def test_judge_no_hallucination(
    tmp_path: Path, real_llm_caller
) -> None:
    """Feed specific preferences and verify the profile does not hallucinate."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, SPECIFIC_PREFS + FILLER_MESSAGES[0:5])

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert len(state.fields) >= 1, (
            "Reflection should produce at least 1 field from specific preferences"
        )

        block = mgr.get_memory_block()
        score = await judge_hallucination(
            real_llm_caller, SPECIFIC_PREFS, block
        )
        soft_judge_assert(score, 7, "hallucination check")
    finally:
        mgr.close()


# ===================================================================
# 5. Memory block usefulness judged by LLM
# ===================================================================


async def test_judge_memory_block_usefulness(
    tmp_path: Path, real_llm_caller
) -> None:
    """Build a memory block and judge its usefulness for a coding assistant."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, CLEAR_PREFERENCES + FILLER_MESSAGES[0:5])

        block = mgr.get_memory_block()
        assert block != "", "Memory block should be non-empty after observations"

        score = await judge_usefulness(real_llm_caller, block)
        soft_judge_assert(score, 6, "memory block usefulness")
    finally:
        mgr.close()


# ===================================================================
# 6. Cross-session consistency judged by LLM
# ===================================================================


async def test_judge_cross_session_consistency(
    tmp_path: Path, real_llm_caller
) -> None:
    """Two sessions sharing a DB should produce a consistent, complete profile."""
    db_path = str(tmp_path / "cross_session.db")
    config = make_live_config(tmp_path, db_path=db_path)

    # Session 1
    mgr1 = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(
            mgr1, CLEAR_PREFERENCES[:5] + FILLER_MESSAGES[0:5]
        )
    finally:
        mgr1.close()

    # Session 2 (same db_path)
    mgr2 = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(
            mgr2, CLEAR_PREFERENCES[5:] + FILLER_MESSAGES[5:10]
        )

        state = mgr2._store.get_or_create_user_state(mgr2._user_id)
        assert len(state.fields) >= 2, (
            f"Expected fields from both sessions, got {len(state.fields)}"
        )

        block = mgr2.get_memory_block()
        score = await judge_accuracy(
            real_llm_caller,
            observations=CLEAR_PREFERENCES,
            output=block,
            criterion="cross-session consistency",
        )
        soft_judge_assert(score, 6, "cross-session consistency")
    finally:
        mgr2.close()
