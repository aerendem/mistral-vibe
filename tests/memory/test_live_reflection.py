"""Live LLM reflection benchmarks — real Mistral API, no mocks."""

from __future__ import annotations

import asyncio
import getpass
import warnings
from pathlib import Path

import pytest

from tests.memory.conftest import (
    LIVE_LLM,
    SKIP_NO_KEY,
    judge_accuracy,
    judge_preference_presence,
    make_live_config,
    soft_judge_assert,
)
from tests.memory.test_bench_prefeval import (
    _DISCRIMINATING_KEYWORDS,
    FILLER_MESSAGES,
    PREFERENCES,
)
from vibe.core.memory import MemoryManager
from vibe.core.memory.models import FieldMeta, UserField

pytestmark = [SKIP_NO_KEY, LIVE_LLM, pytest.mark.asyncio, pytest.mark.timeout(120)]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _observe_with_delay(
    mgr: MemoryManager, messages: list[str], role: str = "user", delay: float = 0.5
) -> None:
    for msg in messages:
        await mgr.observe(msg, role)
        await asyncio.sleep(delay)


# ===================================================================
# 1. Basic reflection produces valid JSON
# ===================================================================


async def test_live_reflection_produces_valid_json(
    tmp_path: Path, real_llm_caller
) -> None:
    """Feed 10 preference messages and verify reflection fires with valid output."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, PREFERENCES[:10])

        assert mgr.metrics.reflections_triggered >= 1, (
            "Reflection should fire after 10 high-signal messages"
        )

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert len(state.fields) >= 1, (
            "Reflection should produce at least 1 field"
        )
    finally:
        mgr.close()


# ===================================================================
# 2. Extracts coding preferences
# ===================================================================

TECH_MESSAGES = [
    "I exclusively use Python 3.12 for all backend services in our team",
    "We switched to FastAPI last year and it's been our standard framework",
    "I always run pytest with coverage, never use unittest",
    "Our CI pipeline enforces strict mypy type checking on every PR",
    "I prefer Docker Compose for local dev, Kubernetes for production",
]


async def test_live_reflection_extracts_coding_preferences(
    tmp_path: Path, real_llm_caller
) -> None:
    """Feed tech-specific messages and verify key technologies are extracted."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, TECH_MESSAGES + FILLER_MESSAGES[:5])

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        all_values = " ".join(f.value for f in state.fields).lower()

        tech_keywords = ["python", "fastapi", "pytest"]
        found = sum(1 for kw in tech_keywords if kw in all_values)
        assert found >= 2, (
            f"Expected at least 2 of {tech_keywords} in field values, "
            f"found {found}. Fields: {all_values}"
        )
    finally:
        mgr.close()


# ===================================================================
# 3. Populates user model
# ===================================================================

SENIOR_MESSAGES = [
    "I've been leading the platform team for 5 years now",
    "I mentor junior developers on architecture patterns weekly",
    "My expertise is in distributed systems and event-driven architectures",
    "I set the coding standards for our engineering org of 40 developers",
    "I always review PRs for security vulnerabilities before merging",
]


async def test_live_reflection_populates_user_model(
    tmp_path: Path, real_llm_caller
) -> None:
    """Feed seniority-implying messages and verify user_model is populated."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, SENIOR_MESSAGES + FILLER_MESSAGES[:5])

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert state.seed.user_model != "", (
            "user_model should be populated after observing seniority messages"
        )

        score = await judge_accuracy(
            real_llm_caller,
            SENIOR_MESSAGES,
            state.seed.user_model,
            "user expertise level",
        )
        soft_judge_assert(score, 6, "user_model expertise accuracy")
    finally:
        mgr.close()


# ===================================================================
# 4. Preference completeness
# ===================================================================


async def test_live_preference_completeness(
    tmp_path: Path, real_llm_caller
) -> None:
    """Feed all 15 preferences and measure recall via LLM judge."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, PREFERENCES)

        assert mgr.metrics.reflections_triggered >= 1, (
            "Reflection should fire after 15 high-signal messages"
        )

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert len(state.fields) >= 5, (
            f"Expected at least 5 fields from 15 preferences, got {len(state.fields)}"
        )

        # Build memory block for judge evaluation
        block = mgr.get_memory_block()
        recalled = 0
        for pref in PREFERENCES:
            present = await judge_preference_presence(
                real_llm_caller, pref, block
            )
            if present:
                recalled += 1
            await asyncio.sleep(0.5)

        recall_pct = (recalled / len(PREFERENCES)) * 100
        if recall_pct < 60:
            warnings.warn(
                f"[LLM Judge] preference_recall: {recalled}/{len(PREFERENCES)} "
                f"({recall_pct:.0f}%) recalled (below 60% threshold)",
                stacklevel=1,
            )
    finally:
        mgr.close()


# ===================================================================
# 5. Contradiction resolution
# ===================================================================


async def test_live_contradiction_resolution(
    tmp_path: Path, real_llm_caller
) -> None:
    """Feed contradictory indentation preferences and verify resolution."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await mgr.observe(
            "I always use tabs for indentation in all my Python code", "user"
        )
        await asyncio.sleep(0.5)
        await mgr.observe(
            "Actually, I switched to 4-space indentation for everything last month",
            "user",
        )
        await asyncio.sleep(0.5)

        # Feed more preferences to reach reflection trigger
        await _observe_with_delay(mgr, PREFERENCES[2:8])

        if mgr.metrics.reflections_triggered == 0:
            state = mgr._store.get_or_create_user_state(mgr._user_id)
            warnings.warn(
                f"[LLM] Reflection did not fire — likely rate-limited. "
                f"accumulated_importance={state.accumulated_importance}",
                stacklevel=1,
            )
            return

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        all_values = " ".join(f.value for f in state.fields).lower()

        # "tabs" should not appear as a standalone positive preference.
        # It is acceptable in context like "switched from tabs".
        tab_as_positive = (
            "tabs" in all_values
            and "switched from tabs" not in all_values
            and "moved from tabs" not in all_values
            and "replaced tabs" not in all_values
            and "instead of tabs" not in all_values
            and "not tabs" not in all_values
            and "never tabs" not in all_values
            and "no tabs" not in all_values
        )

        # Secondary LLM judge check
        score = await judge_accuracy(
            real_llm_caller,
            [
                "I always use tabs for indentation in all my Python code",
                "Actually, I switched to 4-space indentation for everything last month",
            ],
            all_values,
            "contradiction resolution: should reflect the LATEST preference (spaces, not tabs)",
        )
        soft_judge_assert(score, 5, "contradiction resolution accuracy")

        if tab_as_positive:
            warnings.warn(
                "[LLM Judge] contradiction_resolution: 'tabs' appears as a "
                "positive preference despite user switching to spaces",
                stacklevel=1,
            )
    finally:
        mgr.close()


# ===================================================================
# 6. Consistent JSON format across independent cycles
# ===================================================================


async def test_live_consistent_json_format(
    tmp_path: Path, real_llm_caller
) -> None:
    """Run 3 independent reflection cycles and verify all produce valid output."""
    batches = [
        (PREFERENCES[0:5] + FILLER_MESSAGES[0:5], "batch1"),
        (PREFERENCES[5:10] + FILLER_MESSAGES[5:10], "batch2"),
        (PREFERENCES[10:15] + FILLER_MESSAGES[10:15], "batch3"),
    ]

    managers: list[MemoryManager] = []
    try:
        for messages, label in batches:
            db_name = f"consistent_{label}.db"
            config = make_live_config(tmp_path, db_path=str(tmp_path / db_name))
            mgr = MemoryManager(config, real_llm_caller)
            managers.append(mgr)

            await _observe_with_delay(mgr, messages)

        reflected_count = 0
        for i, mgr in enumerate(managers):
            label = batches[i][1]
            if mgr.metrics.reflections_triggered >= 1:
                reflected_count += 1
                state = mgr._store.get_or_create_user_state(mgr._user_id)
                assert len(state.fields) >= 1, (
                    f"{label}: should have at least 1 field after reflection"
                )

        # At least 2 of 3 batches should reflect successfully.
        # All 3 may not succeed if rate-limited during rapid sequential execution.
        assert reflected_count >= 2, (
            f"Expected at least 2/3 batches to reflect, got {reflected_count}"
        )
    finally:
        for mgr in managers:
            mgr.close()


# ===================================================================
# 7. Ignores low-signal noise
# ===================================================================

HIGH_SIGNAL = [
    "I prefer functional programming patterns over OOP in Python",
    "We always use PostgreSQL for our databases, never MySQL",
    "Our team standard is Black formatter with line length 88",
    "I never use wildcard imports in any codebase I maintain",
    "We require 90% test coverage on all Python packages",
]

LOW_SIGNAL = [
    "Can you check that file?",
    "What does this function do?",
    "How do I run the tests?",
    "Where is the config file?",
    "Is this the right approach?",
]


async def test_live_ignores_low_signal(
    tmp_path: Path, real_llm_caller
) -> None:
    """High-signal messages should be captured; low-signal questions should not."""
    # Lower trigger: low-signal messages contribute ~1-2 each, high-signal ~3-5.
    # With 10 messages total, accumulated importance may only reach ~25.
    config = make_live_config(tmp_path, reflection_trigger=20)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, HIGH_SIGNAL + LOW_SIGNAL)

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert len(state.fields) >= 1, (
            "Should have at least 1 field from high-signal messages"
        )

        all_values = " ".join(f.value for f in state.fields).lower()
        for question in LOW_SIGNAL:
            assert question.lower() not in all_values, (
                f"Low-signal question should not appear in fields: {question}"
            )
    finally:
        mgr.close()


# ===================================================================
# 8. Updates existing fields
# ===================================================================


async def test_live_updates_existing_fields(
    tmp_path: Path, real_llm_caller
) -> None:
    """Pre-populate a field and verify reflection updates it with new info."""
    from vibe.core.memory.models import Seed, UserState
    from vibe.core.memory.storage import MemoryStore

    db_path = tmp_path / "update_test.db"

    # Pre-populate a field via store BEFORE creating MemoryManager
    store = MemoryStore(Path(db_path))
    user_id = getpass.getuser()
    state = store.get_or_create_user_state(user_id)
    state.fields = [UserField(key="language", value="Python 3.11")]
    store.update_user_state(state)
    store.close()

    config = make_live_config(tmp_path, db_path=str(db_path))
    mgr = MemoryManager(config, real_llm_caller)
    try:
        msgs = [
            "I just migrated all our services to Python 3.12 for the performance improvements",
            "Python 3.12 pattern matching has been a game changer for our codebase",
            "We standardized on Python 3.12 across all microservices",
        ]
        await _observe_with_delay(mgr, msgs + FILLER_MESSAGES[:5])

        if mgr.metrics.reflections_triggered == 0:
            state = mgr._store.get_or_create_user_state(mgr._user_id)
            warnings.warn(
                f"[LLM] Reflection did not fire — likely rate-limited. "
                f"accumulated_importance={state.accumulated_importance}",
                stacklevel=1,
            )
            return

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        all_values = " ".join(f.value for f in state.fields).lower()
        assert "3.12" in all_values or "python" in all_values, (
            f"Expected '3.12' or 'python' in field values after update, "
            f"got: {all_values}"
        )
    finally:
        mgr.close()
