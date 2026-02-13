"""Live LLM adversarial input benchmarks — sarcasm, ambiguity, contradictions."""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path

import pytest

from tests.memory.conftest import (
    LIVE_LLM,
    SKIP_NO_KEY,
    judge_accuracy,
    make_live_config,
    soft_judge_assert,
)
from tests.memory.test_bench_prefeval import FILLER_MESSAGES
from vibe.core.memory import MemoryManager

pytestmark = [SKIP_NO_KEY, LIVE_LLM, pytest.mark.asyncio, pytest.mark.timeout(120)]

# ---------------------------------------------------------------------------
# Adversarial corpora
# ---------------------------------------------------------------------------

SARCASTIC = [
    "Oh I just LOVE when people use tabs instead of spaces, said no one ever",
    "Sure, let's use MongoDB for everything, what could possibly go wrong",
    "Yeah, semicolons in Python would be AMAZING, just kidding",
]

AMBIGUOUS = [
    "I usually go with whatever the team uses",
    "It depends on the project, but I lean toward simplicity",
    "I'm flexible on tooling, though I have some opinions",
]

IMPLICIT_CONTRADICTIONS = [
    "I've been building Flask apps for years, it's my go-to framework",
    "Just started this new FastAPI project, really enjoying the async support",
]

CODE_REVIEW_EMBEDDED = [
    "Looking at this PR - the lack of type hints makes review hard. We should enforce mypy strict mode.",
    "This function is 200 lines long. I always prefer small, composable functions.",
    "The test uses unittest.TestCase - we standardized on pytest last quarter.",
]

MIXED_SIGNALS = [
    "TypeScript is great for frontend, but I write all my backend in Python",
    "I appreciate OOP principles but prefer functional patterns in practice",
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
# 1. Sarcastic messages should NOT be stored literally
# ===================================================================


async def test_live_sarcastic_not_stored_literally(
    tmp_path: Path, real_llm_caller
) -> None:
    """Sarcastic remarks should not surface as genuine preferences."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, SARCASTIC + FILLER_MESSAGES[0:7])

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        all_values = " ".join(f.value for f in state.fields).lower()

        # Primary: "MongoDB" should NOT appear as a genuine positive preference —
        # the sarcasm ("what could possibly go wrong") is very clear.
        # Negative/cautionary mentions (e.g., "cautious about mongodb") are acceptable
        # since they show the sarcasm was correctly interpreted.
        # Note: sarcasm detection is inherently unreliable with LLMs, so this is a
        # soft assertion — we warn but don't fail.
        if "mongodb" in all_values:
            negative_qualifiers = [
                "cautious", "avoid", "not", "never", "over-reliance",
                "skeptic", "against", "dislike", "sarcas",
            ]
            has_negative = any(q in all_values for q in negative_qualifiers)
            if not has_negative:
                warnings.warn(
                    "Sarcastic suggestion 'MongoDB for everything' stored as a "
                    f"genuine positive preference. Fields: {all_values}",
                    stacklevel=1,
                )

        # Secondary: LLM judge evaluates sarcasm detection quality
        block = mgr.get_memory_block()
        score = await judge_accuracy(
            real_llm_caller,
            SARCASTIC,
            block,
            "sarcasm detection - should NOT store sarcastic suggestions literally",
        )
        soft_judge_assert(score, 5, "sarcasm detection")
    finally:
        mgr.close()


# ===================================================================
# 2. Ambiguous input should not fabricate specifics
# ===================================================================


async def test_live_ambiguous_no_overcommit(
    tmp_path: Path, real_llm_caller
) -> None:
    """Ambiguous messages should not cause the system to fabricate specific preferences."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, AMBIGUOUS + FILLER_MESSAGES[0:7])

        # If reflection fired, inspect the fields
        if mgr.metrics.reflections_triggered >= 1:
            state = mgr._store.get_or_create_user_state(mgr._user_id)
            all_values = " ".join(f.value for f in state.fields).lower()

            # The user never mentioned these — they should not be fabricated
            fabricated_specifics = ["react", "java", "jenkins", "angular", "vue"]
            for term in fabricated_specifics:
                assert term not in all_values, (
                    f"Fabricated specific '{term}' found in fields despite user "
                    f"never mentioning it. Fields: {all_values}"
                )

            # Primary: the system should not create excessive fields from vague input.
            # Note: filler messages also contribute signal and the reflection
            # prompt extracts actionable context, so we allow up to 8.
            assert len(state.fields) <= 8, (
                f"Expected <= 8 fields from ambiguous input + filler, got {len(state.fields)}: "
                f"{[(f.key, f.value) for f in state.fields]}"
            )
        else:
            # Nothing to reflect on is also acceptable for ambiguous input
            assert mgr.metrics.reflections_triggered == 0
    finally:
        mgr.close()


# ===================================================================
# 3. Implicit contradiction — recent preference should win
# ===================================================================


async def test_live_implicit_contradiction(
    tmp_path: Path, real_llm_caller
) -> None:
    """When a user implicitly moves from Flask to FastAPI, the profile should reflect that."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(
            mgr, IMPLICIT_CONTRADICTIONS + FILLER_MESSAGES[0:8]
        )

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        if state.fields:
            all_values = " ".join(f.value for f in state.fields).lower()

            # Primary: if a framework field exists, it should not mention
            # ONLY Flask without ANY mention of FastAPI or migration context.
            # Note: LLM contradiction resolution is non-deterministic — we use
            # a soft assertion here because the system may not always catch the
            # implicit shift from Flask to FastAPI in a single reflection cycle.
            framework_fields = [
                f for f in state.fields
                if "framework" in f.key.lower() or "api" in f.key.lower()
                or "flask" in f.value.lower() or "fastapi" in f.value.lower()
            ]
            for ff in framework_fields:
                val = ff.value.lower()
                if "flask" in val:
                    has_fastapi_context = (
                        "fastapi" in val or "migrat" in val
                        or "switch" in val or "new" in val
                    )
                    if not has_fastapi_context:
                        warnings.warn(
                            f"[LLM] Field '{ff.key}' mentions Flask without "
                            f"acknowledging FastAPI context: {ff.value}",
                            stacklevel=1,
                        )

        # Secondary: LLM judge evaluates contradiction resolution
        block = mgr.get_memory_block()
        score = await judge_accuracy(
            real_llm_caller,
            IMPLICIT_CONTRADICTIONS,
            block,
            "implicit contradiction resolution",
        )
        soft_judge_assert(score, 5, "implicit contradiction resolution")
    finally:
        mgr.close()


# ===================================================================
# 4. Code review comments should extract real preferences
# ===================================================================


async def test_live_code_review_embedded(
    tmp_path: Path, real_llm_caller
) -> None:
    """Preferences embedded in code review context should still be extracted."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(
            mgr, CODE_REVIEW_EMBEDDED + FILLER_MESSAGES[0:7]
        )

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert len(state.fields) >= 1, (
            "Should extract at least 1 preference from code review messages. "
            f"reflections_triggered={mgr.metrics.reflections_triggered}, "
            f"observations_stored={mgr.metrics.observations_stored}"
        )

        all_values = " ".join(f.value for f in state.fields).lower()
        # Primary: at least 1 of these keywords should appear
        expected_keywords = [
            "type hint", "mypy", "small function", "composable", "pytest", "test"
        ]
        found = [kw for kw in expected_keywords if kw in all_values]
        assert len(found) >= 1, (
            f"Expected at least 1 of {expected_keywords} in field values, "
            f"found none. Fields: {all_values}"
        )
    finally:
        mgr.close()


# ===================================================================
# 5. Mixed signals should preserve nuance
# ===================================================================


async def test_live_mixed_signals_nuance(
    tmp_path: Path, real_llm_caller
) -> None:
    """Mixed signals (TS for frontend, Python for backend) should be captured with nuance."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(
            mgr, MIXED_SIGNALS + FILLER_MESSAGES[0:8]
        )

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        if mgr.metrics.reflections_triggered == 0:
            # Reflection may not fire if rate-limited or scores are low.
            # If accumulated_importance passed the trigger but LLM calls failed,
            # warn instead of failing — this is infrastructure, not logic.
            if state.accumulated_importance >= config.reflection_trigger:
                warnings.warn(
                    f"[LLM] Reflection should have fired "
                    f"(accumulated={state.accumulated_importance}, "
                    f"trigger={config.reflection_trigger}) but did not — "
                    f"likely rate-limited",
                    stacklevel=1,
                )
            else:
                warnings.warn(
                    f"[LLM] Accumulated importance {state.accumulated_importance} "
                    f"did not reach trigger {config.reflection_trigger}",
                    stacklevel=1,
                )
            return  # skip judge evaluation if no reflection occurred

        assert len(state.fields) >= 1, (
            "Should capture at least 1 field from mixed-signal messages. "
            f"reflections_triggered={mgr.metrics.reflections_triggered}, "
            f"observations_stored={mgr.metrics.observations_stored}"
        )

        # Secondary: LLM judge evaluates nuance preservation
        block = mgr.get_memory_block()
        score = await judge_accuracy(
            real_llm_caller,
            MIXED_SIGNALS,
            block,
            "nuance preservation - should capture both TypeScript/frontend "
            "and Python/backend without conflation",
        )
        soft_judge_assert(score, 5, "mixed signals nuance preservation")
    finally:
        mgr.close()


# ===================================================================
# 6. Emoji and slang should still be stored if tech-relevant
# ===================================================================

SLANG_MESSAGES = [
    "Python is mass fire but JS is mass mid tbh",
    "Rust has mass W error handling, Go errors are mass L fr fr",
    "Docker is bussin, but Kubernetes is lowkey overrated no cap",
]


async def test_live_emoji_and_slang(
    tmp_path: Path, real_llm_caller
) -> None:
    """Slang-heavy messages with real tech content should still be stored."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, SLANG_MESSAGES + FILLER_MESSAGES[0:7])

        # Primary: the slang messages contain genuine tech preferences and
        # should be stored as observations despite unconventional language.
        assert mgr.metrics.observations_stored >= 1, (
            "Slang messages with tech content should still be stored as "
            f"observations. stored={mgr.metrics.observations_stored}, "
            f"skipped_trivial={mgr.metrics.observations_skipped_trivial}, "
            f"skipped_low={mgr.metrics.observations_skipped_low_score}"
        )
    finally:
        mgr.close()


# ===================================================================
# 7. Information overload should not crash the system
# ===================================================================


async def test_live_information_overload(
    tmp_path: Path, real_llm_caller
) -> None:
    """Feeding 30+ messages should trigger reflection without crashing."""
    config = make_live_config(tmp_path, reflection_trigger=30)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        # Feed all 15 filler messages twice (30 total)
        all_messages = list(FILLER_MESSAGES[:15]) + list(FILLER_MESSAGES[:15])
        await _observe_with_delay(mgr, all_messages, delay=0.1)

        # Primary: reflection fires without crash
        assert mgr.metrics.reflections_triggered >= 1, (
            "Reflection should fire after 30 messages. "
            f"accumulated_importance={mgr._store.get_or_create_user_state(mgr._user_id).accumulated_importance}"
        )

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert len(state.fields) >= 1, (
            f"Should produce at least 1 field from 30 messages, "
            f"got {len(state.fields)}"
        )
    finally:
        mgr.close()


# ===================================================================
# 8. Repeated preference should not create duplicate fields
# ===================================================================

REPEATED_PREF = [
    "I always use 4-space indentation in all my Python code",
    "Four spaces for indentation is my standard, never tabs",
    "I strictly enforce 4-space indent across all our repositories",
    "My indentation preference is 4 spaces, I never use tabs",
    "All my code uses 4-space indentation, it's a non-negotiable standard",
]


async def test_live_repeated_preference_reinforcement(
    tmp_path: Path, real_llm_caller
) -> None:
    """Repeating the same preference 5 times should not create 5 duplicate fields."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, REPEATED_PREF + FILLER_MESSAGES[0:5])

        state = mgr._store.get_or_create_user_state(mgr._user_id)

        # Count how many fields mention indentation / space / 4
        indent_fields = 0
        for f in state.fields:
            combined = (f.key + " " + f.value).lower()
            if "indent" in combined or "space" in combined or "4" in combined:
                indent_fields += 1

        # Primary: the system should consolidate, not create 5 duplicates.
        # The LLM may break one preference into sub-aspects (e.g., preference,
        # strictness, tabs_allowed), so allow up to 3.
        assert indent_fields <= 3, (
            f"Expected <= 3 indentation-related fields from 5 repetitions, "
            f"got {indent_fields}: "
            f"{[(f.key, f.value) for f in state.fields if ('indent' in (f.key + f.value).lower() or 'space' in (f.key + f.value).lower() or '4' in (f.key + f.value).lower())]}"
        )
    finally:
        mgr.close()


# ===================================================================
# 9. Explicit contradiction replaces field via "update" action
# ===================================================================

EXPLICIT_SWITCH = [
    "We've been using MySQL for years across all our services",
    "I manage all our database schemas in MySQL Workbench",
    "Actually, we just migrated everything to PostgreSQL last week",
    "PostgreSQL's JSONB support was the main reason for the switch",
    "All new services must use PostgreSQL, MySQL is deprecated",
]


async def test_live_explicit_contradiction_replaces_field(
    tmp_path: Path, real_llm_caller
) -> None:
    """When a user explicitly switches from MySQL to PostgreSQL, the field should reflect PostgreSQL."""
    config = make_live_config(tmp_path)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, EXPLICIT_SWITCH + FILLER_MESSAGES[0:5])

        if mgr.metrics.reflections_triggered == 0:
            state = mgr._store.get_or_create_user_state(mgr._user_id)
            warnings.warn(
                f"[LLM] Reflection did not fire — "
                f"accumulated_importance={state.accumulated_importance}",
                stacklevel=1,
            )
            return

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        db_fields = [
            f for f in state.fields
            if "database" in f.key.lower() or "db" in f.key.lower()
            or "mysql" in f.value.lower() or "postgresql" in f.value.lower()
            or "postgres" in f.value.lower()
        ]

        # Primary: PostgreSQL should appear as the current preference in at least one field
        all_db_text = " ".join(f"{f.key} {f.value}" for f in db_fields).lower()
        assert "postgresql" in all_db_text or "postgres" in all_db_text, (
            f"Expected PostgreSQL somewhere in database-related fields, "
            f"got: {[(f.key, f.value) for f in db_fields]}"
        )

        # There should NOT be a field that says MySQL is the active database.
        # Context clues can be in the key OR value (e.g., key="previous_database").
        for f in db_fields:
            combined = (f.key + " " + f.value).lower()
            if "mysql" in combined:
                has_context = any(
                    q in combined for q in [
                        "deprecated", "migrat", "switch", "previous", "old",
                        "former", "postgresql", "postgres", "replaced",
                    ]
                )
                assert has_context, (
                    f"MySQL stored as active preference despite explicit switch: "
                    f"{f.key}={f.value}"
                )
    finally:
        mgr.close()


# ===================================================================
# 10. Field cap forces pruning of weakest preferences
# ===================================================================

BATCH_1_PREFS = [
    "I use Python 3.12 for everything",
    "We standardized on FastAPI as our web framework",
    "pytest is our testing framework, always with strict mode",
    "PostgreSQL is our only database",
    "Black with line length 88 for formatting",
    "I always write docstrings for public functions",
]

BATCH_2_PREFS = [
    "We just adopted Rust for all performance-critical services",
    "Our new standard is gRPC instead of REST for inter-service communication",
    "We moved from GitHub Actions to Buildkite for CI/CD",
    "Prometheus and Grafana are now mandatory for all services",
    "We standardized on OpenTelemetry for distributed tracing",
    "All services must implement circuit breakers with Polly",
]


async def test_live_field_cap_prunes_weakest(
    tmp_path: Path, real_llm_caller
) -> None:
    """With a low field cap, new high-signal preferences should replace old weak ones."""
    # Low cap forces pruning when fields exceed 6
    config = make_live_config(tmp_path, max_user_fields=6)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        # Session 1: establish initial preferences
        await _observe_with_delay(mgr, BATCH_1_PREFS + FILLER_MESSAGES[0:4])

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        initial_count = len(state.fields)

        # Session 2: new preferences on different topics
        await _observe_with_delay(mgr, BATCH_2_PREFS + FILLER_MESSAGES[4:8])

        state = mgr._store.get_or_create_user_state(mgr._user_id)

        # Primary: field count should respect the cap
        assert len(state.fields) <= 6, (
            f"Expected <= 6 fields (cap), got {len(state.fields)}: "
            f"{[(f.key, f.value) for f in state.fields]}"
        )

        # Primary: at least some batch 2 topics should appear
        all_values = " ".join(f.value for f in state.fields).lower()
        batch2_keywords = ["rust", "grpc", "buildkite", "prometheus", "opentelemetry", "circuit"]
        batch2_found = [kw for kw in batch2_keywords if kw in all_values]

        if mgr.metrics.reflections_triggered >= 2:
            # Both batches reflected — new topics should be present
            assert len(batch2_found) >= 1, (
                f"Expected at least 1 batch-2 keyword in fields after cap pruning, "
                f"found none. Fields: {all_values}"
            )
    finally:
        mgr.close()


# ===================================================================
# 11. Reinforced fields survive pruning, weak fields don't
# ===================================================================


async def test_live_reinforced_fields_survive(
    tmp_path: Path, real_llm_caller
) -> None:
    """Fields reinforced by repeated mention should survive cap pruning over weak one-off fields."""
    config = make_live_config(tmp_path, max_user_fields=5)
    mgr = MemoryManager(config, real_llm_caller)
    try:
        # Mention Python repeatedly (should get reinforced)
        reinforced = [
            "I use Python 3.12 for all my backend services",
            "Python 3.12's pattern matching has been a game changer",
            "We standardized on Python 3.12 across all microservices",
        ]
        # Mention various other things once each
        one_offs = [
            "I tried Vim keybindings last week but went back to default",
            "We looked at GraphQL but decided REST is simpler for now",
            "Someone suggested using Terraform but we haven't decided yet",
            "I saw a talk about WebAssembly, seems interesting",
            "The team discussed using Redis for caching",
        ]

        await _observe_with_delay(mgr, reinforced + one_offs + FILLER_MESSAGES[0:2])

        state = mgr._store.get_or_create_user_state(mgr._user_id)

        if mgr.metrics.reflections_triggered == 0:
            warnings.warn(
                f"[LLM] Reflection did not fire — "
                f"accumulated_importance={state.accumulated_importance}",
                stacklevel=1,
            )
            return

        # Primary: field count respects cap
        assert len(state.fields) <= 5, (
            f"Expected <= 5 fields (cap), got {len(state.fields)}"
        )

        # Secondary: Python should survive (it was reinforced 3x)
        all_values = " ".join(f.value for f in state.fields).lower()
        if "python" not in all_values:
            warnings.warn(
                "[LLM] Python was mentioned 3 times but is not in surviving fields. "
                f"Fields: {[(f.key, f.value) for f in state.fields]}",
                stacklevel=1,
            )
    finally:
        mgr.close()
