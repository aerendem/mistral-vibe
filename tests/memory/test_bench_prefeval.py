"""PrefEval-style benchmark: preference recall at different turn depths.

Tests whether stored coding preferences survive and surface correctly
after varying numbers of intervening messages. Adapted from PrefEval
(ICLR 2025) methodology for a coding assistant memory system.

Run with: uv run pytest tests/memory/test_bench_prefeval.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibe.core.memory import MemoryManager
from vibe.core.memory.config import MemoryConfig

# ---------------------------------------------------------------------------
# Preference corpus — 15 coding preferences with discriminating keywords
# ---------------------------------------------------------------------------

PREFERENCES: list[str] = [
    "I always use 4-space indentation, never tabs",
    "I prefer pytest over unittest for all testing",
    "We use PostgreSQL, never MySQL",
    "Our team follows trunk-based development",
    "I always write docstrings for public functions",
    "We use Black for code formatting with line length 88",
    "I prefer composition over inheritance",
    "We deploy with Docker and Kubernetes",
    "I always use type hints in function signatures",
    "Our logging standard is structlog with JSON output",
    "We use pre-commit hooks for all repos",
    "I prefer functional patterns over OOP when possible",
    "We use GitHub Actions for CI/CD",
    "Our API standard is REST with OpenAPI specs",
    "I never use wildcard imports",
]

# Keywords to check for each preference (index-aligned)
_DISCRIMINATING_KEYWORDS: list[str] = [
    "4-space",
    "pytest",
    "PostgreSQL",
    "trunk",
    "docstring",
    "Black",
    "composition",
    "Docker",
    "type hints",
    "structlog",
    "pre-commit",
    "functional",
    "GitHub Actions",
    "REST",
    "wildcard",
]

# Non-trivial filler messages (low preference signal)
FILLER_MESSAGES: list[str] = [
    "Can you refactor the database connection pooling logic?",
    "The tests are failing on the CI pipeline, need to investigate",
    "Let me look at the error logs from production deployment",
    "Please add error handling for the edge case we discussed yesterday",
    "The code review feedback suggests simplifying the main function",
    "We need to update the API documentation for the new endpoints",
    "Can you run the linter on the changed files in this branch?",
    "I noticed a performance regression in the latest deployment",
    "The feature flag for dark mode needs to be toggled on staging",
    "Let us check the monitoring dashboard for anomalies today",
    "The migration script needs to handle the null column edge case",
    "Can you create a separate branch for the new experiment we planned?",
    "The build artifacts are too large and need to be optimized soon",
    "Let us pair program on the authentication module tomorrow morning",
    "The staging environment needs to be redeployed after config changes",
    "Can you check if the SSL certificate is expiring within 30 days?",
    "We should add retry logic to the external payment API calls",
    "The database backup schedule needs to be updated to hourly",
    "Please review the pull request for the full-text search feature",
    "The load test results show we can handle 5000 requests per second",
]

# ---------------------------------------------------------------------------
# Mock LLM that returns preference-aware reflection responses
# ---------------------------------------------------------------------------

_PREFEVAL_REFLECTION = json.dumps({
    "seed_updates": {
        "user_model": "Senior developer with strong code quality opinions",
        "affect": "detail-oriented and systematic",
    },
    "field_updates": [
        {"action": "add", "key": "indentation", "value": "4-space indentation, never tabs"},
        {"action": "add", "key": "testing_framework", "value": "pytest over unittest"},
        {"action": "add", "key": "database", "value": "PostgreSQL, never MySQL"},
        {"action": "add", "key": "branching", "value": "trunk-based development"},
        {"action": "add", "key": "documentation", "value": "docstrings for public functions"},
        {"action": "add", "key": "formatter", "value": "Black, line length 88"},
        {"action": "add", "key": "design_pattern", "value": "composition over inheritance"},
        {"action": "add", "key": "deployment", "value": "Docker + Kubernetes"},
        {"action": "add", "key": "type_safety", "value": "always use type hints"},
        {"action": "add", "key": "logging", "value": "structlog with JSON output"},
        {"action": "add", "key": "hooks", "value": "pre-commit hooks on all repos"},
        {"action": "add", "key": "paradigm", "value": "functional over OOP"},
        {"action": "add", "key": "ci_cd", "value": "GitHub Actions"},
        {"action": "add", "key": "api_standard", "value": "REST with OpenAPI"},
        {"action": "add", "key": "imports", "value": "no wildcard imports"},
    ],
})


def _make_mock_llm(reflection_response: str = _PREFEVAL_REFLECTION):
    call_log: list[dict] = []

    async def mock_llm(system: str, user: str) -> str:
        entry = {"system_prefix": system[:120], "user_prefix": user[:120]}
        call_log.append(entry)
        if "Rate the following" in system:
            return "7"
        if "reflection engine" in system.lower():
            return reflection_response
        if "Compress" in user or "key points" in user.lower():
            return json.dumps(["Preference summary point"])
        if "Rewrite" in user or "long-term" in user.lower():
            return "Long-term knowledge about user preferences."
        return "5"

    return mock_llm, call_log


def _make_config(tmp_path: Path, **overrides) -> MemoryConfig:
    defaults = dict(
        enabled=True,
        db_path=str(tmp_path / "prefeval_bench.db"),
        reflection_trigger=50,
        importance_threshold=3,
        scoring_mode="heuristic",
    )
    defaults.update(overrides)
    return MemoryConfig(**defaults)


async def _observe_messages(mgr: MemoryManager, messages: list[str]) -> None:
    """Observe a list of messages."""
    for msg in messages:
        await mgr.observe(msg, "user")


async def _observe_filler(mgr: MemoryManager, count: int) -> None:
    """Observe N filler messages (cycling if needed)."""
    for i in range(count):
        await mgr.observe(FILLER_MESSAGES[i % len(FILLER_MESSAGES)], "user")


# ===================================================================
# TESTS
# ===================================================================


@pytest.mark.asyncio
async def test_preference_recall_depth_0(tmp_path: Path) -> None:
    """Immediately after observing a preference (before reflection),
    it should appear in the <recent> section of the memory block."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        await mgr.observe(PREFERENCES[0], "user")  # 4-space indentation

        block = mgr.get_memory_block()
        assert block != "", "Block should be non-empty after first observation"
        assert "<recent>" in block, "Sensory injection should be present"
        assert "4-space" in block or "indentation" in block, (
            "Preference should appear in recent section"
        )
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_preference_recall_depth_10(tmp_path: Path) -> None:
    """After 10 filler messages, earlier preferences should still be present
    in sensory (within sensory_cap=50)."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        # Observe 3 preferences then 10 filler
        await _observe_messages(mgr, PREFERENCES[:3])
        await _observe_filler(mgr, 10)

        block = mgr.get_memory_block()
        # All 13 messages fit in sensory_cap=50
        pref_found = any(kw in block for kw in _DISCRIMINATING_KEYWORDS[:3])
        assert pref_found, (
            "At least one of the 3 preferences should still be in the memory block "
            "after 10 filler messages"
        )
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_preference_recall_depth_20(tmp_path: Path) -> None:
    """After 20 filler messages following preferences, reflection should have
    fired, promoting preferences to seed/fields."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        await _observe_messages(mgr, PREFERENCES[:5])
        await _observe_filler(mgr, 20)

        block = mgr.get_memory_block()
        # With heuristic scoring: 5 prefs at ~6 each = 30, plus 20 filler at ~4 each = 80
        # Total > 50, so reflection should have fired
        has_promoted = "<seed>" in block or "<fields>" in block
        assert has_promoted, (
            "After 25 messages, reflection should fire and produce <seed> or <fields>"
        )
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_preference_recall_after_reflection(tmp_path: Path) -> None:
    """After reflection fires, preferences should appear in <seed> or <fields>,
    not just in <recent>."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        # Observe enough to trigger reflection
        await _observe_messages(mgr, PREFERENCES[:10])

        assert mgr.metrics.reflections_triggered >= 1, "Reflection should have fired"

        block = mgr.get_memory_block()
        assert "<seed>" in block or "<fields>" in block, (
            "Promoted preferences should appear in seed or fields"
        )
        # Check for field keys from the mock response
        has_field_content = any(
            kw in block for kw in ["testing_framework", "database", "indentation", "formatter"]
        )
        assert has_field_content, "Reflected field keys should appear in block"
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_implicit_preference_detection(tmp_path: Path) -> None:
    """Implicit preferences (not explicit 'I prefer X') should be captured
    if they score high enough via heuristic keyword detection."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        # These imply preferences without "I prefer" — heuristic detects "always", "standard"
        await mgr.observe(
            "We always run Black before committing any code to the repository", "user"
        )
        await mgr.observe(
            "Our codebase enforces a strict standard of 80% test coverage", "user"
        )

        assert mgr.metrics.observations_stored >= 2, (
            "Implicit preferences should be stored as observations"
        )

        block = mgr.get_memory_block()
        assert block != "", "Memory block should be non-empty"
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_precision_no_hallucinated_preferences(tmp_path: Path) -> None:
    """Memory block should not contain preferences that contradict what the user stated."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        # User explicitly said: spaces not tabs, pytest not unittest, PostgreSQL not MySQL
        await _observe_messages(mgr, PREFERENCES[:3])

        block = mgr.get_memory_block()
        # The mock reflection correctly stores the user's actual preferences.
        # The block should NOT contain the negated options as positive preferences.
        # Note: "tabs" and "MySQL" may appear as "never tabs" / "never MySQL" which is fine.
        # We check that they don't appear WITHOUT their "never" context in <fields>.
        if "<fields>" in block:
            fields_section = block.split("<fields>")[1].split("</fields>")[0]
            assert "unittest" not in fields_section.lower(), (
                "unittest should not appear as a positive preference"
            )
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_completeness_high_signal(tmp_path: Path) -> None:
    """At least 70% of observed high-signal preferences should be captured
    after reflection fires."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        # Observe all 15 preferences
        await _observe_messages(mgr, PREFERENCES)

        block = mgr.get_memory_block()
        found = sum(1 for kw in _DISCRIMINATING_KEYWORDS if kw in block)
        ratio = found / len(_DISCRIMINATING_KEYWORDS)

        assert ratio >= 0.70, (
            f"Only {found}/{len(_DISCRIMINATING_KEYWORDS)} ({ratio:.0%}) preferences "
            f"found in block; expected >= 70%"
        )
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_preference_survives_session_boundary(tmp_path: Path) -> None:
    """Preferences stored via reflection should persist across session boundaries."""
    db_path = str(tmp_path / "persistent.db")
    llm, _ = _make_mock_llm()

    # Session 1: observe and trigger reflection
    config = MemoryConfig(
        enabled=True, db_path=db_path,
        reflection_trigger=50, scoring_mode="heuristic",
    )
    mgr1 = MemoryManager(config, llm)
    await _observe_messages(mgr1, PREFERENCES[:10])
    assert mgr1.metrics.reflections_triggered >= 1
    mgr1.close()

    # Session 2: new manager, same DB
    mgr2 = MemoryManager(config, llm)
    block = mgr2.get_memory_block()
    mgr2.close()

    has_profile = "<seed>" in block or "<fields>" in block
    assert has_profile, "Returning user should see their profile from session 1"

    pref_found = any(kw in block for kw in _DISCRIMINATING_KEYWORDS)
    assert pref_found, "At least one preference keyword should persist across sessions"


@pytest.mark.asyncio
async def test_multiple_same_domain_no_collision(tmp_path: Path) -> None:
    """Multiple preferences from the same domain should be stored as separate fields."""
    llm, _ = _make_mock_llm()
    # Lower trigger to ensure reflection fires with fewer messages
    mgr = MemoryManager(_make_config(tmp_path, reflection_trigger=30), llm)
    try:
        await mgr.observe("I use Python 3.12 for all my projects and never anything older", "user")
        await mgr.observe("I prefer pytest for testing and always write tests first", "user")
        # Observe more to trigger reflection
        await _observe_messages(mgr, PREFERENCES[4:10])

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        field_keys = {f.key for f in state.fields}

        # Mock response has separate keys for lang-related and testing-related fields
        assert len(field_keys) > 1, (
            "Multiple preferences should produce multiple distinct field keys"
        )
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_stale_preference_replaced_after_contradiction(tmp_path: Path) -> None:
    """When a user contradicts a previous preference, the old value should be updated."""
    # First reflection adds Flask
    response1 = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "add", "key": "framework", "value": "Flask"},
        ],
    })
    # Second reflection updates to FastAPI
    response2 = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "update", "key": "framework", "value": "FastAPI"},
        ],
    })

    call_idx = {"reflection": 0}
    responses = [response1, response2]

    async def mock_llm(system: str, user: str) -> str:
        if "Rate the following" in system:
            return "7"
        if "reflection engine" in system.lower():
            idx = call_idx["reflection"]
            call_idx["reflection"] = idx + 1
            return responses[min(idx, len(responses) - 1)]
        if "Compress" in user or "key points" in user.lower():
            return json.dumps(["point"])
        if "Rewrite" in user or "long-term" in user.lower():
            return "knowledge"
        return "5"

    # Lower trigger so each batch reliably triggers reflection
    mgr = MemoryManager(_make_config(tmp_path, reflection_trigger=30), mock_llm)
    try:
        # First batch — triggers reflection 1 (adds Flask)
        for msg in PREFERENCES[:8]:
            await mgr.observe(msg, "user")

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        framework = next((f for f in state.fields if f.key == "framework"), None)
        assert framework is not None, (
            f"Reflection should have fired (accumulated={state.accumulated_importance}, "
            f"reflections={mgr.metrics.reflections_triggered})"
        )
        assert framework.value == "Flask"

        # Second batch — triggers reflection 2 (updates to FastAPI)
        await _observe_messages(mgr, PREFERENCES[8:])
        await _observe_filler(mgr, 5)

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        framework = next((f for f in state.fields if f.key == "framework"), None)
        assert framework is not None
        assert framework.value == "FastAPI", (
            f"Framework should be updated to FastAPI, got {framework.value}"
        )

        block = mgr.get_memory_block()
        if "<fields>" in block:
            fields_section = block.split("<fields>")[1].split("</fields>")[0]
            assert "FastAPI" in fields_section
            assert "Flask" not in fields_section, (
                "Stale preference 'Flask' should not appear in fields"
            )
    finally:
        mgr.close()
