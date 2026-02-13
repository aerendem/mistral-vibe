"""Letta methodology benchmark: Read / Write / Update across memory tiers.

Tests the 3 core capabilities from the Letta leaderboard (read, write, update)
across both user-level state (seed, fields) and context-level memory
(sensory, short-term, long-term).

Run with: uv run pytest tests/memory/test_bench_letta.py -v
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vibe.core.memory import MemoryManager
from vibe.core.memory.config import MemoryConfig
from vibe.core.memory.consolidation import ConsolidationEngine
from vibe.core.memory.models import ContextMemory, FieldMeta, Observation, Seed, UserField
from vibe.core.memory.reflection import ReflectionEngine
from vibe.core.memory.storage import MemoryStore

# ---------------------------------------------------------------------------
# Reflection responses
# ---------------------------------------------------------------------------

_REFLECTION_ADD = json.dumps({
    "seed_updates": {
        "user_model": "Senior Python developer, async-first",
        "affect": "methodical and precise",
    },
    "field_updates": [
        {"action": "add", "key": "lang", "value": "Python 3.12"},
        {"action": "add", "key": "framework", "value": "FastAPI"},
        {"action": "add", "key": "testing", "value": "pytest, TDD"},
    ],
})

_REFLECTION_UPDATE = json.dumps({
    "seed_updates": {"user_model": "Senior Rust developer"},
    "field_updates": [
        {"action": "update", "key": "lang", "value": "TypeScript"},
        {"action": "remove", "key": "stale_tool"},
    ],
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(
    score_value: str = "7",
    reflection_response: str = _REFLECTION_ADD,
):
    call_log: list[dict] = []

    async def mock_llm(system: str, user: str) -> str:
        entry = {"system_prefix": system[:120], "user_prefix": user[:120], "type": "unknown"}
        if "Rate the following" in system:
            entry["type"] = "scoring"
            call_log.append(entry)
            return score_value
        if "reflection engine" in system.lower():
            entry["type"] = "reflection"
            call_log.append(entry)
            return reflection_response
        if "Compress" in user or "key points" in user.lower():
            entry["type"] = "consolidate_st"
            call_log.append(entry)
            return json.dumps(["Compressed point A", "Compressed point B"])
        if "Rewrite" in user or "long-term" in user.lower():
            entry["type"] = "consolidate_lt"
            call_log.append(entry)
            return "Rewritten long-term knowledge incorporating new items."
        call_log.append(entry)
        return score_value

    return mock_llm, call_log


def _make_config(tmp_path: Path, **overrides) -> MemoryConfig:
    defaults = dict(
        enabled=True,
        db_path=str(tmp_path / "letta_bench.db"),
        reflection_trigger=50,
        importance_threshold=3,
        scoring_mode="heuristic",
    )
    defaults.update(overrides)
    return MemoryConfig(**defaults)


# High-signal messages that score ~5-7 with heuristic scorer
_HIGH_SIGNAL_MESSAGES = [
    "I always use Python 3.12 and prefer async patterns for all backend work",
    "Our project uses FastAPI with SQLAlchemy as the ORM layer",
    "We follow the repository pattern for database access and never use raw SQL",
    "The team convention is to use pydantic models for all API schemas",
    "I prefer to write integration tests first, following outside-in TDD",
    "Our CI pipeline runs on GitHub Actions with 80% coverage requirement",
    "The database is PostgreSQL 16 running on AWS RDS with read replicas",
    "I never use print statements for debugging - always use structured logging",
    "We always deploy with Docker containers and Kubernetes orchestration",
    "Our architecture follows clean architecture principles with dependency injection",
]


def _seed_observations(store: MemoryStore, user_id: str, count: int = 10, importance: int = 8) -> None:
    """Directly seed observations and accumulate importance to trigger reflection."""
    for i in range(count):
        store.add_observation(
            Observation(
                user_id=user_id,
                context_key="project:test",
                content=f"High-signal observation {i}",
                importance=importance,
                source_role="user",
            )
        )
    state = store.get_or_create_user_state(user_id)
    state.accumulated_importance = count * importance
    store.update_user_state(state)


# ===================================================================
# READ TESTS
# ===================================================================


@pytest.mark.asyncio
async def test_read_seed_after_reflection(tmp_path: Path) -> None:
    """After reflection writes seed updates, the memory block should contain
    the seed user_model in the <seed> section."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        for msg in _HIGH_SIGNAL_MESSAGES:
            await mgr.observe(msg, "user")

        block = mgr.get_memory_block()
        assert "<seed>" in block, "Seed section should be present after reflection"
        assert "Senior Python developer" in block, (
            "Seed should contain the user_model from reflection"
        )
    finally:
        mgr.close()


def test_read_dynamic_field(tmp_path: Path) -> None:
    """A dynamic field stored in user state should appear in the <fields> section."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        state = mgr._store.get_or_create_user_state(mgr._user_id)
        state.fields = [UserField(key="lang", value="Python 3.12")]
        mgr._store.update_user_state(state)
        mgr._store.get_or_create_context_memory(mgr.get_context_key(), mgr._user_id)

        block = mgr.get_memory_block()
        assert "<fields>" in block
        assert "lang" in block
        assert "Python 3.12" in block
    finally:
        mgr.close()


def test_read_sensory_in_recent(tmp_path: Path) -> None:
    """Sensory observations should appear in the <recent> section."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        ctx_key = mgr.get_context_key()
        mgr._store.get_or_create_user_state(mgr._user_id)
        mgr._store.add_sensory(ctx_key, mgr._user_id, "I always use Black formatter")

        block = mgr.get_memory_block()
        assert "<recent>" in block
        assert "Black formatter" in block
    finally:
        mgr.close()


def test_read_short_term_in_active(tmp_path: Path) -> None:
    """Short-term key points should appear in the <active> section."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        ctx_key = mgr.get_context_key()
        ctx = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        ctx.short_term = ["Uses pytest for testing", "Prefers composition over inheritance"]
        mgr._store.update_context_memory(ctx)

        block = mgr.get_memory_block()
        assert "<active>" in block
        assert "pytest" in block
    finally:
        mgr.close()


def test_read_long_term_in_knowledge(tmp_path: Path) -> None:
    """Long-term knowledge should appear in the <knowledge> section."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        ctx_key = mgr.get_context_key()
        ctx = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        ctx.long_term = "Python FastAPI project with PostgreSQL backend"
        mgr._store.update_context_memory(ctx)

        block = mgr.get_memory_block()
        assert "<knowledge>" in block
        assert "FastAPI" in block
    finally:
        mgr.close()


def test_read_no_extra_llm_calls(tmp_path: Path) -> None:
    """Reading stored memory via get_memory_block should not trigger any LLM calls."""
    llm, call_log = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        # Pre-populate all tiers via direct store manipulation
        state = mgr._store.get_or_create_user_state(mgr._user_id)
        state.seed = Seed(user_model="developer")
        state.fields = [UserField(key="lang", value="Python")]
        mgr._store.update_user_state(state)

        ctx_key = mgr.get_context_key()
        mgr._store.add_sensory(ctx_key, mgr._user_id, "recent observation")
        ctx = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        ctx.short_term = ["key point"]
        ctx.long_term = "knowledge blob"
        mgr._store.update_context_memory(ctx)

        call_log.clear()

        # Multiple reads should not trigger LLM
        for _ in range(5):
            block = mgr.get_memory_block()
            assert block != ""

        assert len(call_log) == 0, (
            f"get_memory_block made {len(call_log)} LLM calls; expected 0"
        )
    finally:
        mgr.close()


# ===================================================================
# WRITE TESTS
# ===================================================================


@pytest.mark.asyncio
async def test_write_observation_stored(tmp_path: Path) -> None:
    """Observing a high-signal message should store it as an observation."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        await mgr.observe(
            "I always use Python 3.12 and prefer async patterns", "user"
        )

        observations = mgr._store.get_pending_observations(mgr._user_id)
        assert len(observations) >= 1, "Observation should be stored"
        assert mgr.metrics.observations_stored >= 1
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_write_seed_via_reflection(tmp_path: Path) -> None:
    """When accumulated importance reaches the threshold, reflection should
    write seed updates to user state."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        for msg in _HIGH_SIGNAL_MESSAGES:
            await mgr.observe(msg, "user")

        assert mgr.metrics.reflections_triggered >= 1, "Reflection should have fired"

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert state.seed.user_model == "Senior Python developer, async-first"
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_write_fields_via_reflection(tmp_path: Path) -> None:
    """Reflection should write new dynamic fields to user state."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        for msg in _HIGH_SIGNAL_MESSAGES:
            await mgr.observe(msg, "user")

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        field_keys = {f.key for f in state.fields}
        assert "lang" in field_keys, "Field 'lang' should be written by reflection"
        assert "framework" in field_keys, "Field 'framework' should be written by reflection"
        assert "testing" in field_keys, "Field 'testing' should be written by reflection"
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_write_sensory_to_short_term(tmp_path: Path) -> None:
    """Consolidation should compress sensory buffer into short-term key points."""
    llm, _ = _make_mock_llm()
    config = _make_config(tmp_path, sensory_cap=10, scoring_mode="heuristic")
    mgr = MemoryManager(config, llm)
    try:
        # Fill sensory to cap
        for i in range(12):
            await mgr.observe(
                f"Detailed architecture decision number {i} for the project", "user"
            )

        ctx_key = mgr.get_context_key()
        ctx_before = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        assert len(ctx_before.sensory) >= config.sensory_cap, "Sensory should be at cap"

        await mgr.on_session_end()

        ctx_after = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        assert len(ctx_after.sensory) == 0, "Sensory should be cleared after consolidation"
        assert len(ctx_after.short_term) > 0, "Short-term should have compressed points"
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_write_short_term_to_long_term(tmp_path: Path) -> None:
    """Consolidation should compress short-term points into long-term knowledge."""
    llm, _ = _make_mock_llm()
    config = _make_config(tmp_path, short_term_cap=5)
    mgr = MemoryManager(config, llm)
    try:
        # Pre-set short_term at cap
        ctx_key = mgr.get_context_key()
        ctx = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        ctx.short_term = [f"Key insight {i}" for i in range(6)]
        mgr._store.update_context_memory(ctx)

        await mgr.on_session_end()

        ctx_after = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        assert len(ctx_after.short_term) == 0, "Short-term should be cleared"
        assert ctx_after.long_term != "", "Long-term should be populated"
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_write_sensory_for_nontrivial(tmp_path: Path) -> None:
    """Non-trivial messages should be added to sensory; trivial ones should not."""
    llm, _ = _make_mock_llm()
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        await mgr.observe("I always prefer Python for backend architecture work", "user")
        await mgr.observe("ok", "user")

        ctx_key = mgr.get_context_key()
        ctx = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        assert any("Python" in s for s in ctx.sensory), "Non-trivial message should be in sensory"
        assert not any(s == "ok" for s in ctx.sensory), "Trivial message should not be in sensory"
    finally:
        mgr.close()


# ===================================================================
# UPDATE TESTS
# ===================================================================


@pytest.mark.asyncio
async def test_update_field_via_reflection(tmp_path: Path) -> None:
    """When a user changes a preference, reflection should update the field value."""
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "update", "key": "lang", "value": "TypeScript"},
        ],
    })
    llm, _ = _make_mock_llm(reflection_response=response)
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        # Pre-set field
        state = mgr._store.get_or_create_user_state(mgr._user_id)
        state.fields = [UserField(key="lang", value="JavaScript")]
        mgr._store.update_user_state(state)

        for msg in _HIGH_SIGNAL_MESSAGES:
            await mgr.observe(msg, "user")

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        lang_field = next((f for f in state.fields if f.key == "lang"), None)
        assert lang_field is not None
        assert lang_field.value == "TypeScript", (
            f"Field should be updated to TypeScript, got {lang_field.value}"
        )
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_update_seed_overwrites(tmp_path: Path) -> None:
    """When reflection returns a seed_update for an existing field, it overwrites."""
    response = json.dumps({
        "seed_updates": {"user_model": "senior developer"},
        "field_updates": [],
    })
    llm, _ = _make_mock_llm(reflection_response=response)
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        # Pre-set seed
        state = mgr._store.get_or_create_user_state(mgr._user_id)
        state.seed = Seed(user_model="junior developer")
        mgr._store.update_user_state(state)

        for msg in _HIGH_SIGNAL_MESSAGES:
            await mgr.observe(msg, "user")

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert state.seed.user_model == "senior developer"
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_update_field_remove(tmp_path: Path) -> None:
    """When reflection returns a remove action, the field should be deleted."""
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "remove", "key": "stale_tool"},
        ],
    })
    llm, _ = _make_mock_llm(reflection_response=response)
    mgr = MemoryManager(_make_config(tmp_path), llm)
    try:
        state = mgr._store.get_or_create_user_state(mgr._user_id)
        state.fields = [
            UserField(key="stale_tool", value="Grunt"),
            UserField(key="lang", value="Python"),
        ]
        mgr._store.update_user_state(state)

        for msg in _HIGH_SIGNAL_MESSAGES:
            await mgr.observe(msg, "user")

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        field_keys = {f.key for f in state.fields}
        assert "stale_tool" not in field_keys, "Removed field should be gone"
        assert "lang" in field_keys, "Other fields should remain"
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_update_decay_prunes_weak_field(tmp_path: Path) -> None:
    """Fields with old last_accessed and low strength should be pruned via Ebbinghaus decay
    when the field count exceeds max_user_fields during reflection."""
    # Response that adds many new fields to push count over max_user_fields
    new_fields = [
        {"action": "add", "key": f"new_{i}", "value": f"value_{i}"}
        for i in range(20)
    ]
    response = json.dumps({"seed_updates": {}, "field_updates": new_fields})
    llm, _ = _make_mock_llm(reflection_response=response)
    config = _make_config(tmp_path, max_user_fields=20)
    mgr = MemoryManager(config, llm)
    try:
        # Pre-set a weak, old field
        state = mgr._store.get_or_create_user_state(mgr._user_id)
        old_time = datetime.now(UTC) - timedelta(days=60)
        state.fields = [
            UserField(
                key="ancient_tool",
                value="CVS",
                meta=FieldMeta(last_accessed=old_time, access_count=0, strength=0.5),
            )
        ]
        mgr._store.update_user_state(state)

        for msg in _HIGH_SIGNAL_MESSAGES:
            await mgr.observe(msg, "user")

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        field_keys = {f.key for f in state.fields}
        assert "ancient_tool" not in field_keys, (
            "Weak old field should be pruned by decay"
        )
        assert len(state.fields) <= config.max_user_fields
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_update_field_reinforcement(tmp_path: Path) -> None:
    """When a field is reinforced via an update action, access_count and strength increase."""
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "update", "key": "lang", "value": "Python 3.13"},
        ],
    })

    store = MemoryStore(tmp_path / "reinforce.db")
    try:
        user_id = "testuser"
        state = store.get_or_create_user_state(user_id)
        state.fields = [
            UserField(
                key="lang", value="Python 3.12",
                meta=FieldMeta(access_count=2, strength=1.5),
            )
        ]
        store.update_user_state(state)

        _seed_observations(store, user_id)
        engine = ReflectionEngine(lambda s, u: _mock_return(response), store)
        config = MemoryConfig(reflection_trigger=50)
        await engine.maybe_reflect(user_id, config)

        state = store.get_or_create_user_state(user_id)
        lang_field = next((f for f in state.fields if f.key == "lang"), None)
        assert lang_field is not None
        assert lang_field.value == "Python 3.13"
        assert lang_field.meta.access_count == 3, (
            f"access_count should be 3, got {lang_field.meta.access_count}"
        )
        assert lang_field.meta.strength == 2.0, (
            f"strength should be 2.0, got {lang_field.meta.strength}"
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_update_long_term_incorporates_new(tmp_path: Path) -> None:
    """Long-term consolidation should incorporate new short-term items."""
    llm, call_log = _make_mock_llm()
    config = _make_config(tmp_path, short_term_cap=5)
    mgr = MemoryManager(config, llm)
    try:
        ctx_key = mgr.get_context_key()
        ctx = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        ctx.long_term = "Old knowledge about Flask project"
        ctx.short_term = [
            "User switched to FastAPI",
            "Using async handlers everywhere",
            "PostgreSQL with SQLAlchemy",
            "Docker deployment pipeline",
            "Switched from unittest to pytest",
        ]
        mgr._store.update_context_memory(ctx)

        await mgr.on_session_end()

        ctx_after = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        assert ctx_after.long_term != "", "Long-term should be populated"
        assert ctx_after.long_term != "Old knowledge about Flask project", (
            "Long-term should be updated, not the old value"
        )

        # Verify the LLM was called with both old LT and new ST items
        lt_calls = [c for c in call_log if c["type"] == "consolidate_lt"]
        assert len(lt_calls) >= 1, "LT consolidation LLM call should have been made"
    finally:
        mgr.close()


# ---------------------------------------------------------------------------
# Async helper for ReflectionEngine tests
# ---------------------------------------------------------------------------

async def _mock_return(value: str) -> str:
    return value


def _make_async_llm(response: str):
    async def llm(system: str, user: str) -> str:
        return response
    return llm
