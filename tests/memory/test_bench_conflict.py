"""MemoryAgentBench benchmark: conflict resolution and selective forgetting.

Tests that the memory system correctly handles fact updates, contradictions,
and field lifecycle. Adapted from MemoryAgentBench (ICLR 2026) methodology
focusing on selective forgetting and conflict resolution.

Run with: uv run pytest tests/memory/test_bench_conflict.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibe.core.memory import MemoryManager
from vibe.core.memory.config import MemoryConfig
from vibe.core.memory.consolidation import ConsolidationEngine
from vibe.core.memory.models import FieldMeta, Observation, Seed, UserField
from vibe.core.memory.reflection import ReflectionEngine
from vibe.core.memory.storage import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_observations(
    store: MemoryStore, user_id: str, count: int = 10, importance: int = 8
) -> None:
    """Seed observations and accumulate importance to exceed reflection trigger."""
    for i in range(count):
        store.add_observation(
            Observation(
                user_id=user_id,
                context_key="project:test",
                content=f"observation {i}",
                importance=importance,
                source_role="user",
            )
        )
    state = store.get_or_create_user_state(user_id)
    state.accumulated_importance = count * importance
    store.update_user_state(state)


def _make_engine(store: MemoryStore, response: str) -> ReflectionEngine:
    async def mock_llm(system: str, user: str) -> str:
        return response

    return ReflectionEngine(mock_llm, store)


def _make_config(**overrides) -> MemoryConfig:
    defaults = dict(enabled=True, reflection_trigger=50)
    defaults.update(overrides)
    return MemoryConfig(**defaults)


def _make_sequential_mock_llm(reflection_responses: list[str]):
    """Mock LLM that returns successive reflection responses on each call."""
    call_idx = {"reflection": 0}

    async def mock_llm(system: str, user: str) -> str:
        if "Rate the following" in system:
            return "7"
        if "reflection engine" in system.lower():
            idx = call_idx["reflection"]
            call_idx["reflection"] = idx + 1
            if idx < len(reflection_responses):
                return reflection_responses[idx]
            return json.dumps({"seed_updates": {}, "field_updates": []})
        if "Compress" in user or "key points" in user.lower():
            return json.dumps(["Compressed point"])
        if "Rewrite" in user or "long-term" in user.lower():
            return "Updated long-term knowledge."
        return "5"

    return mock_llm


USER_ID = "testuser"

# High-signal messages for triggering reflection via MemoryManager
_HIGH_SIGNAL = [
    "I always use Python 3.12 and prefer async patterns for all work",
    "Our project uses FastAPI with SQLAlchemy as the ORM layer",
    "We follow the repository pattern and never use raw SQL anywhere",
    "The team convention is pydantic models for all API schemas",
    "I prefer to write integration tests first, outside-in TDD approach",
    "Our CI runs on GitHub Actions with 80% coverage requirement",
    "The database is PostgreSQL 16 on AWS RDS with read replicas",
    "I never use print debugging - always structured logging with structlog",
    "We deploy with Docker containers and Kubernetes orchestration always",
    "Our architecture follows clean architecture with dependency injection",
]


# ===================================================================
# TESTS
# ===================================================================


@pytest.mark.asyncio
async def test_fact_update_flask_to_fastapi(tmp_path: Path) -> None:
    """After updating from Flask to FastAPI, only FastAPI should remain in fields."""
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "update", "key": "framework", "value": "FastAPI"},
        ],
    })
    store = MemoryStore(tmp_path / "conflict.db")
    try:
        state = store.get_or_create_user_state(USER_ID)
        state.fields = [UserField(key="framework", value="Flask")]
        store.update_user_state(state)

        _seed_observations(store, USER_ID)
        engine = _make_engine(store, response)
        await engine.maybe_reflect(USER_ID, _make_config())

        state = store.get_or_create_user_state(USER_ID)
        framework = next((f for f in state.fields if f.key == "framework"), None)
        assert framework is not None
        assert framework.value == "FastAPI"
        assert not any(f.value == "Flask" for f in state.fields), (
            "Old value 'Flask' should not remain"
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_seed_update_junior_to_senior(tmp_path: Path) -> None:
    """When user_model changes from junior to senior, the seed should reflect it."""
    response = json.dumps({
        "seed_updates": {"user_model": "senior developer"},
        "field_updates": [],
    })
    store = MemoryStore(tmp_path / "conflict.db")
    try:
        state = store.get_or_create_user_state(USER_ID)
        state.seed = Seed(user_model="junior developer")
        store.update_user_state(state)

        _seed_observations(store, USER_ID)
        engine = _make_engine(store, response)
        await engine.maybe_reflect(USER_ID, _make_config())

        state = store.get_or_create_user_state(USER_ID)
        assert state.seed.user_model == "senior developer"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_field_removal(tmp_path: Path) -> None:
    """When a field is removed, it disappears while others remain."""
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "remove", "key": "database"},
        ],
    })
    store = MemoryStore(tmp_path / "conflict.db")
    try:
        state = store.get_or_create_user_state(USER_ID)
        state.fields = [
            UserField(key="database", value="MySQL"),
            UserField(key="lang", value="Python"),
        ]
        store.update_user_state(state)

        _seed_observations(store, USER_ID)
        engine = _make_engine(store, response)
        await engine.maybe_reflect(USER_ID, _make_config())

        state = store.get_or_create_user_state(USER_ID)
        field_keys = {f.key for f in state.fields}
        assert "database" not in field_keys, "Removed field should be gone"
        assert "lang" in field_keys, "Other field should remain"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_contradictory_same_session(tmp_path: Path) -> None:
    """When contradictory preferences are given in the same session,
    the reflection resolution should prevail (mock resolves to spaces)."""
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "add", "key": "indentation", "value": "spaces"},
        ],
    })
    config = MemoryConfig(
        enabled=True,
        db_path=str(tmp_path / "conflict.db"),
        reflection_trigger=50,
        scoring_mode="heuristic",
    )
    llm = _make_sequential_mock_llm([response])
    mgr = MemoryManager(config, llm)
    try:
        # Observe contradictory messages
        await mgr.observe("I prefer tabs for indentation in all my code always", "user")
        await mgr.observe("Actually, I prefer spaces for indentation in all my code always", "user")
        # Observe more to trigger reflection
        for msg in _HIGH_SIGNAL:
            await mgr.observe(msg, "user")

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        indent_field = next((f for f in state.fields if f.key == "indentation"), None)
        assert indent_field is not None
        assert indent_field.value == "spaces", (
            f"Resolution should be 'spaces', got {indent_field.value}"
        )
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_multi_field_update(tmp_path: Path) -> None:
    """When a user changes language, multiple related fields should update."""
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "update", "key": "lang", "value": "Rust"},
            {"action": "update", "key": "ecosystem", "value": "cargo"},
        ],
    })
    store = MemoryStore(tmp_path / "conflict.db")
    try:
        state = store.get_or_create_user_state(USER_ID)
        state.fields = [
            UserField(key="lang", value="Python"),
            UserField(key="ecosystem", value="pip"),
        ]
        store.update_user_state(state)

        _seed_observations(store, USER_ID)
        engine = _make_engine(store, response)
        await engine.maybe_reflect(USER_ID, _make_config())

        state = store.get_or_create_user_state(USER_ID)
        fields = {f.key: f.value for f in state.fields}
        assert fields["lang"] == "Rust"
        assert fields["ecosystem"] == "cargo"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_temporal_ordering(tmp_path: Path) -> None:
    """When the same field is updated in sequential reflections,
    the later update should be the final value."""
    response1 = json.dumps({
        "seed_updates": {},
        "field_updates": [{"action": "add", "key": "db", "value": "PostgreSQL 14"}],
    })
    response2 = json.dumps({
        "seed_updates": {},
        "field_updates": [{"action": "update", "key": "db", "value": "PostgreSQL 16"}],
    })

    store = MemoryStore(tmp_path / "conflict.db")
    try:
        store.get_or_create_user_state(USER_ID)

        # First reflection: add db=14
        _seed_observations(store, USER_ID)
        engine1 = _make_engine(store, response1)
        await engine1.maybe_reflect(USER_ID, _make_config())

        state = store.get_or_create_user_state(USER_ID)
        db_field = next((f for f in state.fields if f.key == "db"), None)
        assert db_field is not None
        assert db_field.value == "PostgreSQL 14"

        # Second reflection: update db=16
        _seed_observations(store, USER_ID)
        engine2 = _make_engine(store, response2)
        await engine2.maybe_reflect(USER_ID, _make_config())

        state = store.get_or_create_user_state(USER_ID)
        db_field = next((f for f in state.fields if f.key == "db"), None)
        assert db_field is not None
        assert db_field.value == "PostgreSQL 16"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_partial_version_update(tmp_path: Path) -> None:
    """A version change should update the value without duplicating the field."""
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "update", "key": "db_version", "value": "PostgreSQL 16"},
        ],
    })
    store = MemoryStore(tmp_path / "conflict.db")
    try:
        state = store.get_or_create_user_state(USER_ID)
        state.fields = [UserField(key="db_version", value="PostgreSQL 14")]
        store.update_user_state(state)

        _seed_observations(store, USER_ID)
        engine = _make_engine(store, response)
        await engine.maybe_reflect(USER_ID, _make_config())

        state = store.get_or_create_user_state(USER_ID)
        db_fields = [f for f in state.fields if f.key == "db_version"]
        assert len(db_fields) == 1, (
            f"Should have exactly 1 db_version field, got {len(db_fields)}"
        )
        assert db_fields[0].value == "PostgreSQL 16"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_non_conflicting_coexist(tmp_path: Path) -> None:
    """Non-conflicting facts should both be preserved after reflection."""
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "add", "key": "lang_secondary", "value": "TypeScript"},
        ],
    })
    store = MemoryStore(tmp_path / "conflict.db")
    try:
        state = store.get_or_create_user_state(USER_ID)
        state.fields = [UserField(key="lang_primary", value="Python")]
        store.update_user_state(state)

        _seed_observations(store, USER_ID)
        engine = _make_engine(store, response)
        await engine.maybe_reflect(USER_ID, _make_config())

        state = store.get_or_create_user_state(USER_ID)
        fields = {f.key: f.value for f in state.fields}
        assert "lang_primary" in fields, "Primary lang should remain"
        assert fields["lang_primary"] == "Python"
        assert "lang_secondary" in fields, "Secondary lang should be added"
        assert fields["lang_secondary"] == "TypeScript"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_consolidation_preserves_updates(tmp_path: Path) -> None:
    """After ST-to-LT consolidation, the updated long-term should reflect
    the new information from short-term."""
    async def mock_llm(system: str, user: str) -> str:
        if "Compress" in user or "key points" in user.lower():
            return json.dumps(["Migrated to FastAPI", "Using async handlers"])
        if "Rewrite" in user or "long-term" in user.lower():
            return "FastAPI project with async handlers and PostgreSQL backend."
        return ""

    store = MemoryStore(tmp_path / "conflict.db")
    try:
        store.get_or_create_user_state(USER_ID)
        ctx = store.get_or_create_context_memory("project:test", USER_ID)
        ctx.long_term = "Flask project with synchronous handlers."
        ctx.short_term = [
            "User migrated from Flask to FastAPI",
            "All handlers now use async/await",
            "PostgreSQL connection uses asyncpg",
            "Docker deployment updated for FastAPI",
            "Test suite migrated to pytest-asyncio",
        ]
        store.update_context_memory(ctx)

        config = _make_config(short_term_cap=5)
        engine = ConsolidationEngine(mock_llm, store)
        await engine.run_full_consolidation("project:test", USER_ID, config)

        ctx_after = store.get_or_create_context_memory("project:test", USER_ID)
        assert len(ctx_after.short_term) == 0, "Short-term should be cleared"
        assert "FastAPI" in ctx_after.long_term, (
            "Long-term should reflect the migration to FastAPI"
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_field_count_stable_on_repeated_updates(tmp_path: Path) -> None:
    """Repeatedly updating the same field should not increase field count."""
    store = MemoryStore(tmp_path / "conflict.db")
    try:
        state = store.get_or_create_user_state(USER_ID)
        state.fields = [UserField(key="lang", value="Python")]
        store.update_user_state(state)

        values = ["JavaScript", "TypeScript", "Rust"]
        for val in values:
            response = json.dumps({
                "seed_updates": {},
                "field_updates": [
                    {"action": "update", "key": "lang", "value": val},
                ],
            })
            _seed_observations(store, USER_ID)
            engine = _make_engine(store, response)
            await engine.maybe_reflect(USER_ID, _make_config())

        state = store.get_or_create_user_state(USER_ID)
        lang_fields = [f for f in state.fields if f.key == "lang"]
        assert len(lang_fields) == 1, (
            f"Should have exactly 1 lang field after 3 updates, got {len(lang_fields)}"
        )
        assert lang_fields[0].value == "Rust", "Value should be the last update"
    finally:
        store.close()
