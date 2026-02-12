from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibe.core.memory.config import MemoryConfig
from vibe.core.memory.consolidation import ConsolidationEngine
from vibe.core.memory.storage import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    db = tmp_path / "test_memory.db"
    s = MemoryStore(db)
    yield s
    s.close()


def _make_engine(
    store: MemoryStore,
    st_response: str = "[]",
    lt_response: str = "",
    should_raise: bool = False,
) -> ConsolidationEngine:
    async def mock_llm(system: str, user: str) -> str:
        if should_raise:
            raise RuntimeError("LLM failed")
        # Detect call type from prompt content
        if "Compress" in user or "key points" in system:
            return st_response
        return lt_response

    return ConsolidationEngine(mock_llm, store)


@pytest.mark.asyncio
async def test_no_consolidation_below_caps(store: MemoryStore) -> None:
    ctx = store.get_or_create_context_memory("project:test", "user1")
    ctx.sensory = ["obs1", "obs2"]
    ctx.short_term = ["pt1"]
    store.update_context_memory(ctx)

    engine = _make_engine(store)
    config = MemoryConfig(enabled=True, sensory_cap=50, short_term_cap=15)

    await engine.run_full_consolidation("project:test", "user1", config)

    ctx = store.get_or_create_context_memory("project:test", "user1")
    assert len(ctx.sensory) == 2  # unchanged
    assert len(ctx.short_term) == 1  # unchanged


@pytest.mark.asyncio
async def test_sensory_to_short_term(store: MemoryStore) -> None:
    ctx = store.get_or_create_context_memory("project:test", "user1")
    ctx.sensory = [f"obs_{i}" for i in range(50)]
    store.update_context_memory(ctx)

    st_response = json.dumps(["Key point A", "Key point B"])
    engine = _make_engine(store, st_response=st_response)
    config = MemoryConfig(enabled=True, sensory_cap=50, short_term_cap=15)

    await engine.run_full_consolidation("project:test", "user1", config)

    ctx = store.get_or_create_context_memory("project:test", "user1")
    assert len(ctx.sensory) == 0  # cleared
    assert "Key point A" in ctx.short_term
    assert "Key point B" in ctx.short_term


@pytest.mark.asyncio
async def test_short_term_to_long_term(store: MemoryStore) -> None:
    ctx = store.get_or_create_context_memory("project:test", "user1")
    ctx.short_term = [f"point_{i}" for i in range(15)]
    store.update_context_memory(ctx)

    lt_response = "This is a Python web project using FastAPI with async patterns."
    engine = _make_engine(store, lt_response=lt_response)
    config = MemoryConfig(enabled=True, sensory_cap=50, short_term_cap=15)

    await engine.run_full_consolidation("project:test", "user1", config)

    ctx = store.get_or_create_context_memory("project:test", "user1")
    assert len(ctx.short_term) == 0  # absorbed
    assert "FastAPI" in ctx.long_term
    assert ctx.consolidation_count == 1


@pytest.mark.asyncio
async def test_both_passes_in_sequence(store: MemoryStore) -> None:
    ctx = store.get_or_create_context_memory("project:test", "user1")
    # Enough sensory for pass 1 AND pre-existing short-term at the cap boundary
    ctx.sensory = [f"obs_{i}" for i in range(50)]
    ctx.short_term = [f"existing_{i}" for i in range(13)]
    store.update_context_memory(ctx)

    st_response = json.dumps(["Compressed A", "Compressed B"])
    lt_response = "Comprehensive summary of all knowledge."
    engine = _make_engine(store, st_response=st_response, lt_response=lt_response)
    config = MemoryConfig(enabled=True, sensory_cap=50, short_term_cap=15)

    await engine.run_full_consolidation("project:test", "user1", config)

    ctx = store.get_or_create_context_memory("project:test", "user1")
    assert len(ctx.sensory) == 0
    assert len(ctx.short_term) == 0
    assert "Comprehensive" in ctx.long_term


@pytest.mark.asyncio
async def test_sensory_llm_failure_is_safe(store: MemoryStore) -> None:
    ctx = store.get_or_create_context_memory("project:test", "user1")
    ctx.sensory = [f"obs_{i}" for i in range(50)]
    store.update_context_memory(ctx)

    engine = _make_engine(store, should_raise=True)
    config = MemoryConfig(enabled=True, sensory_cap=50, short_term_cap=15)

    await engine.run_full_consolidation("project:test", "user1", config)

    ctx = store.get_or_create_context_memory("project:test", "user1")
    assert len(ctx.sensory) == 50  # unchanged on failure


@pytest.mark.asyncio
async def test_consolidation_logs_to_table(store: MemoryStore) -> None:
    ctx = store.get_or_create_context_memory("project:test", "user1")
    ctx.short_term = [f"point_{i}" for i in range(15)]
    store.update_context_memory(ctx)

    lt_response = "New knowledge summary."
    engine = _make_engine(store, lt_response=lt_response)
    config = MemoryConfig(enabled=True, sensory_cap=50, short_term_cap=15)

    await engine.run_full_consolidation("project:test", "user1", config)

    rows = store._conn.execute("SELECT * FROM consolidations").fetchall()
    assert len(rows) == 1
    assert rows[0]["new_long_term"] == "New knowledge summary."


def test_parse_points_valid_json() -> None:
    raw = '["point 1", "point 2"]'
    assert ConsolidationEngine._parse_points(raw) == ["point 1", "point 2"]


def test_parse_points_markdown_fenced() -> None:
    raw = '```json\n["point A"]\n```'
    assert ConsolidationEngine._parse_points(raw) == ["point A"]


def test_parse_points_garbage() -> None:
    assert ConsolidationEngine._parse_points("not json") == []


def test_parse_points_non_list() -> None:
    assert ConsolidationEngine._parse_points('{"key": "value"}') == []


def test_parse_points_filters_empty() -> None:
    raw = '["good", "", "also good"]'
    assert ConsolidationEngine._parse_points(raw) == ["good", "also good"]
