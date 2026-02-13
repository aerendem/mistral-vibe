"""Tests for the forgetting mechanism: pruning, budget enforcement, proactive decay."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from vibe.core.memory import MemoryManager, MemoryMetrics
from vibe.core.memory.config import MemoryConfig
from vibe.core.memory.decay import apply_decay, compute_retention, reinforce_field
from vibe.core.memory.models import FieldMeta, Observation, Seed, UserField, UserState
from vibe.core.memory.storage import MemoryStore


# -- Fixtures --


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    db = tmp_path / "test_forgetting.db"
    s = MemoryStore(db)
    yield s
    s.close()


@pytest.fixture
def memory_config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(
        enabled=True,
        db_path=str(tmp_path / "test_memory.db"),
        audit_retention_days=90,
        stale_context_days=60,
        decay_prune_threshold=0.1,
    )


async def _mock_llm(system: str, user: str) -> str:
    return "7"


# -- Storage-layer pruning tests --


def test_prune_audit_logs_by_age(store: MemoryStore) -> None:
    """Reflections and consolidations older than retention period are deleted."""
    store.log_reflection("user1", '{"obs": []}', '{"result": {}}')
    store.log_consolidation("ctx1", "user1", 5, "old", "new")

    # Manually backdate the records
    cutoff = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    store._conn.execute("UPDATE reflections SET created_at = ?", (cutoff,))
    store._conn.execute("UPDATE consolidations SET created_at = ?", (cutoff,))
    store._conn.commit()

    r_del, c_del = store.prune_audit_logs(retention_days=90)
    assert r_del == 1
    assert c_del == 1

    # Verify they're actually gone
    rows_r = store._conn.execute("SELECT * FROM reflections").fetchall()
    rows_c = store._conn.execute("SELECT * FROM consolidations").fetchall()
    assert len(rows_r) == 0
    assert len(rows_c) == 0


def test_prune_audit_logs_empty_table(store: MemoryStore) -> None:
    """Pruning on empty tables returns (0, 0) without error."""
    r_del, c_del = store.prune_audit_logs(retention_days=90)
    assert r_del == 0
    assert c_del == 0


def test_prune_audit_logs_keeps_recent(store: MemoryStore) -> None:
    """Recent audit logs are not pruned."""
    store.log_reflection("user1", '{"obs": []}', '{"result": {}}')
    store.log_consolidation("ctx1", "user1", 5, "old", "new")

    r_del, c_del = store.prune_audit_logs(retention_days=90)
    assert r_del == 0
    assert c_del == 0


def test_get_stale_contexts(store: MemoryStore) -> None:
    """Only stale contexts with volatile data are returned."""
    stale_ctx = store.get_or_create_context_memory("project:stale", "user1")
    stale_ctx.sensory = ["old volatile"]
    store.update_context_memory(stale_ctx)

    fresh_ctx = store.get_or_create_context_memory("project:fresh", "user1")
    fresh_ctx.sensory = ["new volatile"]
    store.update_context_memory(fresh_ctx)

    # Backdate the stale one
    cutoff = (datetime.now(UTC) - timedelta(days=70)).isoformat()
    store._conn.execute(
        "UPDATE context_memories SET updated_at = ? WHERE context_key = 'project:stale'",
        (cutoff,),
    )
    store._conn.commit()

    stale = store.get_stale_contexts(stale_days=60)
    assert len(stale) == 1
    assert stale[0] == ("project:stale", "user1")


def test_prune_context_volatile_preserves_long_term(store: MemoryStore) -> None:
    """Pruning volatile data keeps long_term intact."""
    ctx = store.get_or_create_context_memory("project:prune", "user1")
    ctx.sensory = ["obs1", "obs2"]
    ctx.short_term = ["point1"]
    ctx.long_term = "This is the summary."
    store.update_context_memory(ctx)

    store.prune_context_volatile("project:prune", "user1")

    loaded = store.get_or_create_context_memory("project:prune", "user1")
    assert loaded.sensory == []
    assert loaded.short_term == []
    assert loaded.long_term == "This is the summary."


def test_prune_context_volatile_skips_already_empty(store: MemoryStore) -> None:
    """Pruning an already-empty context does not update updated_at."""
    ctx = store.get_or_create_context_memory("project:empty", "user1")
    assert ctx.sensory == []
    assert ctx.short_term == []

    original_row = store._conn.execute(
        "SELECT updated_at FROM context_memories WHERE context_key='project:empty'"
    ).fetchone()
    original_ts = original_row["updated_at"]

    store.prune_context_volatile("project:empty", "user1")

    updated_row = store._conn.execute(
        "SELECT updated_at FROM context_memories WHERE context_key='project:empty'"
    ).fetchone()
    assert updated_row["updated_at"] == original_ts


def test_run_maintenance_no_error(store: MemoryStore) -> None:
    """PRAGMA optimize runs without error."""
    store.run_maintenance()  # should not raise


# -- Storage hardening tests --


def test_busy_timeout_pragma(store: MemoryStore) -> None:
    """Verify busy_timeout is set to 200ms."""
    row = store._conn.execute("PRAGMA busy_timeout").fetchone()
    assert row[0] == 200


def test_context_manager(tmp_path: Path) -> None:
    """MemoryStore can be used as a context manager."""
    db = tmp_path / "ctx_mgr.db"
    with MemoryStore(db) as s:
        s.get_or_create_user_state("user1")
    # After exiting, conn should be closed
    assert s._conn is None


def test_store_double_close(tmp_path: Path) -> None:
    """Double close should not raise."""
    db = tmp_path / "double_close.db"
    s = MemoryStore(db)
    s.close()
    s.close()  # no error


def test_indexes_created(store: MemoryStore) -> None:
    """Verify secondary indexes exist."""
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    names = {row["name"] for row in rows}
    assert "idx_reflections_user_created" in names
    assert "idx_consolidations_user_context" in names
    assert "idx_context_memories_updated" in names


def test_concurrent_writes_from_threads(tmp_path: Path) -> None:
    """Multiple threads writing observations should not error."""
    db = tmp_path / "concurrent.db"
    s = MemoryStore(db)
    s.get_or_create_user_state("user1")

    errors: list[Exception] = []

    def writer(thread_id: int) -> None:
        try:
            for i in range(50):
                s.add_observation(
                    Observation(
                        user_id="user1",
                        context_key=f"ctx:{thread_id}",
                        content=f"thread{thread_id}_msg{i}",
                        importance=5,
                        source_role="user",
                    )
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    s.close()


# -- Long-term budget enforcement tests --


@pytest.mark.asyncio
async def test_long_term_budget_enforced() -> None:
    """When LLM returns oversized long-term, it's truncated to budget."""
    from vibe.core.memory.consolidation import ConsolidationEngine

    oversized = "A" * 500 + ". " + "B" * 500 + ". " + "C" * 500

    async def mock_llm(system: str, user: str) -> str:
        return oversized

    store_mock = AsyncMock()
    engine = ConsolidationEngine(mock_llm, store_mock)

    result = await engine._consolidate_long_term(
        current_lt="old stuff",
        new_items=["item1", "item2"],
        context_key="project:test",
        budget_chars=600,
    )

    assert len(result) <= 600
    # Should truncate at sentence boundary
    assert result.endswith(".")


@pytest.mark.asyncio
async def test_long_term_budget_not_triggered_when_small() -> None:
    """When LLM returns within budget, no truncation occurs."""
    from vibe.core.memory.consolidation import ConsolidationEngine

    small_result = "This is a concise summary."

    async def mock_llm(system: str, user: str) -> str:
        return small_result

    store_mock = AsyncMock()
    engine = ConsolidationEngine(mock_llm, store_mock)

    result = await engine._consolidate_long_term(
        current_lt="old",
        new_items=["new"],
        context_key="project:test",
        budget_chars=1000,
    )

    assert result == small_result


# -- Proactive decay tests --


def test_proactive_decay_removes_stale_fields_below_cap() -> None:
    """Decay should prune stale fields even when count is below max_user_fields."""
    now = datetime.now(UTC)
    old = now - timedelta(days=365)

    fields = [
        UserField(
            key="stale", value="old info",
            meta=FieldMeta(last_accessed=old, access_count=1, strength=1.0),
        ),
        UserField(
            key="fresh", value="recent info",
            meta=FieldMeta(last_accessed=now, access_count=3, strength=2.5),
        ),
    ]

    result = apply_decay(fields, now, prune_threshold=0.1)
    # Stale field should have been pruned (retention = e^(-365/1.0) ~ 0)
    assert len(result) == 1
    assert result[0].key == "fresh"


def test_retention_formula() -> None:
    """Verify Ebbinghaus retention calculation."""
    now = datetime.now(UTC)
    meta = FieldMeta(last_accessed=now - timedelta(days=1), strength=1.0)
    retention = compute_retention(meta, now)
    # e^(-1/1) ~ 0.368
    assert 0.36 < retention < 0.38

    # High strength means slower decay
    meta_strong = FieldMeta(last_accessed=now - timedelta(days=1), strength=10.0)
    retention_strong = compute_retention(meta_strong, now)
    # e^(-1/10) ~ 0.905
    assert 0.90 < retention_strong < 0.91


def test_reinforce_increases_strength() -> None:
    """Reinforcing a field increases strength and updates access time."""
    now = datetime.now(UTC)
    field = UserField(
        key="lang", value="Python",
        meta=FieldMeta(last_accessed=now - timedelta(days=5), access_count=1, strength=1.0),
    )

    reinforced = reinforce_field(field, now)
    assert reinforced.meta.strength == 1.5
    assert reinforced.meta.access_count == 2
    assert reinforced.meta.last_accessed == now


# -- MemoryManager forgetting orchestration tests --


@pytest.mark.asyncio
async def test_run_forgetting_noop_when_disabled() -> None:
    """run_forgetting is a no-op when memory is disabled."""
    mgr = MemoryManager(MemoryConfig(enabled=False), _mock_llm)
    await mgr.run_forgetting()  # should not raise
    mgr.close()


@pytest.mark.asyncio
async def test_run_forgetting_prunes_old_data(memory_config: MemoryConfig) -> None:
    """run_forgetting prunes old audit logs and stale contexts."""
    mgr = MemoryManager(memory_config, _mock_llm)
    try:
        store = mgr._store

        # Add audit log and backdate it
        store.log_reflection("user1", '{"obs": []}', '{"result": {}}')
        cutoff = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        store._conn.execute("UPDATE reflections SET created_at = ?", (cutoff,))
        store._conn.commit()

        await mgr.run_forgetting()

        assert mgr.metrics.audit_logs_pruned >= 1

        # Verify the reflection was pruned
        rows = store._conn.execute("SELECT * FROM reflections").fetchall()
        assert len(rows) == 0
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_run_forgetting_maintenance_idempotent(memory_config: MemoryConfig) -> None:
    """Maintenance runs at most once per 24h."""
    mgr = MemoryManager(memory_config, _mock_llm)
    try:
        await mgr.run_forgetting()
        assert mgr._last_maintenance is not None
        first_maintenance = mgr._last_maintenance

        await mgr.run_forgetting()
        # Should not have re-run maintenance (same timestamp)
        assert mgr._last_maintenance == first_maintenance
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_run_forgetting_counts_only_actual_context_prunes(
    memory_config: MemoryConfig,
) -> None:
    """stale_contexts_pruned counts only rows actually updated."""
    mgr = MemoryManager(memory_config, _mock_llm)
    try:
        store = mgr._store

        stale_full = store.get_or_create_context_memory("project:stale-full", "user1")
        stale_full.sensory = ["volatile"]
        store.update_context_memory(stale_full)
        store.get_or_create_context_memory("project:stale-empty", "user1")

        cutoff = (datetime.now(UTC) - timedelta(days=70)).isoformat()
        store._conn.execute(
            "UPDATE context_memories SET updated_at = ? WHERE context_key = ?",
            (cutoff, "project:stale-full"),
        )
        store._conn.execute(
            "UPDATE context_memories SET updated_at = ? WHERE context_key = ?",
            (cutoff, "project:stale-empty"),
        )
        store._conn.commit()

        await mgr.run_forgetting()

        assert mgr.metrics.stale_contexts_pruned == 1
    finally:
        mgr.close()
