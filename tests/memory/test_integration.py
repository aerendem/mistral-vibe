from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibe.core.memory import MemoryManager, MemoryMetrics
from vibe.core.memory.config import MemoryConfig
from vibe.core.memory.storage import MemoryStore


async def _mock_llm_score_high(system: str, user: str) -> str:
    return "7"


async def _mock_llm_score_low(system: str, user: str) -> str:
    return "1"


@pytest.fixture
def memory_config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(
        enabled=True,
        db_path=str(tmp_path / "test_memory.db"),
    )


@pytest.fixture
def disabled_config() -> MemoryConfig:
    return MemoryConfig(enabled=False)


def test_disabled_manager_is_noop(disabled_config: MemoryConfig) -> None:
    mgr = MemoryManager(disabled_config, _mock_llm_score_high)
    assert not mgr.enabled
    assert mgr.get_memory_block() == ""
    mgr.close()


@pytest.mark.asyncio
async def test_disabled_observe_noop(disabled_config: MemoryConfig) -> None:
    mgr = MemoryManager(disabled_config, _mock_llm_score_high)
    await mgr.observe("test message")  # should not raise
    mgr.close()


@pytest.mark.asyncio
async def test_disabled_session_end_noop(disabled_config: MemoryConfig) -> None:
    mgr = MemoryManager(disabled_config, _mock_llm_score_high)
    await mgr.on_session_end()  # should not raise
    mgr.close()


@pytest.mark.asyncio
async def test_observe_stores_observation(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        await mgr.observe("I prefer Python over JavaScript", "user")
        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert state.accumulated_importance > 0
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_observe_below_threshold_not_stored(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_low)
    try:
        await mgr.observe("ok", "user")
        state = mgr._store.get_or_create_user_state(mgr._user_id)
        assert state.accumulated_importance == 0.0
    finally:
        mgr.close()


def test_get_memory_block_with_data(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        # Add some data directly
        state = mgr._store.get_or_create_user_state(mgr._user_id)
        from vibe.core.memory.models import Seed

        state.seed = Seed(user_model="Python developer")
        mgr._store.update_user_state(state)

        ctx_key = mgr.get_context_key()
        ctx = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        ctx.short_term = ["Uses pytest for testing"]
        mgr._store.update_context_memory(ctx)

        block = mgr.get_memory_block()
        assert "<memory>" in block
        assert "Python developer" in block
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_memory_failure_doesnt_crash(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        mgr._store.close()  # force DB to be closed
        # These should not raise — they should log warnings and return gracefully
        await mgr.observe("test")
        assert mgr.get_memory_block() == ""
        await mgr.on_session_end()
    except Exception:
        pytest.fail("Memory operations should not raise after DB closure")


def test_close_is_safe(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    mgr.close()
    mgr.close()  # double close should not raise


@pytest.mark.asyncio
async def test_on_session_end_processes_prestart_queue(
    memory_config: MemoryConfig,
) -> None:
    """Queued observations are processed even if worker starts at session end."""
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        message = "I prefer Python backend services with async architecture and tests"
        mgr.enqueue_observe(message, "user")
        await mgr.on_session_end()

        observations = mgr._store.get_pending_observations(mgr._user_id)
        assert any(o.content == message for o in observations)
    finally:
        await mgr.aclose()


def test_close_flushes_pending_queue_without_running_loop(
    tmp_path: Path,
) -> None:
    """Sync close drains queued observations when no loop is running."""
    db_path = tmp_path / "close_flush.db"
    mgr = MemoryManager(
        MemoryConfig(enabled=True, db_path=str(db_path)),
        _mock_llm_score_high,
    )
    message = "I prefer Python backend services with async architecture and tests"
    mgr.enqueue_observe(message, "user")
    mgr.close()

    with MemoryStore(db_path) as reloaded:
        row = reloaded._conn.execute(
            "SELECT observations_json FROM user_states WHERE user_id = ?",
            (mgr._user_id,),
        ).fetchone()
        assert row is not None
        observations = json.loads(row["observations_json"])
        assert any(o["content"] == message for o in observations)


# -- Trivial pre-filter tests --


def test_is_trivial_empty() -> None:
    assert MemoryManager._is_trivial("") is True
    assert MemoryManager._is_trivial("   ") is True


def test_is_trivial_common_phrases() -> None:
    for phrase in ["yes", "no", "ok", "sure", "thanks", "lgtm", "k", "y", "n"]:
        assert MemoryManager._is_trivial(phrase) is True, f"Expected '{phrase}' to be trivial"


def test_is_trivial_with_punctuation() -> None:
    assert MemoryManager._is_trivial("ok.") is True
    assert MemoryManager._is_trivial("sure!") is True
    assert MemoryManager._is_trivial("thanks,") is True


def test_is_trivial_case_insensitive() -> None:
    assert MemoryManager._is_trivial("OK") is True
    assert MemoryManager._is_trivial("LGTM") is True
    assert MemoryManager._is_trivial("Yes") is True


def test_is_trivial_long_message_not_trivial() -> None:
    assert MemoryManager._is_trivial("I prefer Python over JavaScript") is False


def test_is_trivial_short_but_unknown_not_trivial() -> None:
    assert MemoryManager._is_trivial("fix the bug") is False


@pytest.mark.asyncio
async def test_trivial_message_skips_scoring(memory_config: MemoryConfig) -> None:
    call_count = 0

    async def counting_llm(system: str, user: str) -> str:
        nonlocal call_count
        call_count += 1
        return "7"

    mgr = MemoryManager(memory_config, counting_llm)
    try:
        await mgr.observe("ok", "user")
        assert call_count == 0  # no LLM call for trivial message
        assert mgr.metrics.observations_skipped_trivial == 1
    finally:
        mgr.close()


# -- Metrics tests --


def test_metrics_defaults() -> None:
    m = MemoryMetrics()
    assert m.observations_stored == 0
    assert m.observations_skipped_trivial == 0
    assert m.observations_skipped_low_entropy == 0
    assert m.observations_skipped_dedup == 0
    assert m.observations_skipped_low_score == 0
    assert m.sensory_dedup_skipped == 0
    assert m.reflections_triggered == 0
    assert m.injections_served == 0
    assert m.injections_empty == 0
    assert m.audit_logs_pruned == 0
    assert m.stale_contexts_pruned == 0


@pytest.mark.asyncio
async def test_metrics_stored(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        await mgr.observe("I prefer using async patterns for all backend code", "user")
        assert mgr.metrics.observations_stored == 1
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_metrics_skipped_trivial(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        await mgr.observe("ok", "user")
        await mgr.observe("yes", "user")
        assert mgr.metrics.observations_skipped_trivial == 2
        assert mgr.metrics.observations_stored == 0
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_metrics_skipped_low_score(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_low)
    try:
        await mgr.observe("Something not trivial but low score for testing", "user")
        assert mgr.metrics.observations_skipped_low_score == 1
    finally:
        mgr.close()


def test_metrics_injections_served(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        from vibe.core.memory.models import Seed

        state = mgr._store.get_or_create_user_state(mgr._user_id)
        state.seed = Seed(user_model="developer")
        mgr._store.update_user_state(state)
        mgr._store.get_or_create_context_memory(mgr.get_context_key(), mgr._user_id)

        mgr.get_memory_block()
        assert mgr.metrics.injections_served == 1
    finally:
        mgr.close()


def test_metrics_injections_empty(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        mgr.get_memory_block()
        assert mgr.metrics.injections_empty == 1
    finally:
        mgr.close()


# -- Heuristic mode test --


@pytest.mark.asyncio
async def test_heuristic_scoring_mode(tmp_path: Path) -> None:
    config = MemoryConfig(
        enabled=True,
        db_path=str(tmp_path / "test_memory.db"),
        scoring_mode="heuristic",
    )

    call_count = 0

    async def counting_llm(system: str, user: str) -> str:
        nonlocal call_count
        call_count += 1
        return "7"

    mgr = MemoryManager(config, counting_llm)
    try:
        await mgr.observe(
            "I always prefer to use Python for backend architecture work", "user"
        )
        assert call_count == 0  # no LLM scoring calls
        assert mgr.metrics.observations_stored == 1
    finally:
        mgr.close()


# -- Config defaults test --


def test_config_defaults() -> None:
    config = MemoryConfig()
    assert config.enabled is False
    assert config.reflection_trigger == 50
    assert config.scoring_mode == "llm"
    assert config.observe_assistant is False
    assert config.importance_threshold == 3
    assert config.compress_storage is True
    assert config.dedup_sensory is False
    assert config.injection_budget_tokens == 500
    assert config.decay_prune_threshold == 0.1
    assert config.audit_retention_days == 90
    assert config.stale_context_days == 60


# -- Compression integration tests --


@pytest.mark.asyncio
async def test_entropy_filter_skips_boilerplate(memory_config: MemoryConfig) -> None:
    call_count = 0

    async def counting_llm(system: str, user: str) -> str:
        nonlocal call_count
        call_count += 1
        return "7"

    mgr = MemoryManager(memory_config, counting_llm)
    try:
        await mgr.observe("aaa bbb aaa bbb aaa bbb aaa bbb", "user")
        assert mgr.metrics.observations_skipped_low_entropy == 1
        assert call_count == 0  # no LLM call for low entropy
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_exact_dedup_skips_duplicate_scoring(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        msg = "I strongly prefer using Python for backend development"
        await mgr.observe(msg, "user")
        await mgr.observe(msg, "user")  # exact duplicate
        assert mgr.metrics.observations_stored == 1
        assert mgr.metrics.observations_skipped_dedup == 1
    finally:
        mgr.close()


@pytest.mark.asyncio
async def test_exact_dedup_different_role_not_deduped(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        msg = "I strongly prefer using Python for backend development"
        await mgr.observe(msg, "user")
        await mgr.observe(msg, "assistant")  # same text, different role
        assert mgr.metrics.observations_skipped_dedup == 0
        assert mgr.metrics.observations_stored == 2
    finally:
        mgr.close()


def test_sensory_consecutive_dedup(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        ctx_key = mgr.get_context_key()
        result1 = mgr._store.add_sensory(ctx_key, mgr._user_id, "hello world")
        result2 = mgr._store.add_sensory(ctx_key, mgr._user_id, "hello world")
        assert result1 is True
        assert result2 is False  # consecutive duplicate

        ctx = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        assert len(ctx.sensory) == 1
    finally:
        mgr.close()


def test_long_term_compression_round_trip(memory_config: MemoryConfig) -> None:
    mgr = MemoryManager(memory_config, _mock_llm_score_high)
    try:
        ctx_key = mgr.get_context_key()
        ctx = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        long_text = "x" * 300 + " some varied content to ensure compression works well " * 5
        ctx.long_term = long_text
        mgr._store.update_context_memory(ctx)

        loaded = mgr._store.get_or_create_context_memory(ctx_key, mgr._user_id)
        assert loaded.long_term == long_text
    finally:
        mgr.close()
