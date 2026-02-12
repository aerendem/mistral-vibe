"""Global/cross-context memory benchmark.

Verifies that user-level state (seed, fields) works across different
context_keys while context-level state (sensory, short-term, long-term)
remains properly isolated per context.

Run with: uv run pytest tests/memory/test_bench_global.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibe.core.memory.config import MemoryConfig
from vibe.core.memory.injection import MemoryInjector
from vibe.core.memory.models import FieldMeta, Observation, Seed, UserField
from vibe.core.memory.reflection import ReflectionEngine
from vibe.core.memory.storage import MemoryStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CTX_A = "project:/workspace/project-alpha"
CTX_B = "project:/workspace/project-beta"
USER_ID = "testuser"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    db = tmp_path / "global_bench.db"
    s = MemoryStore(db)
    yield s
    s.close()


def _seed_observations(
    store: MemoryStore, user_id: str, count: int = 10, importance: int = 8
) -> None:
    """Seed observations and accumulate importance above reflection trigger."""
    for i in range(count):
        store.add_observation(
            Observation(
                user_id=user_id,
                context_key=CTX_A,
                content=f"observation {i}",
                importance=importance,
                source_role="user",
            )
        )
    state = store.get_or_create_user_state(user_id)
    state.accumulated_importance = count * importance
    store.update_user_state(state)


_REFLECTION_RESPONSE = json.dumps({
    "seed_updates": {
        "user_model": "Full-stack developer",
        "affect": "collaborative",
    },
    "field_updates": [
        {"action": "add", "key": "lang", "value": "Python"},
        {"action": "add", "key": "editor", "value": "VS Code"},
    ],
})


def _make_reflection_engine(store: MemoryStore, response: str = _REFLECTION_RESPONSE) -> ReflectionEngine:
    async def mock_llm(system: str, user: str) -> str:
        return response

    return ReflectionEngine(mock_llm, store)


def _make_config(**overrides) -> MemoryConfig:
    defaults = dict(enabled=True, reflection_trigger=50)
    defaults.update(overrides)
    return MemoryConfig(**defaults)


# ===================================================================
# 1. USER STATE SHARED ACROSS CONTEXTS
# ===================================================================


def test_seed_persists_across_contexts(store: MemoryStore) -> None:
    """User seed set globally should be visible in memory blocks for any context."""
    state = store.get_or_create_user_state(USER_ID)
    state.seed = Seed(user_model="Full-stack developer")
    store.update_user_state(state)

    # Ensure both contexts exist
    store.get_or_create_context_memory(CTX_A, USER_ID)
    store.get_or_create_context_memory(CTX_B, USER_ID)

    injector = MemoryInjector(store)
    block_a = injector.build_memory_block(USER_ID, CTX_A)
    block_b = injector.build_memory_block(USER_ID, CTX_B)

    assert "Full-stack developer" in block_a, "Seed should appear in context A"
    assert "Full-stack developer" in block_b, "Seed should appear in context B"


def test_fields_visible_in_both_contexts(store: MemoryStore) -> None:
    """User fields should be accessible from any context_key."""
    state = store.get_or_create_user_state(USER_ID)
    state.fields = [
        UserField(key="lang", value="Python"),
        UserField(key="editor", value="VS Code"),
    ]
    store.update_user_state(state)

    store.get_or_create_context_memory(CTX_A, USER_ID)
    store.get_or_create_context_memory(CTX_B, USER_ID)

    injector = MemoryInjector(store)
    block_a = injector.build_memory_block(USER_ID, CTX_A)
    block_b = injector.build_memory_block(USER_ID, CTX_B)

    for block, label in [(block_a, "A"), (block_b, "B")]:
        assert "lang" in block, f"Field 'lang' should appear in context {label}"
        assert "Python" in block, f"Field value 'Python' should appear in context {label}"


# ===================================================================
# 2. CONTEXT MEMORY ISOLATED PER CONTEXT
# ===================================================================


def test_sensory_isolated_per_context(store: MemoryStore) -> None:
    """Sensory observations in context A should NOT appear in context B."""
    store.get_or_create_user_state(USER_ID)
    store.add_sensory(CTX_A, USER_ID, "Alpha-specific observation about the project")
    store.get_or_create_context_memory(CTX_B, USER_ID)

    injector = MemoryInjector(store)
    block_a = injector.build_memory_block(USER_ID, CTX_A)
    block_b = injector.build_memory_block(USER_ID, CTX_B)

    assert "Alpha-specific" in block_a, "Context A should have its sensory"
    assert "Alpha-specific" not in block_b, "Context B should NOT have A's sensory"


def test_short_term_isolated_per_context(store: MemoryStore) -> None:
    """Short-term points in one context should not leak to another."""
    store.get_or_create_user_state(USER_ID)

    ctx_a = store.get_or_create_context_memory(CTX_A, USER_ID)
    ctx_a.short_term = ["Alpha architecture decision"]
    store.update_context_memory(ctx_a)

    ctx_b = store.get_or_create_context_memory(CTX_B, USER_ID)
    ctx_b.short_term = ["Beta deployment plan"]
    store.update_context_memory(ctx_b)

    injector = MemoryInjector(store)
    block_a = injector.build_memory_block(USER_ID, CTX_A)
    block_b = injector.build_memory_block(USER_ID, CTX_B)

    assert "Alpha architecture" in block_a
    assert "Beta deployment" not in block_a, "A should not have B's short-term"
    assert "Beta deployment" in block_b
    assert "Alpha architecture" not in block_b, "B should not have A's short-term"


def test_long_term_isolated_per_context(store: MemoryStore) -> None:
    """Long-term knowledge in one context should not appear in another."""
    store.get_or_create_user_state(USER_ID)

    ctx_a = store.get_or_create_context_memory(CTX_A, USER_ID)
    ctx_a.long_term = "Alpha uses Flask with PostgreSQL"
    store.update_context_memory(ctx_a)

    ctx_b = store.get_or_create_context_memory(CTX_B, USER_ID)
    ctx_b.long_term = "Beta uses React with MongoDB"
    store.update_context_memory(ctx_b)

    injector = MemoryInjector(store)
    block_a = injector.build_memory_block(USER_ID, CTX_A)
    block_b = injector.build_memory_block(USER_ID, CTX_B)

    assert "Flask" in block_a
    assert "React" not in block_a, "A should not have B's long-term"
    assert "React" in block_b
    assert "Flask" not in block_b, "B should not have A's long-term"


# ===================================================================
# 3. CROSS-CONTEXT INTERACTIONS
# ===================================================================


@pytest.mark.asyncio
async def test_reflection_updates_seed_visible_cross_context(store: MemoryStore) -> None:
    """Reflection triggered by observations should update user seed,
    which is then visible when building the block for a different context."""
    store.get_or_create_user_state(USER_ID)
    store.get_or_create_context_memory(CTX_A, USER_ID)
    store.get_or_create_context_memory(CTX_B, USER_ID)

    _seed_observations(store, USER_ID)

    engine = _make_reflection_engine(store)
    config = _make_config()
    reflected = await engine.maybe_reflect(USER_ID, config)
    assert reflected, "Reflection should have fired"

    # Build block for CTX_B (reflection came from CTX_A observations)
    injector = MemoryInjector(store)
    block_b = injector.build_memory_block(USER_ID, CTX_B)

    assert "<seed>" in block_b, "Seed should appear in context B after reflection"
    assert "Full-stack developer" in block_b, (
        "Seed user_model from reflection should be visible in context B"
    )


def test_sensory_not_leaked_across_contexts(store: MemoryStore) -> None:
    """Sensory data added to context A should not appear in context B's storage."""
    store.get_or_create_user_state(USER_ID)
    store.add_sensory(CTX_A, USER_ID, "Secret project Alpha detail")

    ctx_b = store.get_or_create_context_memory(CTX_B, USER_ID)
    assert len(ctx_b.sensory) == 0, (
        "Context B sensory should be empty — A's data should not leak"
    )

    ctx_a = store.get_or_create_context_memory(CTX_A, USER_ID)
    assert len(ctx_a.sensory) == 1
    assert "Secret project Alpha" in ctx_a.sensory[0]


@pytest.mark.asyncio
async def test_field_reinforcement_visible_cross_context(store: MemoryStore) -> None:
    """When a field is reinforced by reflection (from context A observations),
    the increased strength should be visible in context B's memory block."""
    state = store.get_or_create_user_state(USER_ID)
    state.fields = [
        UserField(
            key="lang",
            value="Python",
            meta=FieldMeta(access_count=1, strength=1.0),
        )
    ]
    store.update_user_state(state)

    # Reflection response that updates the same field (triggers reinforcement)
    response = json.dumps({
        "seed_updates": {},
        "field_updates": [
            {"action": "update", "key": "lang", "value": "Python 3.12"},
        ],
    })

    store.get_or_create_context_memory(CTX_A, USER_ID)
    store.get_or_create_context_memory(CTX_B, USER_ID)

    _seed_observations(store, USER_ID)
    engine = _make_reflection_engine(store, response)
    await engine.maybe_reflect(USER_ID, _make_config())

    # Check reinforced metadata
    state = store.get_or_create_user_state(USER_ID)
    lang_field = next((f for f in state.fields if f.key == "lang"), None)
    assert lang_field is not None
    assert lang_field.meta.access_count == 2, (
        f"access_count should be 2 after reinforcement, got {lang_field.meta.access_count}"
    )
    assert lang_field.meta.strength == 1.5, (
        f"strength should be 1.5 after reinforcement, got {lang_field.meta.strength}"
    )

    # Verify field visible in context B
    injector = MemoryInjector(store)
    block_b = injector.build_memory_block(USER_ID, CTX_B)
    assert "Python 3.12" in block_b, (
        "Reinforced field should be visible in context B"
    )
