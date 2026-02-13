from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vibe.core.memory.models import (
    ContextMemory,
    FieldMeta,
    Observation,
    Seed,
    UserField,
    UserState,
)


class TestSeed:
    def test_defaults(self) -> None:
        seed = Seed()
        assert seed.affect == ""
        assert seed.trust == ""
        assert seed.intention == ""
        assert seed.salient == ""
        assert seed.user_model == ""

    def test_construction(self) -> None:
        seed = Seed(affect="focused", user_model="senior dev")
        assert seed.affect == "focused"
        assert seed.user_model == "senior dev"

    def test_frozen(self) -> None:
        seed = Seed(affect="calm")
        with pytest.raises(ValidationError):
            seed.affect = "angry"

    def test_serialization_roundtrip(self) -> None:
        seed = Seed(affect="collaborative", trust="high", user_model="backend dev")
        data = seed.model_dump()
        restored = Seed(**data)
        assert restored == seed

    def test_json_roundtrip(self) -> None:
        seed = Seed(intention="learning", salient="Python expert")
        json_str = seed.model_dump_json()
        restored = Seed.model_validate_json(json_str)
        assert restored == seed


class TestFieldMeta:
    def test_defaults(self) -> None:
        meta = FieldMeta()
        assert meta.access_count == 0
        assert meta.strength == 1.0
        assert meta.last_accessed is not None

    def test_custom_values(self) -> None:
        now = datetime.now(UTC)
        meta = FieldMeta(last_accessed=now, access_count=5, strength=3.0)
        assert meta.access_count == 5
        assert meta.strength == 3.0
        assert meta.last_accessed == now


class TestUserField:
    def test_construction(self) -> None:
        field = UserField(key="lang", value="Python")
        assert field.key == "lang"
        assert field.value == "Python"
        assert field.meta.access_count == 0

    def test_with_meta(self) -> None:
        meta = FieldMeta(access_count=10, strength=5.0)
        field = UserField(key="editor", value="Neovim", meta=meta)
        assert field.meta.access_count == 10
        assert field.meta.strength == 5.0

    def test_serialization_roundtrip(self) -> None:
        field = UserField(key="os", value="Linux", meta=FieldMeta(access_count=3))
        data = field.model_dump(mode="json")
        restored = UserField(**data)
        assert restored.key == field.key
        assert restored.value == field.value
        assert restored.meta.access_count == field.meta.access_count


class TestUserState:
    def test_defaults(self) -> None:
        state = UserState(user_id="test")
        assert state.user_id == "test"
        assert state.seed == Seed()
        assert state.fields == []
        assert state.accumulated_importance == 0.0

    def test_with_fields(self) -> None:
        state = UserState(
            user_id="user1",
            seed=Seed(affect="focused"),
            fields=[
                UserField(key="lang", value="Python"),
                UserField(key="editor", value="Vim"),
            ],
            accumulated_importance=42.5,
        )
        assert len(state.fields) == 2
        assert state.accumulated_importance == 42.5
        assert state.seed.affect == "focused"

    def test_mutable_fields(self) -> None:
        state = UserState(user_id="test")
        state.accumulated_importance = 100.0
        assert state.accumulated_importance == 100.0
        state.fields.append(UserField(key="new", value="field"))
        assert len(state.fields) == 1


class TestObservation:
    def test_defaults(self) -> None:
        obs = Observation()
        assert obs.id is None
        assert obs.user_id == ""
        assert obs.importance == 0

    def test_construction(self) -> None:
        obs = Observation(
            user_id="user1",
            context_key="project:test",
            content="test message",
            importance=7,
            source_role="user",
        )
        assert obs.user_id == "user1"
        assert obs.importance == 7


class TestContextMemory:
    def test_defaults(self) -> None:
        ctx = ContextMemory(context_key="project:test", user_id="user1")
        assert ctx.sensory == []
        assert ctx.short_term == []
        assert ctx.long_term == ""
        assert ctx.version == 0
        assert ctx.consolidation_count == 0

    def test_mutable_lists(self) -> None:
        ctx = ContextMemory(context_key="project:test", user_id="user1")
        ctx.sensory.append("obs1")
        ctx.short_term.append("point1")
        ctx.long_term = "summary"
        assert len(ctx.sensory) == 1
        assert len(ctx.short_term) == 1
        assert ctx.long_term == "summary"


