from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vibe.core.memory.decay import apply_decay, compute_retention, reinforce_field
from vibe.core.memory.models import FieldMeta, UserField


def test_compute_retention_fresh() -> None:
    now = datetime.now(UTC)
    meta = FieldMeta(last_accessed=now, access_count=5, strength=2.0)
    r = compute_retention(meta, now)
    assert r == 1.0


def test_compute_retention_old_weak() -> None:
    now = datetime.now(UTC)
    meta = FieldMeta(last_accessed=now - timedelta(days=30), access_count=1, strength=1.0)
    r = compute_retention(meta, now)
    assert r < 0.001  # e^(-30/1) is essentially 0


def test_compute_retention_old_strong() -> None:
    now = datetime.now(UTC)
    meta = FieldMeta(last_accessed=now - timedelta(days=30), access_count=10, strength=10.0)
    r = compute_retention(meta, now)
    assert r > 0.01  # e^(-30/10) = e^(-3) ≈ 0.05


def test_apply_decay_prunes_weak() -> None:
    now = datetime.now(UTC)
    fields = [
        UserField(key="fresh", value="yes", meta=FieldMeta(last_accessed=now, strength=1.0)),
        UserField(key="old", value="no", meta=FieldMeta(last_accessed=now - timedelta(days=60), strength=1.0)),
    ]
    surviving = apply_decay(fields, now)
    assert len(surviving) == 1
    assert surviving[0].key == "fresh"


def test_apply_decay_keeps_strong() -> None:
    now = datetime.now(UTC)
    fields = [
        UserField(key="strong", value="yes", meta=FieldMeta(last_accessed=now - timedelta(days=10), strength=20.0)),
    ]
    surviving = apply_decay(fields, now)
    assert len(surviving) == 1


def test_reinforce_field() -> None:
    now = datetime.now(UTC)
    field = UserField(key="test", value="val", meta=FieldMeta(last_accessed=now - timedelta(days=1), access_count=3, strength=2.0))
    reinforced = reinforce_field(field, now)
    assert reinforced.meta.access_count == 4
    assert reinforced.meta.strength == 2.5
    assert reinforced.meta.last_accessed == now
    assert reinforced.key == "test"
    assert reinforced.value == "val"
