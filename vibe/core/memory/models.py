from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class Seed(BaseModel):
    """Immutable user state seed — fixed shape, never decays."""

    model_config = ConfigDict(frozen=True)

    affect: str = ""
    trust: str = ""
    intention: str = ""
    salient: str = ""
    user_model: str = ""

    def format_summary(self) -> str:
        """Format non-empty seed fields as 'field: value | field: value'."""
        parts = []
        for field_name in ("affect", "trust", "intention", "salient", "user_model"):
            val = getattr(self, field_name, "")
            if val:
                parts.append(f"{field_name}: {val}")
        return " | ".join(parts)


class FieldMeta(BaseModel):
    """Decay tracking metadata for a dynamic user field."""

    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    access_count: int = 0
    strength: float = 1.0


class UserField(BaseModel):
    """A dynamic, emergent user state field."""

    key: str
    value: str
    meta: FieldMeta = Field(default_factory=FieldMeta)


class UserState(BaseModel):
    """Global user state — one per user."""

    user_id: str
    seed: Seed = Field(default_factory=Seed)
    fields: list[UserField] = Field(default_factory=list)
    accumulated_importance: float = 0.0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Observation(BaseModel):
    """A scored observation from a user or assistant message."""

    id: int | None = None
    user_id: str = ""
    context_key: str = ""
    content: str = ""
    importance: int = 0
    source_role: str = ""
    created_at: datetime | None = None


class ContextMemory(BaseModel):
    """Per-context memory with 3-tier LightMem hierarchy."""

    context_key: str
    user_id: str
    sensory: list[str] = Field(default_factory=list)
    short_term: list[str] = Field(default_factory=list)
    long_term: str = ""
    version: int = 0
    updated_at: datetime | None = None
    consolidation_count: int = 0


