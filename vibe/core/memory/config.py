from __future__ import annotations

from pydantic import BaseModel


class MemoryConfig(BaseModel):
    """Configuration for the LightMem memory system."""

    enabled: bool = False
    db_path: str = ""
    importance_threshold: int = 3
    reflection_trigger: int = 50
    scoring_mode: str = "llm"
    observe_assistant: bool = False
    sensory_cap: int = 50
    short_term_cap: int = 15
    max_user_fields: int = 20
    injection_budget_tokens: int = 500
    decay_enabled: bool = True
    decay_prune_threshold: float = 0.1
    auto_consolidate: bool = True
    compress_storage: bool = True
    dedup_sensory: bool = False
    audit_retention_days: int = 90
    stale_context_days: int = 60
