from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path

from vibe import VIBE_ROOT

_PROMPTS_DIR = VIBE_ROOT / "core" / "memory" / "prompts"


class MemoryPrompt(StrEnum):
    @property
    def path(self) -> Path:
        return (_PROMPTS_DIR / self.value).with_suffix(".md")

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8").strip()

    SCORE_IMPORTANCE = auto()
    REFLECT = auto()
    CONSOLIDATE_ST = auto()
    CONSOLIDATE_LT = auto()


__all__ = ["MemoryPrompt"]
