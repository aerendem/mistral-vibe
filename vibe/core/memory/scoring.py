from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable

from vibe.core.memory.prompts import MemoryPrompt

logger = logging.getLogger(__name__)

_HIGH_SIGNAL_KEYWORDS: frozenset[str] = frozenset({
    "prefer", "always", "never", "important", "goal", "must",
    "workflow", "convention", "architecture", "pattern", "principle",
    "requirement", "constraint", "decision", "standard", "rule",
    "team", "project", "stack", "framework", "database",
})

_FIRST_PERSON_RE = re.compile(
    r"^i\s+(prefer|always|never|use|like|work|need|want|believe|think)\b",
    re.IGNORECASE,
)


def heuristic_score(message: str) -> int:
    """Score message importance without an LLM call.

    Returns an integer 1-10.
    """
    text = message.strip()
    if not text:
        return 1

    score = 3

    # Length bonuses
    if len(text) > 300:
        score += 2
    elif len(text) > 100:
        score += 1

    # High-signal keyword detection (+2, once)
    lower = text.lower()
    if any(kw in lower for kw in _HIGH_SIGNAL_KEYWORDS):
        score += 2

    # First-person declaration pattern
    if _FIRST_PERSON_RE.search(text):
        score += 1

    # Question penalty
    if text.endswith("?") or lower.split()[0] in {
        "what", "where", "when", "how", "why", "who", "which", "is", "are",
        "do", "does", "can", "could", "would", "should",
    }:
        score -= 1

    return max(1, min(10, score))


class ImportanceScorer:
    """LLM-based importance scoring for observations (1-10 scale)."""

    def __init__(self, llm_caller: Callable[[str, str], Awaitable[str]]) -> None:
        self._llm_caller = llm_caller

    async def score(self, message: str, context_summary: str = "") -> int:
        system = MemoryPrompt.SCORE_IMPORTANCE.read()
        user_prompt = (
            f"Context: {context_summary}\n\nMessage: {message}"
            if context_summary
            else f"Message: {message}"
        )
        try:
            response = await self._llm_caller(system, user_prompt)
            return self._parse_score(response)
        except Exception:
            logger.warning("Importance scoring failed, defaulting to 3", exc_info=True)
            return 3

    @staticmethod
    def _parse_score(response: str) -> int:
        match = re.search(r"\d+", response.strip())
        if match:
            return max(1, min(10, int(match.group())))
        return 3
