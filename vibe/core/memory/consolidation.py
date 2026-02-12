from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable

from vibe.core.memory._parsing import strip_markdown_fences
from vibe.core.memory.config import MemoryConfig
from vibe.core.memory.prompts import MemoryPrompt
from vibe.core.memory.storage import MemoryStore

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Hierarchical memory consolidation: sensory -> short-term -> long-term.

    Pass 1: Compress sensory observations into 2-3 short-term key points via LLM.
    Pass 2: Rewrite long-term knowledge blob incorporating absorbed short-term items.
    """

    def __init__(
        self,
        llm_caller: Callable[[str, str], Awaitable[str]],
        store: MemoryStore,
    ) -> None:
        self._llm_caller = llm_caller
        self._store = store

    async def run_full_consolidation(
        self, context_key: str, user_id: str, config: MemoryConfig
    ) -> None:
        """Run both consolidation passes."""
        ctx = self._store.get_or_create_context_memory(context_key, user_id)

        # Pass 1: Sensory -> Short-term
        if len(ctx.sensory) >= config.sensory_cap:
            try:
                new_points = await self._consolidate_sensory(ctx.sensory, context_key)
                if new_points:
                    ctx.short_term.extend(new_points)
                    ctx.sensory = []
                    self._store.update_context_memory(ctx)
                    # Re-fetch after version bump
                    ctx = self._store.get_or_create_context_memory(context_key, user_id)
            except Exception:
                logger.warning("Sensory consolidation failed", exc_info=True)

        # Pass 2: Short-term -> Long-term
        if len(ctx.short_term) >= config.short_term_cap:
            try:
                old_lt = ctx.long_term
                budget_chars = config.injection_budget_tokens * 4
                new_lt = await self._consolidate_long_term(
                    ctx.long_term, ctx.short_term, context_key,
                    budget_chars=budget_chars,
                )
                if new_lt:
                    ctx.long_term = new_lt
                    ctx.short_term = []
                    ctx.consolidation_count += 1
                    self._store.update_context_memory(ctx)
                    self._store.log_consolidation(
                        context_key, user_id, len(ctx.sensory), old_lt, new_lt
                    )
            except Exception:
                logger.warning("Long-term consolidation failed", exc_info=True)

    async def _consolidate_sensory(
        self, sensory: list[str], context_key: str
    ) -> list[str]:
        """Compress raw sensory observations into key points."""
        prompt_template = MemoryPrompt.CONSOLIDATE_ST.read()
        obs_str = "\n".join(f"- {s}" for s in sensory)
        system_prompt = prompt_template.format(
            observations=obs_str, context_key=context_key
        )

        raw = await self._llm_caller(
            system_prompt, "Compress the observations into key points."
        )
        return self._parse_points(raw)

    async def _consolidate_long_term(
        self, current_lt: str, new_items: list[str], context_key: str,
        budget_chars: int = 0,
    ) -> str:
        """Rewrite long-term knowledge incorporating new short-term items."""
        prompt_template = MemoryPrompt.CONSOLIDATE_LT.read()
        items_str = "\n".join(f"- {item}" for item in new_items)
        system_prompt = prompt_template.format(
            current_long_term=current_lt or "(empty)",
            new_items=items_str,
            context_key=context_key,
        )

        if budget_chars > 0:
            total_input = len(current_lt) + sum(len(i) for i in new_items)
            if total_input > budget_chars:
                system_prompt += (
                    f"\n\nIMPORTANT: The result MUST be under {budget_chars} characters. "
                    "Prioritize recent and frequently referenced information. "
                    "Drop outdated or low-signal content to stay within budget."
                )

        raw = await self._llm_caller(
            system_prompt, "Rewrite the long-term knowledge."
        )
        result = raw.strip()

        if budget_chars > 0 and len(result) > budget_chars:
            truncated = result[:budget_chars]
            idx = truncated.rfind(". ")
            if idx > 0:
                result = truncated[: idx + 1]
            else:
                result = truncated

        return result

    @staticmethod
    def _parse_points(raw: str) -> list[str]:
        """Extract a JSON array of strings from LLM response."""
        text = strip_markdown_fences(raw)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
            return []
        except json.JSONDecodeError:
            logger.warning("Failed to parse consolidation points: %s", text[:200])
            return []
