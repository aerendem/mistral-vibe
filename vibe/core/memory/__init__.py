from __future__ import annotations

import asyncio
import getpass
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from vibe.core.memory.compression import ExactDedup, is_low_entropy
from vibe.core.memory.config import MemoryConfig
from vibe.core.memory.consolidation import ConsolidationEngine
from vibe.core.memory.injection import MemoryInjector
from vibe.core.memory.models import Observation
from vibe.core.memory.reflection import ReflectionEngine
from vibe.core.memory.scoring import ImportanceScorer, heuristic_score
from vibe.core.memory.storage import MemoryStore

logger = logging.getLogger(__name__)

__all__ = ["MemoryConfig", "MemoryManager"]

_TRIVIAL_PATTERNS: frozenset[str] = frozenset({
    "yes", "no", "ok", "okay", "sure", "thanks", "thank you", "got it",
    "continue", "go ahead", "yep", "nope", "right", "correct", "agreed",
    "sounds good", "looks good", "lgtm", "ack", "k", "y", "n",
    "please", "go on", "next", "done", "great", "nice", "cool",
    "perfect", "fine", "alright",
})

_TRIVIAL_MAX_LENGTH = 15


class _MemoryWriteQueue:
    """Single-writer async queue for serializing memory writes.

    Queues (message, role) data tuples — NOT coroutine objects — to avoid
    'coroutine was never awaited' leaks if close() happens before drain.
    """

    def __init__(self, handler: Callable[[str, str], Awaitable[None]]) -> None:
        self._queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        self._handler = handler
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the worker task. Idempotent — no-op if already running."""
        if self._worker is not None:
            return
        self._worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                message, role = item
                await self._handler(message, role)
            except Exception:
                logger.warning("Memory write failed", exc_info=True)
            self._queue.task_done()

    def enqueue(self, message: str, role: str) -> None:
        """Enqueue a payload for later processing."""
        self._queue.put_nowait((message, role))

    def has_pending(self) -> bool:
        """Return True when unprocessed write items are queued."""
        return self._queue.qsize() > 0

    async def drain(self) -> None:
        """Block until all queued writes complete."""
        await self._queue.join()

    async def shutdown(self) -> None:
        """Drain then stop the worker."""
        if self._worker is None:
            return
        await self.drain()
        self._queue.put_nowait(None)
        await self._worker
        self._worker = None


@dataclass
class MemoryMetrics:
    """Per-session counters for memory operations."""

    observations_stored: int = 0
    observations_skipped_trivial: int = 0
    observations_skipped_low_entropy: int = 0
    observations_skipped_dedup: int = 0
    observations_skipped_low_score: int = 0
    sensory_dedup_skipped: int = 0
    reflections_triggered: int = 0
    injections_served: int = 0
    injections_empty: int = 0
    audit_logs_pruned: int = 0
    stale_contexts_pruned: int = 0


class MemoryManager:
    """Facade for all memory operations. Created once per AgentLoop."""

    def __init__(
        self,
        config: MemoryConfig,
        llm_caller: Callable[[str, str], Awaitable[str]],
    ) -> None:
        self.config = config
        self.enabled = config.enabled
        self.metrics = MemoryMetrics()
        self._last_maintenance: datetime | None = None

        if not self.enabled:
            self._store: MemoryStore | None = None
            self._write_queue = _MemoryWriteQueue(self._observe_impl)
            return

        db_path = Path(config.db_path) if config.db_path else self._default_db_path()
        self._store = MemoryStore(
            db_path,
            compress=config.compress_storage,
            dedup_sensory=config.dedup_sensory,
        )
        self._seen_dedup = ExactDedup(max_size=500)
        self._scorer = ImportanceScorer(llm_caller)
        self._injector = MemoryInjector(self._store)
        self._reflector = ReflectionEngine(llm_caller, self._store)
        self._consolidator = ConsolidationEngine(llm_caller, self._store)
        self._user_id = getpass.getuser()
        self._store.get_or_create_user_state(self._user_id)
        self._write_queue = _MemoryWriteQueue(self._observe_impl)

    def start(self) -> None:
        """Start the write queue worker. Idempotent."""
        self._write_queue.start()

    @staticmethod
    def _default_db_path() -> Path:
        from vibe.core.paths.global_paths import VIBE_HOME

        return VIBE_HOME.path / "memory" / "memory.db"

    @staticmethod
    def _is_trivial(message: str) -> bool:
        """Check if a message is too trivial to score (saves an LLM call)."""
        text = message.strip().lower()
        if not text:
            return True
        if len(text) >= _TRIVIAL_MAX_LENGTH:
            return False
        cleaned = text.rstrip(".,!?;:")
        return cleaned in _TRIVIAL_PATTERNS

    def get_context_key(self) -> str:
        """Derive context_key from the current working directory."""
        return f"project:{Path.cwd().resolve()}"

    async def observe(
        self, message: str, role: str = "user", session_id: str = ""
    ) -> None:
        """Async observe — kept for API compat and direct use in tests."""
        await self._observe_impl(message, role)

    def enqueue_observe(self, message: str, role: str = "user") -> None:
        """Non-blocking enqueue for middleware. No-op if not started."""
        if not self.enabled:
            return
        self._write_queue.enqueue(message, role)

    async def _observe_impl(self, message: str, role: str) -> None:
        """Score and store an observation."""
        if not self.enabled or not self._store:
            return
        try:
            if self._is_trivial(message):
                self.metrics.observations_skipped_trivial += 1
                return

            if is_low_entropy(message):
                self.metrics.observations_skipped_low_entropy += 1
                return

            dedup_key = f"{role}:{message}"
            if self._seen_dedup.seen(dedup_key):
                self.metrics.observations_skipped_dedup += 1
                return

            if self.config.scoring_mode == "heuristic":
                score = heuristic_score(message)
            else:
                score = await self._scorer.score(message)

            if score < self.config.importance_threshold:
                self.metrics.observations_skipped_low_score += 1
                return

            self.metrics.observations_stored += 1

            ctx_key = self.get_context_key()
            self._store.add_observation(
                Observation(
                    user_id=self._user_id,
                    context_key=ctx_key,
                    content=message,
                    importance=score,
                    source_role=role,
                )
            )
            stored = self._store.add_sensory(
                ctx_key, self._user_id, message, cap=self.config.sensory_cap
            )
            if not stored:
                self.metrics.sensory_dedup_skipped += 1

            state = self._store.get_or_create_user_state(self._user_id)
            state.accumulated_importance += score
            self._store.update_user_state(state)

            reflected = await self._reflector.maybe_reflect(
                self._user_id, self.config
            )
            if reflected:
                self.metrics.reflections_triggered += 1

        except Exception:
            logger.warning("Memory observation failed", exc_info=True)

    def get_memory_block(self) -> str:
        """Sync. Returns <memory> XML block or empty string for ephemeral injection."""
        if not self.enabled or not self._store:
            return ""
        try:
            block = self._injector.build_memory_block(
                self._user_id,
                self.get_context_key(),
                self.config.injection_budget_tokens,
            )
            if block:
                self.metrics.injections_served += 1
            else:
                self.metrics.injections_empty += 1
            return block
        except Exception:
            logger.warning("Memory injection failed", exc_info=True)
            return ""

    async def on_session_end(self) -> None:
        """Drain pending writes, run consolidation, then forgetting."""
        if not self.enabled or not self._store:
            return

        # Ensure queued pre-start observations can be consumed before draining.
        self.start()

        # Barrier: all pending observe writes must complete first
        await self._write_queue.drain()

        m = self.metrics
        logger.info(
            "Memory session: stored=%d skipped_trivial=%d skipped_entropy=%d "
            "skipped_dedup=%d skipped_low=%d sensory_dedup=%d "
            "reflections=%d injections=%d (empty=%d)",
            m.observations_stored,
            m.observations_skipped_trivial,
            m.observations_skipped_low_entropy,
            m.observations_skipped_dedup,
            m.observations_skipped_low_score,
            m.sensory_dedup_skipped,
            m.reflections_triggered,
            m.injections_served,
            m.injections_empty,
        )

        if self.config.auto_consolidate:
            try:
                ctx_key = self.get_context_key()
                await self._consolidator.run_full_consolidation(
                    ctx_key, self._user_id, self.config
                )
            except Exception:
                logger.warning("Memory consolidation failed", exc_info=True)

        await self.run_forgetting()

    async def run_forgetting(self) -> None:
        """Prune stale data: audit logs, stale contexts, DB maintenance."""
        if not self.enabled or not self._store:
            return
        try:
            config = self.config

            r_del, c_del = self._store.prune_audit_logs(config.audit_retention_days)
            if r_del or c_del:
                self.metrics.audit_logs_pruned += r_del + c_del
                logger.info("Pruned %d reflections, %d consolidations", r_del, c_del)

            stale = self._store.get_stale_contexts(config.stale_context_days)
            stale_pruned = 0
            for ctx_key, uid in stale:
                if self._store.prune_context_volatile(ctx_key, uid) > 0:
                    stale_pruned += 1
            if stale_pruned:
                self.metrics.stale_contexts_pruned += stale_pruned
                logger.info(
                    "Pruned volatile data from %d stale contexts", stale_pruned
                )

            now = datetime.now(UTC)
            should_maintain = (
                self._last_maintenance is None
                or (now - self._last_maintenance).total_seconds() > 86400
            )
            if should_maintain:
                self._store.run_maintenance()
                self._last_maintenance = now

        except Exception:
            logger.warning("Memory forgetting failed", exc_info=True)

    async def aclose(self) -> None:
        """Async close: drain queue then close DB."""
        if self._write_queue.has_pending() and self._write_queue._worker is None:
            self.start()
        await self._write_queue.shutdown()
        if self._store:
            self._store.close()

    def close(self) -> None:
        """Sync close (best-effort, for non-async contexts)."""
        if not self._store:
            return

        needs_async_close = (
            self._write_queue._worker is not None
            or self._write_queue.has_pending()
        )
        if not needs_async_close:
            self._store.close()
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return
        loop.create_task(self.aclose())
