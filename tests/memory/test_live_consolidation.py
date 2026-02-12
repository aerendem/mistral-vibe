"""Live LLM consolidation benchmarks — real Mistral API, no mocks."""

from __future__ import annotations

import asyncio
import getpass
from pathlib import Path

import pytest

from tests.memory.conftest import (
    LIVE_LLM,
    SKIP_NO_KEY,
    judge_accuracy,
    make_live_config,
    soft_judge_assert,
)
from vibe.core.memory import MemoryManager
from vibe.core.memory.config import MemoryConfig
from vibe.core.memory.consolidation import ConsolidationEngine
from vibe.core.memory.storage import MemoryStore

pytestmark = [SKIP_NO_KEY, LIVE_LLM, pytest.mark.asyncio, pytest.mark.timeout(120)]

CTX_KEY = "project:/workspace/test-consolidation"

DECISION_MESSAGES = [
    "Chose PostgreSQL over MySQL for the new analytics service",
    "Using Docker Compose for local development across the team",
    "Switched from REST to GraphQL for the mobile API",
    "Adopted trunk-based development, deprecating gitflow",
    "Migrated CI from Jenkins to GitHub Actions last sprint",
    "Standardized on pytest for all new test suites",
    "Using structlog for structured JSON logging in production",
    "Deployed Redis for session caching instead of Memcached",
    "Adopted pre-commit hooks for linting and formatting",
    "Using Terraform for infrastructure as code",
    "Switched to FastAPI from Flask for new microservices",
    "Implemented feature flags with LaunchDarkly",
    "Using Sentry for error tracking across all services",
    "Adopted Conventional Commits for all repositories",
    "Using Black and isort for automatic code formatting",
]


async def _observe_with_delay(
    mgr: MemoryManager, messages: list[str], delay: float = 0.5
) -> None:
    for msg in messages:
        await mgr.observe(msg, "user")
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Test 1: Sensory -> Short-term produces valid points
# ---------------------------------------------------------------------------


async def test_live_sensory_to_st_produces_valid_points(
    tmp_path: Path, real_llm_caller
) -> None:
    db_path = tmp_path / "consol_st.db"
    store = MemoryStore(db_path)
    user_id = getpass.getuser()
    store.get_or_create_user_state(user_id)
    engine = ConsolidationEngine(real_llm_caller, store)
    try:
        for msg in DECISION_MESSAGES[:10]:
            store.add_sensory(CTX_KEY, user_id, msg, cap=10)

        ctx = store.get_or_create_context_memory(CTX_KEY, user_id)
        points = await engine._consolidate_sensory(ctx.sensory, CTX_KEY)

        assert len(points) >= 1
        assert all(isinstance(p, str) and p.strip() for p in points)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 2: Sensory -> Short-term captures key info (LLM-judged)
# ---------------------------------------------------------------------------


async def test_live_sensory_to_st_captures_key_info(
    tmp_path: Path, real_llm_caller
) -> None:
    db_path = tmp_path / "consol_st_judge.db"
    store = MemoryStore(db_path)
    user_id = getpass.getuser()
    store.get_or_create_user_state(user_id)
    engine = ConsolidationEngine(real_llm_caller, store)
    try:
        for msg in DECISION_MESSAGES:
            store.add_sensory(CTX_KEY, user_id, msg, cap=15)

        ctx = store.get_or_create_context_memory(CTX_KEY, user_id)
        points = await engine._consolidate_sensory(ctx.sensory, CTX_KEY)

        assert len(points) >= 1

        combined = "\n".join(points)
        score = await judge_accuracy(
            real_llm_caller,
            observations=DECISION_MESSAGES,
            output=combined,
            criterion="key decision capture",
        )
        soft_judge_assert(score, 6, "sensory_to_st_key_info")
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 3: Short-term -> Long-term produces prose
# ---------------------------------------------------------------------------


async def test_live_st_to_lt_produces_prose(
    tmp_path: Path, real_llm_caller
) -> None:
    db_path = tmp_path / "consol_lt.db"
    store = MemoryStore(db_path)
    user_id = getpass.getuser()
    store.get_or_create_user_state(user_id)
    engine = ConsolidationEngine(real_llm_caller, store)
    try:
        ctx = store.get_or_create_context_memory(CTX_KEY, user_id)
        ctx.short_term = [
            "Team uses PostgreSQL and Redis for data storage",
            "CI/CD runs on GitHub Actions with trunk-based development",
            "Python backend stack: FastAPI, pytest, structlog",
            "Docker Compose for local dev, Terraform for infrastructure",
            "Code quality enforced via pre-commit, Black, and isort",
        ]
        store.update_context_memory(ctx)

        result = await engine._consolidate_long_term(
            ctx.long_term, ctx.short_term, CTX_KEY
        )

        assert result.strip() != ""
        assert not result.strip().startswith("[")
        assert not result.strip().startswith("{")
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 4: Short-term -> Long-term incorporates old and new
# ---------------------------------------------------------------------------


async def test_live_st_to_lt_incorporates_old_and_new(
    tmp_path: Path, real_llm_caller
) -> None:
    db_path = tmp_path / "consol_lt_merge.db"
    store = MemoryStore(db_path)
    user_id = getpass.getuser()
    store.get_or_create_user_state(user_id)
    engine = ConsolidationEngine(real_llm_caller, store)
    try:
        ctx = store.get_or_create_context_memory(CTX_KEY, user_id)
        ctx.long_term = (
            "This is a Flask-based project using MySQL for data storage "
            "and Jenkins for CI."
        )
        ctx.short_term = [
            "Migrated from Flask to FastAPI for async support",
            "Switched database from MySQL to PostgreSQL for better JSON support",
            "Moved CI pipeline from Jenkins to GitHub Actions",
        ]
        store.update_context_memory(ctx)

        result = await engine._consolidate_long_term(
            ctx.long_term, ctx.short_term, CTX_KEY
        )

        result_lower = result.lower()
        assert "fastapi" in result_lower
        assert "postgresql" in result_lower

        score = await judge_accuracy(
            real_llm_caller,
            observations=[
                ctx.long_term,
                *ctx.short_term,
            ],
            output=result,
            criterion="migration incorporation",
        )
        soft_judge_assert(score, 6, "st_to_lt_migration_incorporation")
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 5: Full pipeline consolidation via MemoryManager
# ---------------------------------------------------------------------------


async def test_live_full_pipeline_consolidation(
    tmp_path: Path, real_llm_caller
) -> None:
    config = make_live_config(
        tmp_path, sensory_cap=10, short_term_cap=5, auto_consolidate=True
    )
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, DECISION_MESSAGES[:12])
        await mgr.on_session_end()

        ctx = mgr._store.get_or_create_context_memory(
            mgr.get_context_key(), mgr._user_id
        )

        assert len(ctx.short_term) >= 1 or ctx.long_term != ""
    finally:
        mgr.close()


# ---------------------------------------------------------------------------
# Test 6: Consolidation quality judged by LLM
# ---------------------------------------------------------------------------


async def test_live_consolidation_judge_quality(
    tmp_path: Path, real_llm_caller
) -> None:
    config = make_live_config(
        tmp_path, sensory_cap=10, short_term_cap=5, auto_consolidate=True
    )
    mgr = MemoryManager(config, real_llm_caller)
    try:
        await _observe_with_delay(mgr, DECISION_MESSAGES[:12])
        await mgr.on_session_end()

        block = mgr.get_memory_block()
        assert block != ""

        score = await judge_accuracy(
            real_llm_caller,
            observations=DECISION_MESSAGES[:12],
            output=block,
            criterion="consolidation quality",
        )
        soft_judge_assert(score, 6, "full_pipeline_consolidation_quality")
    finally:
        mgr.close()
