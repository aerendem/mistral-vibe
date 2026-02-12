"""Tests for memory system integration with AgentLoop.

Tests _get_messages_for_backend, _MemoryObserverMiddleware, and the
ephemeral injection guarantee (self.messages never contains <memory>).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agent_loop import AgentLoop, _MemoryObserverMiddleware
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.memory.config import MemoryConfig
from vibe.core.middleware import ConversationContext, MiddlewareAction
from vibe.core.types import AgentStats, LLMMessage, Role


def _make_config(
    memory_enabled: bool = False,
    memory_db_path: str = "",
) -> VibeConfig:
    return VibeConfig(
        session_logging=SessionLoggingConfig(enabled=False),
        auto_compact_threshold=0,
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
        include_model_info=False,
        include_commit_signature=False,
        enabled_tools=[],
        tools={},
        memory=MemoryConfig(enabled=memory_enabled, db_path=memory_db_path),
    )


# -- _get_messages_for_backend tests --


def test_get_messages_returns_self_messages_when_disabled() -> None:
    backend = FakeBackend([mock_llm_chunk(content="ok")])
    agent = AgentLoop(_make_config(memory_enabled=False), backend=backend)

    result = agent._get_messages_for_backend()
    assert result is agent.messages  # same object, no copy


def test_get_messages_returns_self_messages_when_no_memory_data(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_memory.db")
    backend = FakeBackend([mock_llm_chunk(content="ok")])
    agent = AgentLoop(_make_config(memory_enabled=True, memory_db_path=db_path), backend=backend)

    # Memory is enabled but no data has been stored, so get_memory_block returns ""
    result = agent._get_messages_for_backend()
    assert result is agent.messages  # same object when no memory block


def test_get_messages_inserts_memory_at_index_1(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_memory.db")
    backend = FakeBackend([mock_llm_chunk(content="ok")])
    agent = AgentLoop(_make_config(memory_enabled=True, memory_db_path=db_path), backend=backend)

    # Manually populate memory so get_memory_block returns something
    from vibe.core.memory.models import Seed

    state = agent.memory_manager._store.get_or_create_user_state(agent.memory_manager._user_id)
    state.seed = Seed(user_model="test developer")
    agent.memory_manager._store.update_user_state(state)
    agent.memory_manager._store.get_or_create_context_memory(
        agent.memory_manager.get_context_key(), agent.memory_manager._user_id
    )

    result = agent._get_messages_for_backend()

    # Result should be a new list, NOT the same object
    assert result is not agent.messages
    assert len(result) == len(agent.messages) + 1
    # Memory message at index 1
    assert result[1].role == Role.system
    assert "<memory>" in result[1].content
    assert "test developer" in result[1].content
    # Original messages list is unmodified
    assert all("<memory>" not in m.content for m in agent.messages)

    agent.memory_manager.close()


def test_get_messages_preserves_original_messages(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_memory.db")
    backend = FakeBackend([mock_llm_chunk(content="ok")])
    agent = AgentLoop(_make_config(memory_enabled=True, memory_db_path=db_path), backend=backend)

    from vibe.core.memory.models import Seed

    state = agent.memory_manager._store.get_or_create_user_state(agent.memory_manager._user_id)
    state.seed = Seed(affect="focused")
    agent.memory_manager._store.update_user_state(state)
    agent.memory_manager._store.get_or_create_context_memory(
        agent.memory_manager.get_context_key(), agent.memory_manager._user_id
    )

    original_len = len(agent.messages)
    original_contents = [m.content for m in agent.messages]

    _ = agent._get_messages_for_backend()

    # Original list unchanged
    assert len(agent.messages) == original_len
    assert [m.content for m in agent.messages] == original_contents

    agent.memory_manager.close()


# -- _MemoryObserverMiddleware tests --


def _make_context(messages: list[LLMMessage]) -> ConversationContext:
    return ConversationContext(
        messages=messages,
        stats=AgentStats(),
        config=_make_config(),
    )


def _mock_mm(observe_assistant: bool = False) -> MagicMock:
    mm = MagicMock()
    mm.config = MagicMock()
    mm.config.observe_assistant = observe_assistant
    return mm


@pytest.mark.asyncio
async def test_middleware_before_turn_is_noop() -> None:
    mm = _mock_mm()
    middleware = _MemoryObserverMiddleware(mm)
    ctx = _make_context([LLMMessage(role=Role.system, content="sys")])

    result = await middleware.before_turn(ctx)
    assert result.action == MiddlewareAction.CONTINUE


@pytest.mark.asyncio
async def test_middleware_observes_user_messages() -> None:
    mm = _mock_mm()
    middleware = _MemoryObserverMiddleware(mm)

    ctx = _make_context([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="hello world"),
        LLMMessage(role=Role.assistant, content="hi"),
    ])

    await middleware.after_turn(ctx)

    mm.enqueue_observe.assert_called_once_with("hello world", "user")


@pytest.mark.asyncio
async def test_middleware_skips_non_user_messages() -> None:
    mm = _mock_mm()
    middleware = _MemoryObserverMiddleware(mm)

    ctx = _make_context([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.assistant, content="response"),
    ])

    await middleware.after_turn(ctx)

    mm.enqueue_observe.assert_not_called()


@pytest.mark.asyncio
async def test_middleware_tracks_last_observed_idx() -> None:
    mm = _mock_mm()
    middleware = _MemoryObserverMiddleware(mm)

    ctx1 = _make_context([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="msg1"),
    ])

    await middleware.after_turn(ctx1)

    assert middleware._last_observed_idx == 2
    mm.enqueue_observe.assert_called_once_with("msg1", "user")

    mm.enqueue_observe.reset_mock()

    # Second call with more messages — should only process new ones
    ctx2 = _make_context([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="msg1"),
        LLMMessage(role=Role.assistant, content="resp1"),
        LLMMessage(role=Role.user, content="msg2"),
    ])

    await middleware.after_turn(ctx2)

    # Only one enqueue_observe call for "msg2" (not "msg1" again)
    mm.enqueue_observe.assert_called_once_with("msg2", "user")
    assert middleware._last_observed_idx == 4


def test_middleware_reset_clears_index() -> None:
    mm = _mock_mm()
    middleware = _MemoryObserverMiddleware(mm)
    middleware._last_observed_idx = 10

    middleware.reset()

    assert middleware._last_observed_idx == 0


@pytest.mark.asyncio
async def test_middleware_skips_empty_content() -> None:
    mm = _mock_mm()
    middleware = _MemoryObserverMiddleware(mm)

    ctx = _make_context([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content=""),
    ])

    await middleware.after_turn(ctx)

    mm.enqueue_observe.assert_not_called()


# -- Middleware registration tests --


def test_middleware_not_registered_when_disabled() -> None:
    backend = FakeBackend([mock_llm_chunk(content="ok")])
    agent = AgentLoop(_make_config(memory_enabled=False), backend=backend)

    middlewares = agent.middleware_pipeline.middlewares
    assert not any(isinstance(m, _MemoryObserverMiddleware) for m in middlewares)


def test_middleware_registered_when_enabled(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_memory.db")
    backend = FakeBackend([mock_llm_chunk(content="ok")])
    agent = AgentLoop(
        _make_config(memory_enabled=True, memory_db_path=db_path), backend=backend
    )

    middlewares = agent.middleware_pipeline.middlewares
    assert any(isinstance(m, _MemoryObserverMiddleware) for m in middlewares)

    agent.memory_manager.close()


# -- Assistant observation tests --


@pytest.mark.asyncio
async def test_middleware_observes_assistant_when_enabled() -> None:
    mm = MagicMock()
    mm.config = MagicMock()
    mm.config.observe_assistant = True
    middleware = _MemoryObserverMiddleware(mm)

    long_content = "x" * 150
    ctx = _make_context([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="hello world"),
        LLMMessage(role=Role.assistant, content=long_content),
    ])

    await middleware.after_turn(ctx)

    # Both user and assistant messages should be observed
    assert mm.enqueue_observe.call_count == 2
    mm.enqueue_observe.assert_any_call("hello world", "user")
    mm.enqueue_observe.assert_any_call(long_content, "assistant")


@pytest.mark.asyncio
async def test_middleware_skips_short_assistant_messages() -> None:
    mm = MagicMock()
    mm.config = MagicMock()
    mm.config.observe_assistant = True
    middleware = _MemoryObserverMiddleware(mm)

    ctx = _make_context([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.assistant, content="ok"),  # too short (<=100)
    ])

    await middleware.after_turn(ctx)

    mm.enqueue_observe.assert_not_called()


@pytest.mark.asyncio
async def test_middleware_does_not_observe_assistant_when_disabled() -> None:
    mm = MagicMock()
    mm.config = MagicMock()
    mm.config.observe_assistant = False
    middleware = _MemoryObserverMiddleware(mm)

    long_content = "x" * 150
    ctx = _make_context([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.assistant, content=long_content),
    ])

    await middleware.after_turn(ctx)

    mm.enqueue_observe.assert_not_called()
