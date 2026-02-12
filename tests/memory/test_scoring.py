from __future__ import annotations

import pytest

from vibe.core.memory.scoring import ImportanceScorer, heuristic_score


@pytest.fixture
def make_scorer():
    def _make(response: str = "7", should_raise: bool = False):
        async def mock_llm(system: str, user: str) -> str:
            if should_raise:
                raise RuntimeError("LLM call failed")
            return response

        return ImportanceScorer(mock_llm)

    return _make


@pytest.mark.asyncio
async def test_score_returns_integer(make_scorer) -> None:
    scorer = make_scorer("7")
    assert await scorer.score("test message") == 7


@pytest.mark.asyncio
async def test_score_parses_from_text(make_scorer) -> None:
    scorer = make_scorer("I'd rate this a 6")
    assert await scorer.score("test message") == 6


@pytest.mark.asyncio
async def test_score_clamps_high(make_scorer) -> None:
    scorer = make_scorer("15")
    assert await scorer.score("test message") == 10


@pytest.mark.asyncio
async def test_score_clamps_low(make_scorer) -> None:
    scorer = make_scorer("0")
    # 0 is found by regex, clamped to max(1, min(10, 0)) = 1
    assert await scorer.score("test message") == 1


@pytest.mark.asyncio
async def test_score_defaults_on_garbage(make_scorer) -> None:
    scorer = make_scorer("not a number at all")
    assert await scorer.score("test message") == 3


@pytest.mark.asyncio
async def test_score_defaults_on_exception(make_scorer) -> None:
    scorer = make_scorer(should_raise=True)
    assert await scorer.score("test message") == 3


def test_parse_score_direct() -> None:
    assert ImportanceScorer._parse_score("5") == 5
    assert ImportanceScorer._parse_score("  8  ") == 8
    assert ImportanceScorer._parse_score("Rating: 3") == 3
    assert ImportanceScorer._parse_score("") == 3
    assert ImportanceScorer._parse_score("abc") == 3
    assert ImportanceScorer._parse_score("99") == 10
    assert ImportanceScorer._parse_score("-1") == 1


# -- Heuristic scorer tests --


def test_heuristic_empty_string() -> None:
    assert heuristic_score("") == 1


def test_heuristic_short_message() -> None:
    score = heuristic_score("fix the bug")
    assert 1 <= score <= 10
    assert score == 3  # short, no keywords, no declaration


def test_heuristic_long_message_bonus() -> None:
    short_score = heuristic_score("x" * 50)
    medium_score = heuristic_score("x" * 150)
    long_score = heuristic_score("x" * 350)
    assert medium_score > short_score
    assert long_score > medium_score


def test_heuristic_keyword_bonus() -> None:
    without_kw = heuristic_score("Please update the file")
    with_kw = heuristic_score("I think we should always use this architecture pattern")
    assert with_kw > without_kw


def test_heuristic_first_person_declaration() -> None:
    plain = heuristic_score("The team uses Python for backend work and data processing")
    first_person = heuristic_score("I prefer Python for backend work and data processing")
    assert first_person > plain


def test_heuristic_question_penalty() -> None:
    statement = heuristic_score("We use PostgreSQL for the database layer")
    question = heuristic_score("What database do you use for the backend layer?")
    assert statement >= question


def test_heuristic_clamp_range() -> None:
    # Even extreme inputs should be in [1, 10]
    assert 1 <= heuristic_score("?") <= 10
    assert 1 <= heuristic_score("I always prefer to never use any framework, the architecture must follow this important convention rule " * 10) <= 10
