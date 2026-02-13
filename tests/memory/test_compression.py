"""Unit tests for vibe.core.memory.compression primitives."""

from __future__ import annotations

import json
import string

import pytest

from vibe.core.memory.compression import (
    ExactDedup,
    compact_dumps,
    compress_text,
    decompress_text,
    hamming_distance,
    is_low_entropy,
    is_near_duplicate,
    shannon_entropy,
    simhash,
)


# -- Compact JSON -----------------------------------------------------------


def test_compact_dumps_no_spaces() -> None:
    obj = {"key": "value", "list": [1, 2, 3]}
    result = compact_dumps(obj)
    assert " " not in result


def test_compact_dumps_round_trip() -> None:
    obj = {"nested": {"a": 1}, "arr": [True, None, "x"]}
    assert json.loads(compact_dumps(obj)) == obj


def test_compact_dumps_non_ascii() -> None:
    obj = {"name": "café", "emoji": "🐍"}
    result = compact_dumps(obj)
    assert "café" in result
    assert "🐍" in result
    assert "\\u" not in result  # not escaped


# -- Shannon Entropy ---------------------------------------------------------


def test_entropy_empty() -> None:
    assert shannon_entropy("") == 0.0


def test_entropy_single_char() -> None:
    assert shannon_entropy("aaaaaa") == 0.0


def test_entropy_uniform() -> None:
    text = string.ascii_lowercase  # 26 unique chars
    entropy = shannon_entropy(text)
    assert 4.5 < entropy < 5.0  # log2(26) ≈ 4.7


def test_entropy_english() -> None:
    text = "I prefer using Python for backend development with FastAPI"
    entropy = shannon_entropy(text)
    assert 3.5 < entropy < 4.8


# -- is_low_entropy ----------------------------------------------------------


def test_is_low_entropy_repetitive() -> None:
    assert is_low_entropy("aaa bbb aaa bbb aaa bbb aaa bbb") is True


def test_is_low_entropy_normal() -> None:
    text = "I always prefer pytest with strict type hints for testing"
    assert is_low_entropy(text) is False


def test_is_low_entropy_short() -> None:
    # Short texts (<=10 chars) should always return False
    assert is_low_entropy("aaa") is False
    assert is_low_entropy("aaaaaaaaaa") is False  # exactly 10


# -- zlib compression --------------------------------------------------------


def test_compress_round_trip() -> None:
    text = "x" * 300 + " some varied content to ensure compression works well " * 5
    compressed = compress_text(text)
    assert compressed != text  # should actually compress
    assert decompress_text(compressed) == text


def test_compress_short_passthrough() -> None:
    short = "This is a short string"
    assert compress_text(short) is short  # unchanged, not compressed


def test_decompress_raw_text() -> None:
    raw = "This is uncompressed legacy data"
    assert decompress_text(raw) == raw


def test_compress_prefix() -> None:
    text = "a" * 300
    compressed = compress_text(text)
    assert compressed.startswith("z:")


def test_compress_empty() -> None:
    assert compress_text("") == ""
    assert decompress_text("") == ""


def test_compress_skips_when_no_gain() -> None:
    # Random-looking text that doesn't compress well
    # base64 of zlib on incompressible data may be larger
    import os

    random_text = os.urandom(256).hex()  # 512 chars of hex — high entropy
    result = compress_text(random_text)
    # Either compressed (z: prefix) or passed through — both are valid
    assert decompress_text(result) == random_text


# -- ExactDedup --------------------------------------------------------------


def test_exact_dedup_seen() -> None:
    d = ExactDedup(max_size=10)
    assert d.seen("hello") is False  # first time
    assert d.seen("hello") is True  # second time


def test_exact_dedup_different() -> None:
    d = ExactDedup(max_size=10)
    assert d.seen("hello") is False
    assert d.seen("world") is False


def test_exact_dedup_lru_eviction() -> None:
    d = ExactDedup(max_size=3)
    d.seen("a")  # set: {a}
    d.seen("b")  # set: {a, b}
    d.seen("c")  # set: {a, b, c}  — full
    d.seen("d")  # evicts "a" → set: {b, c, d}
    # "a" was evicted — seen() returns False (note: also re-adds "a")
    assert d.seen("a") is False
    # "d" (most recent add) and "c" should still be present
    assert d.seen("d") is True
    assert d.seen("c") is True


def test_exact_dedup_lru_refresh() -> None:
    d = ExactDedup(max_size=3)
    d.seen("a")
    d.seen("b")
    d.seen("c")
    # Refresh "a" so it's no longer oldest
    d.seen("a")  # hit → moved to end
    d.seen("d")  # should evict "b" (now oldest)
    assert d.seen("a") is True  # was refreshed, still present
    assert d.seen("b") is False  # was evicted


# -- SimHash -----------------------------------------------------------------


def test_simhash_identical() -> None:
    text = "I prefer pytest for testing"
    assert simhash(text) == simhash(text)


def test_simhash_similar() -> None:
    a = "I prefer pytest for testing"
    b = "I always use pytest for testing"
    dist = hamming_distance(simhash(a), simhash(b))
    assert dist <= 15  # similar texts should have low distance


def test_simhash_different() -> None:
    a = "I prefer pytest for testing"
    b = "FastAPI async web framework with PostgreSQL database"
    dist = hamming_distance(simhash(a), simhash(b))
    assert dist >= 15  # different topics should have high distance


def test_simhash_empty() -> None:
    assert simhash("") == 0


def test_hamming_identical() -> None:
    assert hamming_distance(42, 42) == 0


def test_hamming_all_different() -> None:
    assert hamming_distance(0, (1 << 64) - 1) == 64


def test_is_near_duplicate_true() -> None:
    existing = [simhash("I prefer pytest for testing")]
    assert is_near_duplicate("I prefer pytest for testing", existing) is True


def test_is_near_duplicate_false() -> None:
    existing = [simhash("I prefer pytest for testing")]
    assert (
        is_near_duplicate(
            "FastAPI async web framework with PostgreSQL", existing, threshold=3
        )
        is False
    )
