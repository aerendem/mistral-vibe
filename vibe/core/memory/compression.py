"""Training-free compression primitives for the memory system.

All stdlib. No external dependencies. No training required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from collections import OrderedDict
from collections.abc import Hashable
from math import log2
from typing import Any

# ---------------------------------------------------------------------------
# Compact JSON
# ---------------------------------------------------------------------------

def compact_dumps(obj: Any) -> str:
    """json.dumps with minimal whitespace and non-ASCII pass-through."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Shannon Entropy (pre-filter)
# ---------------------------------------------------------------------------

def shannon_entropy(text: str) -> float:
    """Character-level Shannon entropy in bits. English text ~ 4.0-4.5."""
    if not text:
        return 0.0
    from collections import Counter

    counts = Counter(text.lower())
    n = len(text)
    return -sum((c / n) * log2(c / n) for c in counts.values())


def is_low_entropy(text: str, threshold: float = 3.0) -> bool:
    """True if text has unusually low information content.

    Short texts (<=10 chars) always return False to avoid false positives.
    """
    return len(text) > 10 and shannon_entropy(text) < threshold


# ---------------------------------------------------------------------------
# zlib with explicit ``z:`` prefix (long_term storage)
# ---------------------------------------------------------------------------

_COMPRESS_PREFIX = "z:"
_MIN_COMPRESS_LEN = 256


def compress_text(text: str) -> str:
    """Conditionally zlib-compress text.

    Only compresses when byte length >= 256 AND the compressed form is
    actually smaller. Returns the original text unchanged otherwise.
    Compressed output is stored as ``z:<base64>``.
    """
    if not text:
        return text
    text_bytes = text.encode()
    if len(text_bytes) < _MIN_COMPRESS_LEN:
        return text
    compressed_b64 = base64.b64encode(zlib.compress(text_bytes, 6)).decode("ascii")
    # Compare byte lengths, not char lengths (matters for non-ASCII)
    if len(compressed_b64) + len(_COMPRESS_PREFIX) >= len(text_bytes):
        return text  # compression didn't help
    return _COMPRESS_PREFIX + compressed_b64


def decompress_text(data: str) -> str:
    """Decompress if ``z:`` prefix present. Raw text passes through."""
    if not data:
        return ""
    if data.startswith(_COMPRESS_PREFIX):
        try:
            return zlib.decompress(
                base64.b64decode(data[len(_COMPRESS_PREFIX) :])
            ).decode()
        except Exception:
            return data
    return data


# ---------------------------------------------------------------------------
# Exact Dedup (scoring dedup — replaces Bloom filter)
# ---------------------------------------------------------------------------


class ExactDedup:
    """Bounded LRU exact-match dedup using blake2b digests.

    Zero false positives.  Uses ``OrderedDict`` with ``move_to_end()`` on
    hits for true LRU eviction.
    """

    def __init__(self, max_size: int = 500) -> None:
        self._max_size = max_size
        self._seen: OrderedDict[str, None] = OrderedDict()

    def _digest(self, text: str) -> str:
        return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()

    def seen(self, text: str) -> bool:
        """Return True if *text* was already seen.  Adds it if not.

        On a hit the entry is refreshed (moved to end) so the eviction
        policy is true LRU.
        """
        d = self._digest(text)
        if d in self._seen:
            self._seen.move_to_end(d)
            return True
        if len(self._seen) >= self._max_size:
            self._seen.popitem(last=False)  # evict least-recently-used
        self._seen[d] = None
        return False


# ---------------------------------------------------------------------------
# SimHash (disabled by default — fuzzy sensory dedup)
# ---------------------------------------------------------------------------


def simhash(text: str, bits: int = 64) -> int:
    """SimHash fingerprint using word-level features + MD5 hashing."""
    tokens = text.lower().split()
    if not tokens:
        return 0
    v = [0] * bits
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)  # noqa: S324
        for i in range(bits):
            v[i] += 1 if h & (1 << i) else -1
    return sum(1 << i for i in range(bits) if v[i] > 0)


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two SimHash fingerprints."""
    return bin(a ^ b).count("1")


def is_near_duplicate(
    text: str, existing_hashes: list[int], threshold: int = 3
) -> bool:
    """Check if *text* is a near-duplicate of any existing hash."""
    h = simhash(text)
    return any(hamming_distance(h, eh) <= threshold for eh in existing_hashes)
