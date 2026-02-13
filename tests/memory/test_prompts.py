from __future__ import annotations

from vibe.core.memory.prompts import MemoryPrompt


def test_all_prompts_readable() -> None:
    for prompt in MemoryPrompt:
        content = prompt.read()
        assert len(content) > 0, f"Prompt {prompt.name} is empty"


def test_score_importance_prompt_has_placeholders() -> None:
    # score_importance.md is used as a system prompt directly (no format vars)
    content = MemoryPrompt.SCORE_IMPORTANCE.read()
    assert "importance" in content.lower() or "score" in content.lower()


def test_reflect_prompt_has_format_vars() -> None:
    content = MemoryPrompt.REFLECT.read()
    assert "{current_state}" in content
    assert "{observations}" in content


def test_consolidate_st_prompt_has_format_vars() -> None:
    content = MemoryPrompt.CONSOLIDATE_ST.read()
    assert "{observations}" in content
    assert "{context_key}" in content


def test_consolidate_lt_prompt_has_format_vars() -> None:
    content = MemoryPrompt.CONSOLIDATE_LT.read()
    assert "{current_long_term}" in content
    assert "{new_items}" in content
    assert "{context_key}" in content


def test_prompt_paths_exist() -> None:
    for prompt in MemoryPrompt:
        assert prompt.path.exists(), f"Prompt file missing: {prompt.path}"
