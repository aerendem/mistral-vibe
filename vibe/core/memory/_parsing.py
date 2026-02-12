from __future__ import annotations


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM responses."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    lines = lines[1:]  # Strip opening fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]  # Strip closing fence
    return "\n".join(lines).strip()
