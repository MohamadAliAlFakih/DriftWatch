"""Prompt loader. Prompts live as .md files; this loader reads + caches them."""

from functools import lru_cache
from pathlib import Path

# resolve prompts directory relative to this file (works inside Docker too)
_PROMPTS_DIR = Path(__file__).resolve().parent


# load a named prompt file, cached per process so we don't re-read on every node call
@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
