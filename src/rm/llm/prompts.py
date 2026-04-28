"""Versioned prompt template loader.

Prompts live under ``src/rm/llm/prompts/<version>/<name>.txt``. We use
``str.format_map`` substitution (``{var}``) — keep prompts free of curly braces
that aren't placeholders, or escape with ``{{`` / ``}}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class PromptTemplate:
    name: str
    version: str
    body: str

    def format(self, /, **kwargs) -> str:
        try:
            return self.body.format_map(_SafeDict(kwargs))
        except KeyError as exc:
            raise KeyError(f"Missing prompt variable {exc} for {self.version}/{self.name}") from exc


class _SafeDict(dict):
    def __missing__(self, key):
        raise KeyError(key)


def load_prompt(name: str, version: str = "v1") -> PromptTemplate:
    path = PROMPTS_DIR / version / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return PromptTemplate(name=name, version=version, body=path.read_text(encoding="utf-8"))


def list_prompts(version: str = "v1") -> list[str]:
    base = PROMPTS_DIR / version
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.txt"))
