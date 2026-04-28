"""Unified LLM client.

Design goals
------------
* One ``LLMClient`` class — *no* framework abstraction (LangChain etc.).
* Routes to ``openai``-compat endpoints (vLLM / OpenAI / DeepSeek / Together /
  ...) **or** native Anthropic via their SDK.
* Retries with exponential backoff (``tenacity``).
* Tracks token usage so the eval runner can report cost.
* JSON-mode helper that parses & repairs LLM JSON output.
* ``MockLLMClient`` for tests — deterministic, no network.

The interface is intentionally tiny (``chat``, ``chat_json``, ``count_tokens``).
Everything in the rest of the codebase consumes this; never call openai/
anthropic directly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rm.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Lightweight types                                                           #
# --------------------------------------------------------------------------- #

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: Any = None
    finish_reason: str | None = None

    @property
    def tokens(self) -> int:
        return self.total_tokens or (self.prompt_tokens + self.completion_tokens)


@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0
    total: int = 0
    calls: int = 0

    def add(self, r: ChatResult) -> None:
        self.prompt += r.prompt_tokens
        self.completion += r.completion_tokens
        self.total += r.total_tokens or (r.prompt_tokens + r.completion_tokens)
        self.calls += 1


# --------------------------------------------------------------------------- #
# Real client                                                                 #
# --------------------------------------------------------------------------- #

class LLMClient:
    """Thin wrapper. ``provider`` selects the routing path.

    Supported providers
    -------------------
    * ``openai_compat``: any server speaking the OpenAI Chat Completions API
      (vLLM, DeepSeek, Together, OpenRouter, ...).
    * ``openai``: alias for ``openai_compat`` with the OpenAI default base URL.
    * ``anthropic``: native Anthropic ``messages`` endpoint.
    """

    def __init__(
        self,
        provider: str = "openai_compat",
        base_url: str | None = "http://localhost:8000/v1",
        api_key: str | None = "EMPTY",
        model: str = "qwen7b",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 120.0,
        retry_max_attempts: int = 4,
        retry_initial_wait: float = 1.0,
        retry_max_wait: float = 30.0,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.retry_max_attempts = retry_max_attempts
        self.retry_initial_wait = retry_initial_wait
        self.retry_max_wait = retry_max_wait

        self.usage = TokenUsage()
        self._client = self._build_client()

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #

    def _build_client(self):
        if self.provider in {"openai", "openai_compat"}:
            from openai import OpenAI

            return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        if self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
        raise ValueError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: Sequence[ChatMessage] | Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        json_mode: bool = False,
        stop: list[str] | None = None,
    ) -> ChatResult:
        msgs = _normalise_messages(messages)
        kwargs = dict(
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            top_p=self.top_p if top_p is None else top_p,
            stop=stop,
            json_mode=json_mode,
        )
        result = self._chat_with_retry(msgs, kwargs)
        self.usage.add(result)
        return result

    def chat_json(
        self,
        messages: Sequence[ChatMessage] | Sequence[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        repair: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Call ``chat`` with JSON-mode hints, then parse and (optionally) repair."""
        result = self.chat(messages, json_mode=True, **kwargs)
        try:
            return _parse_json(result.text, repair=repair)
        except json.JSONDecodeError as exc:
            logger.warning(f"chat_json: failed to parse '{result.text[:200]}...': {exc}")
            if not repair:
                raise
            # Last-ditch repair: ask the LLM to fix it.
            fix = self.chat(
                [
                    {"role": "system", "content": "Return only valid JSON. No prose."},
                    {"role": "user", "content": f"Fix this to be valid JSON:\n{result.text}"},
                ],
                temperature=0.0,
            )
            return _parse_json(fix.text, repair=True)

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)  # crude fallback

    # ------------------------------------------------------------------ #
    # Retry shell                                                        #
    # ------------------------------------------------------------------ #

    def _chat_with_retry(
        self, messages: list[dict[str, str]], kwargs: dict[str, Any]
    ) -> ChatResult:
        @retry(
            stop=stop_after_attempt(self.retry_max_attempts),
            wait=wait_exponential(min=self.retry_initial_wait, max=self.retry_max_wait),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _go() -> ChatResult:
            return self._chat_once(messages, kwargs)

        return _go()

    def _chat_once(self, messages: list[dict[str, str]], kw: dict[str, Any]) -> ChatResult:
        if self.provider in {"openai", "openai_compat"}:
            extra: dict[str, Any] = {}
            if kw.get("json_mode"):
                extra["response_format"] = {"type": "json_object"}
            if kw.get("stop"):
                extra["stop"] = kw["stop"]
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kw["temperature"],
                max_tokens=kw["max_tokens"],
                top_p=kw["top_p"],
                **extra,
            )
            choice = resp.choices[0]
            usage = resp.usage
            return ChatResult(
                text=(choice.message.content or "").strip(),
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                finish_reason=choice.finish_reason,
                raw=resp,
            )
        if self.provider == "anthropic":
            system_msgs = [m["content"] for m in messages if m["role"] == "system"]
            user_msgs = [m for m in messages if m["role"] != "system"]
            resp = self._client.messages.create(
                model=self.model,
                system="\n\n".join(system_msgs) if system_msgs else None,
                messages=user_msgs,
                temperature=kw["temperature"],
                max_tokens=kw["max_tokens"],
                top_p=kw["top_p"],
                stop_sequences=kw.get("stop") or None,
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            usage = getattr(resp, "usage", None)
            return ChatResult(
                text=text.strip(),
                prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                total_tokens=(getattr(usage, "input_tokens", 0) or 0)
                + (getattr(usage, "output_tokens", 0) or 0),
                finish_reason=getattr(resp, "stop_reason", None),
                raw=resp,
            )
        raise ValueError(f"Unknown provider: {self.provider}")


# --------------------------------------------------------------------------- #
# Mock client                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class _ScriptedReply:
    pattern: re.Pattern[str] | None
    text: str


class MockLLMClient:
    """Deterministic LLM stub for unit tests.

    You can either:
    * pass ``responses=[...]`` — returned in order, then the last one repeats; or
    * register pattern→reply rules via ``when(<regex>, <reply>)``.
    """

    def __init__(self, responses: Iterable[str] | None = None, *, default: str = "OK") -> None:
        self._queue: list[str] = list(responses or [])
        self._rules: list[_ScriptedReply] = []
        self.default = default
        self.calls: list[list[dict[str, str]]] = []
        self.usage = TokenUsage()
        self.model = "mock"
        self.provider = "mock"

    def when(self, pattern: str, reply: str) -> MockLLMClient:
        self._rules.append(_ScriptedReply(re.compile(pattern, re.S | re.I), reply))
        return self

    def chat(self, messages, **_kwargs) -> ChatResult:
        msgs = _normalise_messages(messages)
        self.calls.append(msgs)
        text = self._pick(msgs)
        result = ChatResult(text=text, prompt_tokens=10, completion_tokens=5, total_tokens=15)
        self.usage.add(result)
        return result

    def chat_json(self, messages, **kwargs):
        result = self.chat(messages, **kwargs)
        return _parse_json(result.text, repair=True)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------ #

    def _pick(self, msgs: list[dict[str, str]]) -> str:
        joined = "\n".join(m["content"] for m in msgs)
        for rule in self._rules:
            if rule.pattern is None or rule.pattern.search(joined):
                return rule.text
        if self._queue:
            return self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        return self.default


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

def build_client(cfg: dict[str, Any] | Any) -> LLMClient | MockLLMClient:
    """Construct from a dict / OmegaConf cfg. Pass ``provider='mock'`` for tests."""
    cfg = dict(cfg) if not isinstance(cfg, dict) else cfg
    provider = cfg.get("provider", "openai_compat")
    if provider == "mock":
        return MockLLMClient(default=cfg.get("default", "OK"))
    retry_cfg = cfg.get("retry", {}) or {}
    return LLMClient(
        provider=provider,
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key"),
        model=cfg.get("model", "qwen7b"),
        max_tokens=int(cfg.get("max_tokens", 1024)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        timeout=float(cfg.get("timeout", 120.0)),
        retry_max_attempts=int(retry_cfg.get("max_attempts", 4)),
        retry_initial_wait=float(retry_cfg.get("initial_wait", 1.0)),
        retry_max_wait=float(retry_cfg.get("max_wait", 30.0)),
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _normalise_messages(
    messages: Sequence[ChatMessage] | Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, ChatMessage):
            out.append(m.to_openai())
        elif isinstance(m, dict):
            out.append({"role": m["role"], "content": m["content"]})
        else:
            raise TypeError(f"Bad message type: {type(m).__name__}")
    return out


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S)
_BRACE_RE = re.compile(r"(\{.*\}|\[.*\])", re.S)


def _parse_json(text: str, *, repair: bool = True) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not repair:
            raise
    # Try fenced code block first.
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Greedy outermost-brace fallback.
    m = _BRACE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    raise json.JSONDecodeError("No JSON object found", text, 0)
