"""API clients (embedding, GLM chat, Jev). Every call is cached by request hash and returns
a CallMeta with the token counts / latency / retries that go into the JSONL records."""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field

import httpx
import numpy as np

from .cache import ApiCache, request_hash
from .config import EmbedCfg, GLMCfg, JevCfg

RETRY_STATUS = {0, 408, 409, 429, 500, 502, 503, 504, 529}  # 0 = connection error / timeout


def _status_of(e) -> int:
    """HTTP status of an openai.APIStatusError. A 429 that says the account is out of credit is a
    permanent failure (reported as 402), not rate limiting, so it is not retried."""
    text = str(e).lower()
    if e.status_code == 429 and any(k in text for k in (
            "insufficient_quota", "credit_balance", "no credits",
            "insufficient balance", "no resource package")):  # last two: z.ai code 1113
        return 402
    return e.status_code


def _retry_after(text: str) -> float | None:
    """Parse "Please try again in 26.127s" / "...in 350ms" from a rate-limit message."""
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)", text)
    if not m:
        return None
    return float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)


class ApiError(Exception):
    def __init__(self, status: int, message: str = ""):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retry_after = _retry_after(message)
        # Provider content filters (e.g. z.ai code 1301) can fire on generated text, so they are not
        # always repeatable: worth a couple of retries before giving up on the question.
        self.content_filter = status == 400 and ("1301" in message or "unsafe or sensitive" in message)


@dataclass
class CallMeta:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0  # end-to-end including retry back-off (rate-limit waits show up here)
    retry_wait_ms: float = 0.0
    http_status: int = 200
    retries: int = 0
    input_hash: str = ""
    n_calls: int = 1  # >1 for aggregated Jev batch scoring (jev_rank)
    cached: bool = False  # latency/retries/tokens are those of the original live call

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMResult:
    content: str
    tool_calls: list[dict] = field(default_factory=list)  # [{id, name, arguments}]
    finish_reason: str | None = None
    meta: CallMeta | None = None


@dataclass
class JevResult:
    answers: dict
    meta: CallMeta


def call_with_retries(fn, max_retries: int, sleep=time.sleep, base: float = 1.0, cap: float = 60.0):
    """Run fn(); retry on RETRY_STATUS with jittered exponential back-off.

    Returns (value, retries, retry_wait_ms, status_of_last_failed_attempt_or_200).
    """
    retries, waited, filter_hits = 0, 0.0, 0
    while True:
        try:
            return fn(), retries, waited * 1000.0
        except ApiError as e:
            if e.content_filter and filter_hits < 2:
                filter_hits += 1
                retries += 1
                sleep(1.0)
                waited += 1.0
                continue
            if e.status not in RETRY_STATUS or retries >= max_retries:
                raise
            delay = min(cap, base * (2**retries)) * (0.5 + random.random() / 2)
            if e.retry_after is not None:  # the server told us how long to wait
                delay = max(delay, e.retry_after + 0.5)
            sleep(delay)
            waited += delay
            retries += 1


class TokenBucket:
    def __init__(self, rate: float, capacity: float | None = None):
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        n = min(n, self.capacity)
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / self.rate
            time.sleep(wait)


def _get_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"environment variable {env_name} is not set")
    return key


class EmbedClient:
    def __init__(self, cfg: EmbedCfg, cache: ApiCache | None = None, client=None):
        self.cfg = cfg
        self.cache = cache
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=_get_key(cfg.api_key_env), base_url=cfg.base_url, max_retries=0)
        self._client = client
        # Client-side TPM throttle, shared by all worker threads (OpenAI's embedding limit is per account tier).
        self._bucket = (TokenBucket(cfg.tokens_per_min / 60.0, capacity=cfg.tokens_per_min / 60.0 * 20)
                        if cfg.tokens_per_min else None)

    def _raw(self, texts: list[str], est_tokens: float | None = None) -> tuple[list[list[float]], int]:
        import openai

        if self._bucket is not None:  # exact token count if the caller has it, else a chars->tokens guess
            self._bucket.acquire(est_tokens if est_tokens is not None else sum(len(t) for t in texts) / 3.0)

        try:
            resp = self._client.embeddings.create(model=self.cfg.model, input=texts)
        except openai.APIStatusError as e:
            raise ApiError(_status_of(e), str(e)) from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise ApiError(0, str(e)) from e
        return [d.embedding for d in resp.data], int(resp.usage.prompt_tokens)

    def embed(self, texts: list[str], use_cache: bool = True,
              est_tokens: float | None = None) -> tuple[np.ndarray, CallMeta]:
        key = request_hash({"p": "openai-embed", "m": self.cfg.model, "t": texts})
        if use_cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                meta = CallMeta(**{**hit["meta"], "cached": True})
                return np.asarray(hit["vectors"], dtype=np.float32), meta
        t0 = time.perf_counter()
        (vecs, in_tok), retries, waited = call_with_retries(lambda: self._raw(texts, est_tokens), self.cfg.max_retries)
        meta = CallMeta(
            provider="openai", model=self.cfg.model, input_tokens=in_tok,
            latency_ms=(time.perf_counter() - t0) * 1000, retry_wait_ms=waited, retries=retries,
            input_hash=key,
        )
        if use_cache and self.cache is not None:
            self.cache.put(key, {"vectors": vecs, "meta": meta.to_dict()})
        return np.asarray(vecs, dtype=np.float32), meta


class GLMClient:
    """OpenAI-format chat client. Reasoning tokens are read from
    usage.completion_tokens_details.reasoning_tokens when the provider returns them."""

    def __init__(self, cfg: GLMCfg, cache: ApiCache | None = None, client=None):
        if cfg.model is None or cfg.provider is None:
            raise RuntimeError("glm.provider / glm.model are TBD in the spec; set them in the config first")
        self.cfg = cfg
        self.cache = cache
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=_get_key(cfg.api_key_env), base_url=cfg.base_url,
                            max_retries=0, timeout=cfg.timeout_s)
        self._client = client
        self._rpm = (TokenBucket(cfg.requests_per_min / 60.0, capacity=max(2.0, cfg.requests_per_min / 60.0 * 3))
                     if cfg.requests_per_min else None)

    def _params(self) -> dict:
        p: dict = {}
        if self.cfg.temperature is not None:
            p["temperature"] = self.cfg.temperature
        if self.cfg.max_tokens is not None:
            p["max_tokens"] = self.cfg.max_tokens
        if self.cfg.extra_body:
            p["extra_body"] = self.cfg.extra_body
        return p

    def _raw(self, messages, tools, tool_choice) -> dict:
        import openai

        if self._rpm is not None:  # every attempt counts against the provider's request limit
            self._rpm.acquire(1)

        kw = self._params()
        if tools:
            kw["tools"] = tools
            if tool_choice is not None:
                kw["tool_choice"] = tool_choice
        try:
            r = self._client.chat.completions.create(model=self.cfg.model, messages=messages, **kw)
        except openai.APIStatusError as e:
            raise ApiError(_status_of(e), str(e)) from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise ApiError(0, str(e)) from e
        if not r.choices:  # provider hiccup: no completion at all -> retry
            raise ApiError(502, "empty response (no choices)")
        ch = r.choices[0]
        usage = r.usage
        calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (ch.message.tool_calls or [])
        ]
        content = ch.message.content or ""
        if usage is None:
            # Some providers occasionally return a completion without `usage`. With no content it is a failed
            # response (retry); with content, keep the answer and estimate tokens (flagged) rather than lose it.
            if not content and not calls:
                raise ApiError(502, "empty response without usage")
            est_in = len(json.dumps(messages, ensure_ascii=False)) // 4
            est_out = (len(content) + sum(len(c["arguments"]) for c in calls)) // 4
            return {"content": content, "tool_calls": calls, "finish_reason": ch.finish_reason,
                    "usage": {"input": est_in, "output": est_out, "reasoning": 0}, "usage_estimated": True}
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        return {
            "content": content,
            "tool_calls": calls,
            "finish_reason": ch.finish_reason,
            "usage": {
                "input": int(usage.prompt_tokens),
                "output": int(usage.completion_tokens),
                "reasoning": reasoning,
            },
        }

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | dict | None = None, salt: str = "") -> LLMResult:
        """`salt` identifies the replicate: stochastic repeats must not share a cache entry."""
        key = request_hash({
            "p": self.cfg.provider, "m": self.cfg.model, "params": self._params(),
            "messages": messages, "tools": tools, "tool_choice": tool_choice, "salt": salt,
        })
        hit = self.cache.get(key) if self.cache is not None else None
        if hit is not None:
            data, meta = hit["data"], CallMeta(**{**hit["meta"], "cached": True})
        else:
            t0 = time.perf_counter()
            data, retries, waited = call_with_retries(
                lambda: self._raw(messages, tools, tool_choice), self.cfg.max_retries
            )
            meta = CallMeta(
                provider=self.cfg.provider, model=self.cfg.model,
                input_tokens=data["usage"]["input"], output_tokens=data["usage"]["output"],
                reasoning_tokens=data["usage"]["reasoning"],
                latency_ms=(time.perf_counter() - t0) * 1000, retry_wait_ms=waited,
                retries=retries, input_hash=key,
            )
            if self.cache is not None:
                self.cache.put(key, {"data": data, "meta": meta.to_dict()})
        return LLMResult(content=data["content"], tool_calls=data["tool_calls"],
                         finish_reason=data["finish_reason"], meta=meta)


class JevClient:
    """POST {base_url}/v1/systemone. Respects the published 40 req/s and 100K tokens/s limits
    client-side; 429/529 are retried with back-off and counted in meta.retries."""

    def __init__(self, cfg: JevCfg, cache: ApiCache | None = None, http: httpx.Client | None = None,
                 token_counter=None):
        self.cfg = cfg
        self.cache = cache
        self._http = http or httpx.Client(
            base_url=cfg.base_url, timeout=cfg.timeout_s,
            headers={"Authorization": f"Bearer {_get_key(cfg.api_key_env)}", "Content-Type": "application/json"},
            limits=httpx.Limits(max_connections=max(cfg.workers, 4) * 2),
        )
        self._req_bucket = TokenBucket(cfg.req_per_s)
        self._tok_bucket = TokenBucket(cfg.tokens_per_s)
        self._count = token_counter or (lambda s: max(1, len(s) // 3))  # rate-limit estimate only

    def _raw(self, body: dict, est_tokens: int) -> dict:
        self._req_bucket.acquire(1)
        self._tok_bucket.acquire(est_tokens)
        try:
            r = self._http.post("/v1/systemone", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise ApiError(0, str(e)) from e
        if r.status_code != 200:
            raise ApiError(r.status_code, r.text[:300])
        return r.json()

    def ask(self, state, questions: dict) -> JevResult:
        # Jev is deterministic enough to share across replicates, so no salt in the key.
        body = {"model": self.cfg.model, "state": state, "questions": questions}
        key = request_hash({"p": "typesafe", **body})
        hit = self.cache.get(key) if self.cache is not None else None
        if hit is not None:
            return JevResult(hit["answers"], CallMeta(**{**hit["meta"], "cached": True}))
        import json

        est = self._count(json.dumps(body, ensure_ascii=False))
        t0 = time.perf_counter()
        data, retries, waited = call_with_retries(lambda: self._raw(body, est), self.cfg.max_retries)
        usage = data.get("usage", {})
        meta = CallMeta(
            provider="typesafe", model=self.cfg.model,
            input_tokens=int(usage.get("input_tokens", 0)), output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000, retry_wait_ms=waited, retries=retries,
            input_hash=key,
        )
        answers = data["answers"]
        if self.cache is not None:
            self.cache.put(key, {"answers": answers, "meta": meta.to_dict()})
        return JevResult(answers, meta)
