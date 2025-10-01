#!/usr/bin/env python3
"""
bug_finder.py — Multi-LLM static analysis runner with resumability, dry-run, and cost tracking.

Key features
- Analyze C/C++ source files with one or two LLMs (Gemini or OpenAI-compatible).
- Best-of-N repeated analysis per file/model with temperature schedules.
- Aggregation step (optionally a different model) to consolidate N runs per file.
- Async execution with per-model concurrency and rate limiting; resilient retries.
- Robust cost and token accounting, with live progress and global/per-model budgets.
- Context-window checks; skip oversized files with visible warnings.
- Dry-run to estimate tokens/cost before running.
- Full resumability via SQLite state database; saves outputs and JSON sidecars.
- Strict Markdown templates for trial outputs and aggregation to support parsing.
- YAML configuration for models, prompts, rate limits, pricing, etc.
- HTML index to navigate aggregated outputs; detailed logs to file and console.

Dependencies
- PyYAML (yaml)
- tiktoken (for OpenAI token estimation; optional but recommended)
- google-genai (for Gemini; optional unless using Gemini)
- openai (for OpenAI-compatible endpoints; optional unless using OpenAI)
"""

# SPDX-License-Identifier: Elastic-2.0

#
# Copyright 2021 Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under
# one or more contributor license agreements. Licensed under the Elastic
# License 2.0; you may not use this file except in compliance with the Elastic
# License 2.0.


from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as dt
import functools
import hashlib
import json
import logging
import os
import random
import re
import shutil
import signal
import sqlite3
import sys
import textwrap
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Optional dependencies
try:
    import yaml  # PyYAML
except Exception as e:  # pragma: no cover
    yaml = None

try:
    import tiktoken  # For OpenAI token estimation
except Exception:
    tiktoken = None

# OpenAI (async)
try:
    from openai import AsyncOpenAI, RateLimitError, APIError
except Exception:
    AsyncOpenAI = None
    RateLimitError = None  # type: ignore
    APIError = None  # type: ignore

# Google GenAI
try:
    from google import genai as google_genai
    from google.genai import types as google_types
except Exception:
    google_genai = None
    google_types = None


# ------------------------------- Constants ------------------------------------

DEFAULT_OUTPUT_DIR = Path("llm-outputs")
DEFAULT_STATE_DIR = Path(".bugfinder")
DEFAULT_LOG_DIR = DEFAULT_STATE_DIR / "logs"
DEFAULT_DB_PATH = DEFAULT_STATE_DIR / "state.sqlite"
DEFAULT_CONFIG_PATH = Path("bugfinder.yml")

# Unified per-request timeout (seconds) for all LLM calls.
# Heavy LLM APIs can take a while; keep this conservative but consistent.
REQUEST_TIMEOUT_S: float = 300.0

SUPPORTED_LANGS = {
    "c": "c",
    "h": "c",
    "hh": "c++",
    "hpp": "c++",
    "hxx": "c++",
    "cxx": "c++",
    "cc": "c++",
    "cpp": "c++",
}

# -------------------- Token Estimation Defaults (centralized) -----------------

# Heuristic used when counting characters to tokens
CHARS_PER_TOKEN_ESTIMATE: int = 4

# Default estimated max output tokens for a single trial per provider when
# a model-specific cap is not provided via `ModelConfig.max_output_tokens`.
DEFAULT_TRIAL_OUTPUT_TOKENS_GEMINI: int = 11000
DEFAULT_TRIAL_OUTPUT_TOKENS_OPENAI: int = 17000

# Default estimated output tokens for aggregation per provider (final report).
DEFAULT_AGG_OUTPUT_TOKENS_GEMINI: int = 9000
DEFAULT_AGG_OUTPUT_TOKENS_OPENAI: int = 9000

# Estimated overhead tokens for aggregation user prompt structure (headers,
# separators between drafts, etc.). Kept modest to avoid double-counting.
DEFAULT_AGG_INPUT_OVERHEAD_TOKENS: int = 500

# Default prompts (core + per-language adapter)
CORE_ANALYSIS_DIRECTIVE = textwrap.dedent(
    """
    You are an expert code reviewer for systems code. Your goal is to find real,
    impactful defects and bugs.
    Focus on (among others):
    - Undefined behavior (UB), lifetime/ownership, buffer over/underflows, off-by-one, alignment issues
    - Concurrency issues: data races, atomics misuse, improper synchronization, deadlocks, lock ordering
    - Memory: overflow, off-by-one errors, leaks, double free, invalid frees, UAF, dangling pointers, aliasing/strict-aliasing violations
    - Error handling: ignored error codes, exceptions, panic/abort risks, resource leaks on error paths
    - API misuse, reentrancy, TOCTOU, integer overflow/underflow, signedness issues, path traversal, etc.
    - Any logic bugs you can find

    Guidance:
    - Be concrete. Cite evidence with line ranges and code excerpts.
    - Prefer high-signal findings over stylistic nitpicks.
    - If a macro or function is undefined in the snippet, infer intent from names/types and reason prudently.
    - If uncertain, note uncertainty and suggest tests/invariants to validate.
    - Do not propose trivial refactors unless they affect correctness or robustness.
    """).strip()

LANG_SPECIFIC = {
    "c": "Language context: C. Watch for manual memory management, UB, and concurrency primitives.",
    "c++": "Language context: C++. Watch for RAII, copy/move/ownership, exception safety, UB, and concurrency.",
}


TRIAL_OUTPUT_MD_TEMPLATE = textwrap.dedent(
    """
    # File Review (Trial {trial_index}/{best_of}) — {file_path}

    ## Summary
    Provide a 3–7 sentence summary of the most significant risks and areas to review.

    ## Findings
    Organize by severity: Critical, High, Medium, Low. For each finding, include:
    - ID: F{{index}}
    - Title: Short summary
    - Severity: Critical|High|Medium|Low
    - Confidence: 0.0–1.0 (likelihood that this is a real bug)
    - Evidence: Quote relevant code lines or describe the path; include line ranges
    - Why it’s a bug: Explain the defect
    - Fix: Practical remediation steps or patterns
    - Tests: Assertions/unit/integration tests to catch or reproduce

    ## False Positives or Uncertain
    List any items you flagged but consider likely false positives, with rationale.

    ## Notes
    Any caveats about missing context or environment dependencies.

    -- DO NOT MODIFY THE SECTION TITLES ABOVE. USE CLEAR MARKDOWN STRUCTURE. --
    """).strip()

AGGREGATION_PROMPT = textwrap.dedent(
    """
    You are consolidating multiple independent static-analysis drafts into one final engineering review.
    Objectives:
    - Elevate unique or rare but critical findings that appear only in one draft.
    - De-duplicate overlapping findings; cluster similar ones.
    - Assign severity and confidence per consolidated finding.
    - Mark likely false positives and note uncertainty clearly.
    - Provide specific fixes and tests to validate behavior.
    - Keep it focused and engineering-grade; avoid style-only nits.

    Output strictly in the following sections and sequence.
    At the very end, include a JSON code block with `findings` array, one object per consolidated finding:
    { "id": "F1", "title": "...", "severity": "Critical|High|Medium|Low", "confidence": 0.0-1.0, "evidence": ["..."], "why": "...", "fix": "...", "tests": ["..."] }

    REQUIRED MARKDOWN STRUCTURE:
    # Final Review — {file_path}

    ## Executive Summary
    A compact summary of critical risks and overall quality.

    ## Consolidated Findings
    Grouped by severity. For each finding, include:
    - ID: Fx
    - Title
    - Severity
    - Confidence
    - Evidence
    - Why it’s a bug
    - Fix
    - Tests

    ## Model Disagreements or Uncertain Items
    Note any items that were not consistent across drafts, and what to verify.

    ## Recommendations
    Prioritized next steps and suggested guardrails (assertions, fuzzing, sanitizers, CI checks).

    ## Machine-Readable Findings (JSON)
    ```
    {
      "findings": [
        {
          "id": "F1",
          "title": "Example",
          "severity": "High",
          "confidence": 0.8,
          "evidence": ["..."],
          "why": "...",
          "fix": "...",
          "tests": ["..."]
        }
      ]
    }
    ```
    """).strip()

# ------------------------------- Dataclasses ----------------------------------


@dataclass
class TierRate:
    upto_input_tokens: int
    input_cost_per_token_usd: float
    output_cost_per_token_usd: float
    cached_input_cost_per_token_usd: Optional[float] = None


@dataclass
class PricingConfig:
    input_cost_per_token_usd: float = 0.0
    output_cost_per_token_usd: float = 0.0
    cached_input_cost_per_token_usd: Optional[float] = None
    tiers: List[TierRate] = field(default_factory=list)

    def compute_cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        non_cached_input = input_tokens - cached_input_tokens
        
        # Determine the correct tier based on total input tokens
        tier = None
        if self.tiers:
            for t in sorted(self.tiers, key=lambda x: x.upto_input_tokens):
                if input_tokens <= t.upto_input_tokens:
                    tier = t
                    break
            if tier is None:
                tier = sorted(self.tiers, key=lambda x: x.upto_input_tokens)[-1]
        
        # Get prices from the tier or the base config
        input_price = tier.input_cost_per_token_usd if tier else self.input_cost_per_token_usd
        output_price = tier.output_cost_per_token_usd if tier else self.output_cost_per_token_usd
        cached_price = (tier.cached_input_cost_per_token_usd if tier and tier.cached_input_cost_per_token_usd is not None 
                        else self.cached_input_cost_per_token_usd)

        # If no specific cached price, use the regular input price (no discount)
        if cached_price is None:
            cached_price = input_price

        cost = (non_cached_input * input_price) + (cached_input_tokens * cached_price) + (output_tokens * output_price)
        return cost


@dataclass
class RateLimitConfig:
    rpm: Optional[int] = None  # requests per minute
    # tpm: Optional[int] = None  # tokens per minute (not implemented)


@dataclass
class ModelConfig:
    key: str
    provider: str  # "gemini" or "openai_compat"
    model_name: str
    context_window: int
    pricing: PricingConfig
    base_url: Optional[str] = None  # for openai_compat
    api_key_env: Optional[str] = None
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    concurrency: int = 2
    temperature_schedule: Optional[List[float]] = None
    use_count_tokens_in_dry_run: bool = False  # Gemini only; if free
    max_output_tokens: Optional[int] = None  # optional cap
    thinking_budget: Optional[int] = None  # Gemini-specific; can be overridden via CLI
    request_timeout_s: Optional[float] = None  # per-model override for request timeout


@dataclass
class RunConfig:
    models: Dict[str, ModelConfig]
    best_of: int = 3
    aggregator_model_key: Optional[str] = None
    include_globs: List[str] = field(default_factory=lambda: ["**/*.c", "**/*.h", "**/*.cc", "**/*.cpp", "**/*.cxx", "**/*.hh", "**/*.hpp", "**/*.hxx"])
    exclude_globs: List[str] = field(default_factory=lambda: [".git/**", "build/**", "dist/**", "out/**", "target/**", "vendor/**", "third_party/**"])
    follow_symlinks: bool = True
    output_dir: Path = DEFAULT_OUTPUT_DIR
    state_dir: Path = DEFAULT_STATE_DIR
    max_usd_total: Optional[float] = None
    per_model_max_usd: Dict[str, float] = field(default_factory=dict)


@dataclass
class FileInfo:
    path: Path
    language: str
    size_bytes: int
    num_lines: int
    rel_path: str  # relative to cwd


@dataclass
class TrialResult:
    ok: bool
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    error: Optional[str] = None


# ------------------------------- Utilities ------------------------------------


def now_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def detect_language(path: Path) -> Optional[str]:
    ext = path.suffix.lower().lstrip(".")
    return SUPPORTED_LANGS.get(ext)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    """Write text atomically so partial writes do not leak into final files."""
    ensure_dirs(path.parent)
    tmp_path = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Same as write_text_atomic but for JSON payloads."""
    ensure_dirs(path.parent)
    tmp_path = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def human_usd(x: float) -> str:
    return f"${x:,.4f}"


def human_tokens(x: int) -> str:
    return f"{x:,d} tok"


def _parse_retry_after_seconds(err: Exception) -> Optional[float]:
    """Attempt to extract Retry-After seconds from an exception's response headers.

    Supports OpenAI-style exceptions and generic httpx-style attributes when surfaced.
    Returns None if not available/parseable.
    """
    # Try OpenAI exceptions exposing a response with headers
    resp = getattr(err, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers:
            for k in ("retry-after", "Retry-After"):
                if k in headers:
                    try:
                        return float(headers[k])
                    except Exception:
                        pass
    # Some exceptions may carry .headers directly
    headers = getattr(err, "headers", None)
    if headers and isinstance(headers, dict):
        for k in ("retry-after", "Retry-After"):
            if k in headers:
                try:
                    return float(headers[k])
                except Exception:
                    pass
    return None


def _is_rate_limited(err: Exception) -> bool:
    if RateLimitError is not None and isinstance(err, RateLimitError):
        return True
    msg = str(err)
    if "429" in msg or "rate limit" in msg.lower() or "TooManyRequests" in msg:
        return True
    # Bedrock sometimes returns 500 with throttling messages
    if "throttl" in msg.lower():
        return True
    return False


def _backoff_delay_seconds(
    attempt: int,
    err: Optional[Exception],
    provider: str,
    base_url: Optional[str],
) -> float:
    """Provider-aware exponential backoff with jitter.

    - Honors Retry-After if present.
    - Heavier defaults for Bedrock-style OpenAI endpoints.
    - Full-jitter: sleep ~ U(0, min(cap, base*2**(attempt-1))).
    """
    # Retry-After takes precedence when available
    if err is not None:
        ra = _parse_retry_after_seconds(err)
        if ra is not None and ra > 0:
            # Add small jitter to spread contention
            return max(0.0, ra + random.uniform(0.25, 1.25))

    is_bedrock = bool(base_url and "bedrock" in base_url)
    is_rate = _is_rate_limited(err) if err is not None else False
    if provider == "openai_compat" and is_bedrock and isinstance(err, asyncio.TimeoutError):
        # Treat timeouts from Bedrock as a heavy throttle event.
        is_rate = True

    if provider == "openai_compat" and is_bedrock:
        # Bedrock can enforce very low RPM/TPM. Start high and cap high.
        # First retry on rate limit should be ~30s.
        base = 30.0 if is_rate else 12.0
        cap = 300.0
    elif provider == "openai_compat":
        base = 6.0 if is_rate else 3.0
        cap = 150.0
    elif provider == "gemini":
        base = 8.0 if is_rate else 2.0
        cap = 60.0
    else:
        base = 5.0
        cap = 60.0

    # Full-jitter exponential backoff (special-case first retry for Bedrock rate limits)
    expo = base * (2 ** max(0, attempt - 1))
    limit = min(cap, expo)
    if provider == "openai_compat" and is_bedrock and is_rate:
        if attempt == 1:
            # Aim around 30s (±10%) on first retry when rate-limited by Bedrock
            sleep = random.uniform(0.9 * limit, 1.1 * limit)
        else:
            # From attempt 2+, never go below the computed limit; allow slight overshoot.
            sleep = random.uniform(limit, min(cap, 1.25 * limit))
    else:
        sleep = random.uniform(0.5 * limit, limit)
    # Ensure we never spin too fast
    return max(1.0, sleep)


def default_temperature_schedule(n: int) -> List[float]:
    if n <= 1:
        return [0.3]
    if n == 2:
        return [1.0, 0.2]
    if n == 3:
        return [1.0, 0.5, 0.2]
    # For n > 3, interpolate between 1.0 and 0.2
    vals = []
    for i in range(n):
        if n == 1:
            vals.append(0.3)
        else:
            t = 1.0 - (0.8 * i / (n - 1))  # 1.0 .. 0.2
            vals.append(round(t, 2))
    return vals


def sanitize_model_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", s)


def sanitize_rel_path(path: Path) -> str:
    cwd = Path.cwd()
    try:
        rel = path.relative_to(cwd)
    except Exception:
        rel = _external_rel_path(path)
    return str(rel).replace(os.sep, "/")


def _external_rel_path(path: Path) -> Path:
    anchor = path.anchor
    parts: List[str] = []
    for part in path.parts:
        if not part or part == anchor or part == os.sep:
            continue
        parts.append(_sanitize_path_component(part))
    if not parts:
        parts.append("unknown")
    return Path("external").joinpath(*parts)


def _sanitize_path_component(part: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", part)
    if clean in ("", ".", ".."):
        clean = clean.replace(".", "_") or "_"
    return clean


# ------------------------------- Token Estimation -----------------------------


class TokenEstimator:
    def __init__(self, model_cfg: ModelConfig):
        self.model_cfg = model_cfg
        self.model_name = model_cfg.model_name
        self.logger = logging.getLogger("bug_finder")

        # Optional OpenAI tiktoken encoding
        self._tok = None
        if self.model_cfg.provider == "openai_compat" and tiktoken is not None:
            try:
                self._tok = tiktoken.encoding_for_model(self.model_name)
                self.logger.info(f"[{self.model_cfg.key}] Loaded tiktoken encoding for model '{self.model_name}'")
            except Exception:
                try:
                    if "gpt-oss" in self.model_name:
                        picked_encoding = "o200k_harmony"
                    else:
                        picked_encoding = "cl100k_base"
                    self._tok = tiktoken.get_encoding(picked_encoding)
                    self.logger.info(f"[{self.model_cfg.key}] Model '{self.model_name}' not found, falling back to '{picked_encoding}' tiktoken encoding")
                except Exception:
                    self._tok = None

    def estimate(self, text: str) -> int:
        # For OpenAI-compatible, try tiktoken
        if self.model_cfg.provider == "openai_compat" and self._tok is not None:
            try:
                return len(self._tok.encode(text))
            except Exception:
                pass
        # For Gemini or unknown, use heuristic ~ 4 chars/token for code
        self.logger.warning(f"[{self.model_cfg.key}] Using rough token estimation heuristic (len/{CHARS_PER_TOKEN_ESTIMATE})")
        return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))

    async def count_gemini_tokens(self, client: Any, text: str, is_dry_run: bool = False) -> Optional[int]:
        # Use Gemini count_tokens if available and requested; avoid if not free (assumed free)
        if self.model_cfg.provider != "gemini":
            return None
        if is_dry_run and not self.model_cfg.use_count_tokens_in_dry_run:
            return None
        if client is None:
            return None
        try:
            loop = asyncio.get_running_loop()
            # google-genai client is sync; run in thread
            def _count():
                return client.models.count_tokens(model=self.model_cfg.model_name, contents=text)
            resp = await loop.run_in_executor(None, _count)
            if hasattr(resp, "total_tokens"):
                return int(resp.total_tokens)
        except Exception:
            return None
        return None


# ------------------------------- Rate Limiter ---------------------------------


class AsyncRateLimiter:
    def __init__(self, rpm: Optional[int]):
        self.rpm = rpm
        self._lock = asyncio.Lock()
        self._events: List[float] = []

    async def acquire(self) -> None:
        if not self.rpm or self.rpm <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            window = 60.0
            # Prune old events
            self._events = [t for t in self._events if now - t < window]
            if len(self._events) < self.rpm:
                self._events.append(now)
                return
            # Need to wait until earliest event leaves window
            earliest = self._events[0]
            wait_s = max(0.0, window - (now - earliest))
            await asyncio.sleep(wait_s + 0.01)
            # After sleep, record now
            now2 = time.monotonic()
            self._events = [t for t in self._events if now2 - t < window]
            self._events.append(now2)


# ------------------------------- LLM Clients ----------------------------------


class BaseLLMClient:
    def __init__(self, cfg: ModelConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    async def generate(
        self,
        user_prompt: str,
        temperature: float,
        system_prompt: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> TrialResult:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class OpenAICompatClient(BaseLLMClient):
    def __init__(self, cfg: ModelConfig, logger: logging.Logger):
        super().__init__(cfg, logger)
        if AsyncOpenAI is None:
            raise RuntimeError("openai package not installed; required for openai_compat provider")
        api_key = os.getenv(cfg.api_key_env) if cfg.api_key_env else None
        base_url = cfg.base_url
        # Disable the SDK's internal short backoff to control retries ourselves.
        # These calls can be long-running and heavily rate limited (e.g., Bedrock).
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    async def generate(
        self,
        user_prompt: str,
        temperature: float,
        system_prompt: Optional[str] = None,
        thinking_budget: Optional[int] = None,  # ignored
        max_output_tokens: Optional[int] = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> TrialResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        kwargs = dict(
            model=self.cfg.model_name,
            messages=messages,
            temperature=temperature,
            reasoning_effort="high"
        )
        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens

        # Retry strategy: provider-aware exponential backoff with jitter + Retry-After
        max_retries = 8
        for attempt in range(1, max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(**kwargs),
                    timeout=timeout_s,
                )
                text = resp.choices[0].message.content or ""
                reasoning_prefix = "<reasoning>"
                reasoning_suffix = "</reasoning>"
                if text.startswith(reasoning_prefix):
                    end_idx = text.find(reasoning_suffix, len(reasoning_prefix))
                    if end_idx != -1:
                        text = text[end_idx + len(reasoning_suffix):]
                usage = getattr(resp, "usage", None)
                in_tok = getattr(usage, "prompt_tokens", None) or 0
                out_tok = getattr(usage, "completion_tokens", None) or 0
                cost = self.cfg.pricing.compute_cost(in_tok, out_tok)
                return TrialResult(ok=True, text=text, input_tokens=in_tok, output_tokens=out_tok, cached_input_tokens=0, cost_usd=cost)
            except asyncio.TimeoutError:
                err_obj = asyncio.TimeoutError()
                err = f"Timeout after {timeout_s}s"
            except Exception as e:
                err_obj = e
                err = f"{type(e).__name__}: {e}"

            if attempt < max_retries:
                sleep_s = _backoff_delay_seconds(
                    attempt=attempt,
                    err=locals().get("err_obj"),
                    provider=self.cfg.provider,
                    base_url=self.cfg.base_url,
                )
                self.logger.warning(
                    f"[{self.cfg.key}] generate attempt {attempt} failed: {err}; retrying in {sleep_s:.1f}s"
                )
                await asyncio.sleep(sleep_s)
                continue
            self.logger.error(f"[{self.cfg.key}] generate final failure: {err}")
            return TrialResult(ok=False, text="", input_tokens=0, output_tokens=0, cached_input_tokens=0, cost_usd=0.0, error=err)

    async def close(self) -> None:
        # openai AsyncOpenAI doesn't need explicit close
        return


class GeminiClient(BaseLLMClient):
    def __init__(self, cfg: ModelConfig, logger: logging.Logger):
        super().__init__(cfg, logger)
        if google_genai is None or google_types is None:
            raise RuntimeError("google-genai package not installed; required for gemini provider")
        api_key = os.getenv(cfg.api_key_env) if cfg.api_key_env else os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini API key not found; set via model.api_key_env or GEMINI_API_KEY")
        self.client = google_genai.Client(api_key=api_key)

    async def generate(
        self,
        user_prompt: str,
        temperature: float,
        system_prompt: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> TrialResult:
        # Build config
        kwargs = {}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens

        cfg_kwargs = dict(
            temperature=temperature,
            thinking_config=google_types.ThinkingConfig(
                thinking_budget=thinking_budget if thinking_budget is not None else -1
            ),
            **kwargs,
        )
        if system_prompt:
            cfg_kwargs["system_instruction"] = system_prompt
        gen_cfg = google_types.GenerateContentConfig(**cfg_kwargs)

        # The SDK is synchronous; run in a thread with timeout + retries
        def _call():
            return self.client.models.generate_content(
                model=self.cfg.model_name,
                contents=user_prompt,
                config=gen_cfg,
            )

        max_retries = 8
        for attempt in range(1, max_retries + 1):
            try:
                loop = asyncio.get_running_loop()
                resp = await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=timeout_s)
                text = getattr(resp, "text", "") or ""
                usage = getattr(resp, "usage_metadata", None)
                in_tok = 0
                out_tok = 0
                cached_tok = 0
                if usage is not None:
                    # Per Gemini docs, prompt_token_count is the total effective size including cached tokens.
                    prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
                    tool_prompt_tok = int(getattr(usage, "tool_use_prompt_token_count", 0) or 0)
                    cached_tok = int(getattr(usage, "cached_content_token_count", 0) or 0)
                    candidates_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
                    thoughts_tok = int(getattr(usage, "thoughts_token_count", 0) or 0)

                    in_tok = prompt_tok + tool_prompt_tok
                    out_tok = candidates_tok + thoughts_tok
                    self.logger.debug(
                        f"[{self.cfg.key}] usage: prompt={prompt_tok}, tool_prompt={tool_prompt_tok}, "
                        f"cached={cached_tok}, visible_out={candidates_tok}, thoughts={thoughts_tok}, "
                        f"total_in={in_tok}, total_out={out_tok}"
                    )
                
                cost = self.cfg.pricing.compute_cost(in_tok, out_tok, cached_tok)
                return TrialResult(ok=True, text=text, input_tokens=in_tok, output_tokens=out_tok, cached_input_tokens=cached_tok, cost_usd=cost)
            except asyncio.TimeoutError:
                err_obj = asyncio.TimeoutError()
                err = f"Timeout after {timeout_s}s"
            except Exception as e:
                err_obj = e
                err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                sleep_s = _backoff_delay_seconds(
                    attempt=attempt,
                    err=locals().get("err_obj"),
                    provider=self.cfg.provider,
                    base_url=self.cfg.base_url,
                )
                self.logger.warning(
                    f"[{self.cfg.key}] generate attempt {attempt} failed: {err}; retrying in {sleep_s:.1f}s"
                )
                await asyncio.sleep(sleep_s)
                continue
            self.logger.error(f"[{self.cfg.key}] generate final failure: {err}")
            return TrialResult(ok=False, text="", input_tokens=0, output_tokens=0, cached_input_tokens=0, cost_usd=0.0, error=err)


# ------------------------------- SQLite State ---------------------------------


class StateDB:
    def __init__(self, db_path: Path, logger: logging.Logger):
        self.db_path = db_path
        self.logger = logger
        ensure_dirs(db_path.parent)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                status TEXT,
                config_json TEXT,
                best_of INTEGER,
                aggregator_model_key TEXT,
                max_usd_total REAL
            );
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                path TEXT,
                rel_path TEXT,
                language TEXT,
                size_bytes INTEGER,
                num_lines INTEGER,
                sha1 TEXT,
                skipped_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                file_id INTEGER,
                model_key TEXT,
                trial_index INTEGER,
                temperature REAL,
                status TEXT,
                started_at TEXT,
                ended_at TEXT,
                prompt_text TEXT,
                output_text TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cached_input_tokens INTEGER,
                cost_usd REAL,
                output_path TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS aggregates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                file_id INTEGER,
                aggregator_model_key TEXT,
                status TEXT,
                started_at TEXT,
                ended_at TEXT,
                prompt_text TEXT,
                output_md TEXT,
                output_json TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cached_input_tokens INTEGER,
                cost_usd REAL,
                output_path TEXT,
                json_path TEXT,
                error TEXT
            );
            """)
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.commit()
            self._conn.close()
        except Exception:
            pass

    def create_run(self, run_id: str, cfg: RunConfig) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO runs (id, started_at, status, config_json, best_of, aggregator_model_key, max_usd_total) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, now_ts(), "running", json.dumps(serialize_run_config(cfg)), cfg.best_of, cfg.aggregator_model_key, cfg.max_usd_total),
        )
        self._conn.commit()

    def end_run(self, run_id: str, status: str) -> None:
        cur = self._conn.cursor()
        cur.execute("UPDATE runs SET status=?, ended_at=? WHERE id=?", (status, now_ts(), run_id))
        self._conn.commit()

    def add_file(self, run_id: str, fi: FileInfo, sha1: str, skipped_reason: Optional[str]) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO files (run_id, path, rel_path, language, size_bytes, num_lines, sha1, skipped_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, str(fi.path), fi.rel_path, fi.language, fi.size_bytes, fi.num_lines, sha1, skipped_reason),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_pending_trials(self, run_id: str) -> List[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM trials WHERE run_id=? AND status IN ('pending','running') ORDER BY id",
            (run_id,),
        )
        return list(cur.fetchall())

    def list_completed_trials_for_file(self, run_id: str, file_id: int) -> List[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM trials WHERE run_id=? AND file_id=? AND status='done' ORDER BY trial_index",
            (run_id, file_id),
        )
        return list(cur.fetchall())

    def list_trials_for_run(self, run_id: str) -> List[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM trials WHERE run_id=? ORDER BY id", (run_id,))
        return list(cur.fetchall())

    def add_trial(self, run_id: str, file_id: int, model_key: str, trial_index: int, temperature: float) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO trials (run_id, file_id, model_key, trial_index, temperature, status) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, file_id, model_key, trial_index, temperature, "pending"),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def set_trial_status(self, trial_id: int, status: str, **kwargs) -> None:
        fields = ["status"]
        values = [status]
        for k, v in kwargs.items():
            fields.append(k)
            values.append(v)
        set_clause = ", ".join(f"{f}=?" for f in fields)
        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE trials SET {set_clause} WHERE id=?",
            (*values, trial_id),
        )
        self._conn.commit()

    def append_trial_output(
        self,
        trial_id: int,
        prompt_text: str,
        output_text: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        cost_usd: float,
    ) -> None:
        """Persist the generated text and token counters even before finalizing status."""
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE trials
            SET prompt_text=?, output_text=?, input_tokens=?, output_tokens=?, cached_input_tokens=?, cost_usd=?
            WHERE id=?
            """,
            (prompt_text, output_text, input_tokens, output_tokens, cached_input_tokens, cost_usd, trial_id),
        )
        self._conn.commit()

    def add_aggregate(self, run_id: str, file_id: int, aggregator_model_key: str) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO aggregates (run_id, file_id, aggregator_model_key, status) VALUES (?, ?, ?, ?)",
            (run_id, file_id, aggregator_model_key, "pending"),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def set_aggregate_status(self, agg_id: int, status: str, **kwargs) -> None:
        fields = ["status"]
        values = [status]
        for k, v in kwargs.items():
            fields.append(k)
            values.append(v)
        set_clause = ", ".join(f"{f}=?" for f in fields)
        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE aggregates SET {set_clause} WHERE id=?",
            (*values, agg_id),
        )
        self._conn.commit()

    def list_aggregates_for_run(self, run_id: str) -> List[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM aggregates WHERE run_id=? ORDER BY id", (run_id,))
        return list(cur.fetchall())

    def append_aggregate_output(
        self,
        aggregate_id: int,
        prompt_text: str,
        output_md: str,
        output_json: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        cost_usd: float,
    ) -> None:
        """Store aggregator prompt/output text and counters prior to writing files."""
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE aggregates
            SET prompt_text=?, output_md=?, output_json=?, input_tokens=?, output_tokens=?, cached_input_tokens=?, cost_usd=?
            WHERE id=?
            """,
            (prompt_text, output_md, output_json, input_tokens, output_tokens, cached_input_tokens, cost_usd, aggregate_id),
        )
        self._conn.commit()

    def totals(self, run_id: str) -> Dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute(
            """SELECT IFNULL(SUM(input_tokens),0), IFNULL(SUM(output_tokens),0), IFNULL(SUM(cached_input_tokens),0), IFNULL(SUM(cost_usd),0.0) FROM (
            SELECT input_tokens, output_tokens, cached_input_tokens, cost_usd FROM trials WHERE run_id=? AND status='done'
            UNION ALL
            SELECT input_tokens, output_tokens, cached_input_tokens, cost_usd FROM aggregates WHERE run_id=? AND status='done'
            )""",
            (run_id, run_id),
        )
        r = cur.fetchone()
        return {"input_tokens": int(r[0]), "output_tokens": int(r[1]), "cached_input_tokens": int(r[2]), "cost_usd": float(r[3])}

    def per_model_totals(self, run_id: str) -> Dict[str, Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            """SELECT model_key, IFNULL(SUM(input_tokens),0) as in_tok, IFNULL(SUM(output_tokens),0) as out_tok, IFNULL(SUM(cached_input_tokens),0) as cached_tok, IFNULL(SUM(cost_usd),0.0) as cost 
            FROM trials WHERE run_id=? AND status='done' GROUP BY model_key""",
            (run_id,),
        )
        res = {}
        for row in cur.fetchall():
            res[row["model_key"]] = {
                "input_tokens": int(row["in_tok"]),
                "output_tokens": int(row["out_tok"]),
                "cached_input_tokens": int(row["cached_tok"]),
                "cost_usd": float(row["cost"]),
            }
        # Aggregator is keyed separately under 'aggregator:<model_key>'
        cur.execute(
            """SELECT aggregator_model_key as mk, IFNULL(SUM(input_tokens),0), IFNULL(SUM(output_tokens),0), IFNULL(SUM(cached_input_tokens),0), IFNULL(SUM(cost_usd),0.0) 
            FROM aggregates WHERE run_id=? AND status='done' GROUP BY aggregator_model_key""",
            (run_id,),
        )
        for row in cur.fetchall():
            res[f"aggregator:{row['mk']}"] = {
                "input_tokens": int(row[1]),
                "output_tokens": int(row[2]),
                "cached_input_tokens": int(row[3]),
                "cost_usd": float(row[4]),
            }
        return res

    def list_files_for_run(self, run_id: str) -> List[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM files WHERE run_id=? ORDER BY id", (run_id,))
        return list(cur.fetchall())

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM runs WHERE id=?", (run_id,))
        return cur.fetchone()

    def list_runs(self) -> List[sqlite3.Row]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM runs ORDER BY started_at DESC")
        return list(cur.fetchall())


# ------------------------------- Config Loading -------------------------------


def serialize_run_config(cfg: RunConfig) -> Dict[str, Any]:
    def pricing_to_dict(p: PricingConfig) -> Dict[str, Any]:
        return {
            "input_cost_per_token_usd": p.input_cost_per_token_usd,
            "output_cost_per_token_usd": p.output_cost_per_token_usd,
            "cached_input_cost_per_token_usd": p.cached_input_cost_per_token_usd,
            "tiers": [dataclasses.asdict(t) for t in p.tiers],
        }

    models = {}
    for k, m in cfg.models.items():
        models[k] = {
            "key": m.key,
            "provider": m.provider,
            "model_name": m.model_name,
            "context_window": m.context_window,
            "pricing": pricing_to_dict(m.pricing),
            "base_url": m.base_url,
            "api_key_env": m.api_key_env,
            "rate_limit": dataclasses.asdict(m.rate_limit),
            "concurrency": m.concurrency,
            "temperature_schedule": m.temperature_schedule,
            "use_count_tokens_in_dry_run": m.use_count_tokens_in_dry_run,
            "max_output_tokens": m.max_output_tokens,
            "thinking_budget": m.thinking_budget,
            "request_timeout_s": m.request_timeout_s,
        }
    return {
        "models": models,
        "best_of": cfg.best_of,
        "aggregator_model_key": cfg.aggregator_model_key,
        "include_globs": cfg.include_globs,
        "exclude_globs": cfg.exclude_globs,
        "follow_symlinks": cfg.follow_symlinks,
        "output_dir": str(cfg.output_dir),
        "state_dir": str(cfg.state_dir),
        "max_usd_total": cfg.max_usd_total,
        "per_model_max_usd": cfg.per_model_max_usd,
    }


def parse_yaml_config(path: Path) -> RunConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config; install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    models_cfg = {}
    models_data = data.get("models", {})
    if not models_data:
        # Provide a reasonable default Gemini 2.5 Pro with zero pricing (user should set)
        models_data = {
            "gemini": {
                "provider": "gemini",
                "model_name": "gemini-2.5-pro",
                "context_window": 1_000_000,
                "api_key_env": "GEMINI_API_KEY",
                "pricing": {
                    "input_cost_per_token_usd": 0.0,
                    "output_cost_per_token_usd": 0.0,
                    # For tiered pricing example:
                    # "tiers": [{"upto_input_tokens": 200000, "input_cost_per_token_usd": 0.00000125, "output_cost_per_token_usd": 0.00001}]
                },
                "rate_limit": {"rpm": 60},
                "concurrency": 2,
                "thinking_budget": -1,
                "use_count_tokens_in_dry_run": False,
            }
        }

    for key, m in models_data.items():
        provider = str(m["provider"]).strip()
        model_name = str(m["model_name"]).strip()
        context_window = int(m["context_window"])
        pricing_data = m.get("pricing", {}) or {}
        tiers_data = pricing_data.get("tiers") or []
        tiers = []
        for t in tiers_data:
            tiers.append(
                TierRate(
                    upto_input_tokens=int(t["upto_input_tokens"]),
                    input_cost_per_token_usd=float(t["input_cost_per_token_usd"]),
                    output_cost_per_token_usd=float(t["output_cost_per_token_usd"]),
                    cached_input_cost_per_token_usd=float(t["cached_input_cost_per_token_usd"]) if t.get("cached_input_cost_per_token_usd") is not None else None,
                )
            )
        pricing = PricingConfig(
            input_cost_per_token_usd=float(pricing_data.get("input_cost_per_token_usd", 0.0)),
            output_cost_per_token_usd=float(pricing_data.get("output_cost_per_token_usd", 0.0)),
            cached_input_cost_per_token_usd=float(pricing_data["cached_input_cost_per_token_usd"]) if pricing_data.get("cached_input_cost_per_token_usd") is not None else None,
            tiers=tiers,
        )
        rate_limit = RateLimitConfig(
            rpm=int(m.get("rate_limit", {}).get("rpm")) if m.get("rate_limit", {}).get("rpm") is not None else None
        )
        mc = ModelConfig(
            key=key,
            provider=provider,
            model_name=model_name,
            context_window=context_window,
            pricing=pricing,
            base_url=m.get("base_url"),
            api_key_env=m.get("api_key_env"),
            rate_limit=rate_limit,
            concurrency=int(m.get("concurrency", 2)),
            temperature_schedule=m.get("temperature_schedule"),
            use_count_tokens_in_dry_run=bool(m.get("use_count_tokens_in_dry_run", False)),
            max_output_tokens=int(m["max_output_tokens"]) if m.get("max_output_tokens") is not None else None,
            thinking_budget=int(m["thinking_budget"]) if m.get("thinking_budget") is not None else None,
            request_timeout_s=float(m["request_timeout_s"]) if m.get("request_timeout_s") is not None else None,
        )
        models_cfg[key] = mc

    best_of = int(data.get("best_of", 3))
    aggregator_model_key = data.get("aggregator_model_key")
    include_globs = list(map(str, data.get("include_globs", ["**/*.c", "**/*.h", "**/*.cc", "**/*.cpp", "**/*.cxx", "**/*.hh", "**/*.hpp", "**/*.hxx"])))
    exclude_globs = list(map(str, data.get("exclude_globs", [".git/**", "build/**", "dist/**", "out/**", "target/**", "vendor/**", "third_party/**"])))
    follow_symlinks = bool(data.get("follow_symlinks", True))
    output_dir = Path(data.get("output_dir", str(DEFAULT_OUTPUT_DIR)))
    state_dir = Path(data.get("state_dir", str(DEFAULT_STATE_DIR)))
    max_usd_total = data.get("max_usd_total")
    per_model_max_usd = data.get("per_model_max_usd", {}) or {}

    cfg = RunConfig(
        models=models_cfg,
        best_of=best_of,
        aggregator_model_key=aggregator_model_key,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        follow_symlinks=follow_symlinks,
        output_dir=output_dir,
        state_dir=state_dir,
        max_usd_total=float(max_usd_total) if max_usd_total is not None else None,
        per_model_max_usd={k: float(v) for k, v in per_model_max_usd.items()},
    )
    return cfg


# ------------------------------- Scanning and Planning ------------------------


def find_paths_from_cli(paths: List[str]) -> List[Path]:
    result = []
    for p in paths:
        path = Path(p).expanduser().resolve()
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            # We'll scan later using globs; collect directory
            result.append(path)
        else:
            print(f"Warning: path '{p}' not found; skipping", file=sys.stderr)
    return result


def scan_source_files(inputs: List[Path], include_globs: List[str], exclude_globs: List[str], follow_symlinks: bool = True) -> List[Path]:
    files: List[Path] = []
    # Build exclude regex from globs
    import fnmatch
    exclude_patterns = exclude_globs or []

    for inp in inputs:
        if inp.is_file():
            files.append(inp)
            continue
        # Directory: walk
        for inc in include_globs:
            for path in inp.glob(inc):
                try:
                    if path.is_dir():
                        continue
                    full = path.resolve(strict=False)
                    if not follow_symlinks and full.is_symlink():
                        continue
                    # Exclude check
                    rel_str = sanitize_rel_path(full)
                    excluded = any(fnmatch.fnmatch(rel_str, pat) for pat in exclude_patterns)
                    if excluded:
                        continue
                    # Language check
                    lang = detect_language(full)
                    if not lang:
                        continue
                    files.append(full)
                except Exception:
                    continue
    # Dedupe
    uniq = []
    seen = set()
    for f in files:
        s = str(f)
        if s not in seen:
            seen.add(s)
            uniq.append(f)
    return uniq


def build_analysis_prompt_text(file_path: Path, language: str, file_text: str, trial_index: int, best_of: int) -> str:
    lang_note = LANG_SPECIFIC.get(language, "")
    header = f"--- START OF FILE: {file_path} ---"
    footer = f"\n--- END OF FILE: {file_path} ---"
    guidance = TRIAL_OUTPUT_MD_TEMPLATE.format(
        trial_index=trial_index,
        best_of=best_of,
        file_path=sanitize_rel_path(file_path)
    )
    prompt = (
        f"{lang_note}\n\n"
        "Analyze the following source file for defects. Adhere strictly to the output structure shown below.\n\n"
        f"{header}{file_text}{footer}\n\n"
        "Return a Markdown report following this exact template:\n\n"
        f"{guidance}\n"
    )
    return prompt


def build_aggregation_prompt(file_path: Path, drafts: List[Tuple[str, str, int]], best_of: int) -> str:
    """Return the concatenated drafts as the aggregator's user prompt."""
    pieces: List[str] = []
    for model_key, draft, t_idx in drafts:
        delim = f"\n\n==== DRAFT FROM {model_key.upper()} — Trial {t_idx}/{best_of} — BEGIN ====\n"
        pieces.append(delim + draft + f"\n==== DRAFT FROM {model_key.upper()} — Trial {t_idx}/{best_of} — END ====\n")
    return "\n".join(pieces)


def combine_system_and_user(system_prompt: Optional[str], user_prompt: str) -> str:
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}"
    return user_prompt


def build_analysis_prompts(file_path: Path, language: str, file_text: str, trial_index: int, best_of: int) -> Tuple[str, str]:
    user_prompt = build_analysis_prompt_text(file_path, language, file_text, trial_index, best_of)
    return CORE_ANALYSIS_DIRECTIVE, user_prompt


def build_aggregation_prompts(file_path: Path, drafts: List[Tuple[str, str, int]], best_of: int) -> Tuple[str, str]:
    # Avoid Python str.format on AGGREGATION_PROMPT because it contains literal braces
    # for JSON examples. Just replace the {file_path} placeholder manually.
    system_prompt = AGGREGATION_PROMPT.replace("{file_path}", sanitize_rel_path(file_path))
    user_prompt = build_aggregation_prompt(file_path, drafts, best_of)
    return system_prompt, user_prompt


# ------------------------------- Execution Orchestrator -----------------------


class Orchestrator:
    def __init__(self, cfg: RunConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.db = StateDB(cfg.state_dir / "state.sqlite", logger)
        self.run_id = f"run_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.stop_event = asyncio.Event()
        self.model_clients: Dict[str, BaseLLMClient] = {}
        self.limiters: Dict[str, AsyncRateLimiter] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self._live_counters = {
            "completed_requests": 0,
            "completed_trials": 0,
            "completed_aggregates": 0,
        }

    # ------------------------ Estimation helpers (central) -------------------

    def _trial_output_tokens_est(self, m: ModelConfig) -> int:
        if m.max_output_tokens:
            return int(m.max_output_tokens)
        if m.provider == "gemini":
            return DEFAULT_TRIAL_OUTPUT_TOKENS_GEMINI
        return DEFAULT_TRIAL_OUTPUT_TOKENS_OPENAI

    def _agg_output_tokens_est(self, m: ModelConfig) -> int:
        if m.max_output_tokens:
            return int(m.max_output_tokens)
        if m.provider == "gemini":
            return DEFAULT_AGG_OUTPUT_TOKENS_GEMINI
        return DEFAULT_AGG_OUTPUT_TOKENS_OPENAI

    def runs_root(self) -> Path:
        return self.cfg.output_dir / "runs"

    def run_root(self) -> Path:
        return self.runs_root() / self.run_id

    def latest_root(self) -> Path:
        return self.cfg.output_dir / "latest"

    def _rel_dir(self, fi: FileInfo) -> Path:
        rel_dir = Path(fi.rel_path).parent
        return rel_dir if rel_dir != Path(".") else Path()

    def trial_output_path(self, model_key: str, fi: FileInfo, trial_id: int) -> Path:
        rel_dir = self._rel_dir(fi)
        safe_model = sanitize_model_key(model_key)
        base = self.run_root() / "models" / safe_model
        if rel_dir:
            base = base / rel_dir
        filename = f"{Path(fi.rel_path).name}__trial-{trial_id}.md"
        return base / filename

    def aggregate_md_path(self, fi: FileInfo) -> Path:
        rel_dir = self._rel_dir(fi)
        base = self.run_root() / "aggregated"
        if rel_dir:
            base = base / rel_dir
        return base / f"{Path(fi.rel_path).name}.md"

    def aggregate_json_path(self, fi: FileInfo) -> Path:
        rel_dir = self._rel_dir(fi)
        base = self.run_root() / "aggregated"
        if rel_dir:
            base = base / rel_dir
        return base / f"{Path(fi.rel_path).name}.json"

    def summary_json_path(self) -> Path:
        return self.run_root() / f"summary_{self.run_id}.json"

    def summary_md_path(self) -> Path:
        return self.run_root() / f"summary_{self.run_id}.md"

    def run_index_path(self) -> Path:
        return self.run_root() / f"index_{self.run_id}.html"

    def manifest_path(self) -> Path:
        return self.run_root() / "manifest.json"

    def runs_index_path(self) -> Path:
        return self.runs_root() / "index.html"

    def top_index_path(self) -> Path:
        return self.cfg.output_dir / "index.html"

    async def setup(self) -> None:
        ensure_dirs(self.cfg.output_dir, self.cfg.state_dir, DEFAULT_LOG_DIR, self.runs_root(), self.run_root())
        # Create clients, limiters, semaphores
        for k, m in self.cfg.models.items():
            if m.provider == "openai_compat":
                client = OpenAICompatClient(m, self.logger)
            elif m.provider == "gemini":
                client = GeminiClient(m, self.logger)
            else:
                raise RuntimeError(f"Unsupported provider: {m.provider}")
            self.model_clients[k] = client
            self.limiters[k] = AsyncRateLimiter(m.rate_limit.rpm if m.rate_limit else None)
            self.semaphores[k] = asyncio.Semaphore(int(m.concurrency) if m.concurrency else 1)
        # Aggregator model check
        if not self.cfg.aggregator_model_key:
            # Default to first model key
            first_key = next(iter(self.cfg.models.keys()))
            self.cfg.aggregator_model_key = first_key
            self.logger.info(f"Aggregator model not specified; defaulting to '{self.cfg.aggregator_model_key}'")
        elif self.cfg.aggregator_model_key not in self.cfg.models:
            raise RuntimeError(f"Aggregator model '{self.cfg.aggregator_model_key}' not defined in models")
        # Record run
        self.db.create_run(self.run_id, self.cfg)

    async def close(self) -> None:
        for c in self.model_clients.values():
            with contextlib.suppress(Exception):
                await c.close()
        self.db.close()

    def _model_temp_schedule(self, model_key: str) -> List[float]:
        m = self.cfg.models[model_key]
        if m.temperature_schedule:
            return list(m.temperature_schedule)
        return default_temperature_schedule(self.cfg.best_of)

    async def run(
        self,
        files: List[Path],
        dry_run: bool = False,
        resume: bool = False,
        thinking_budget_override: Optional[int] = None,
    ) -> None:
        try:
            if dry_run:
                await self._run_dry(files, thinking_budget_override)
            else:
                await self._run_real(files, resume, thinking_budget_override)
        except Exception as e:
            self.logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
            raise
        finally:
            await self.close()

    async def _run_dry(self, files: List[Path], thinking_budget_override: Optional[int]) -> None:
        # Build plan, estimate tokens and costs
        self.logger.info("Starting dry-run estimation...")
        totals_by_model: Dict[str, Dict[str, float]] = {}
        oversize_files_by_model: Dict[str, List[str]] = {}
        estimator_cache: Dict[str, TokenEstimator] = {}
        # Iterate files
        for path in files:
            rel = sanitize_rel_path(path)
            lang = detect_language(path)
            if not lang:
                continue
            text = read_text_file(path)
            num_lines = text.count("\n") + 1
            fi = FileInfo(path=path, language=lang, size_bytes=len(text.encode("utf-8")), num_lines=num_lines, rel_path=rel)

            for mk, m in self.cfg.models.items():
                estimator = estimator_cache.get(mk)
                if estimator is None:
                    estimator = TokenEstimator(m)
                    estimator_cache[mk] = estimator
                # Build representative prompt (trial_index placeholders not relevant for tokens)
                system_prompt, user_prompt = build_analysis_prompts(fi.path, fi.language, text, trial_index=1, best_of=self.cfg.best_of)
                prompt_for_est = combine_system_and_user(system_prompt, user_prompt)
                # Try gemini count_tokens if allowed
                gem_client = None
                if m.provider == "gemini" and m.use_count_tokens_in_dry_run and google_genai is not None:
                    try:
                        gem_client = google_genai.Client(api_key=os.getenv(m.api_key_env) or os.getenv("GEMINI_API_KEY"))
                    except Exception:
                        gem_client = None
                in_tokens_est = None
                if gem_client:
                    in_tokens_est = await estimator.count_gemini_tokens(gem_client, prompt_for_est, is_dry_run=True)
                if in_tokens_est is None:
                    in_tokens_est = estimator.estimate(prompt_for_est)
                # Output tokens estimate heuristic: enforce cap if configured; else provider-specific default
                trial_out_est = self._trial_output_tokens_est(m)
                # Context check: input + expected output <= window
                if in_tokens_est + trial_out_est > m.context_window:
                    oversize_files_by_model.setdefault(mk, []).append(rel)
                    continue
                # Accumulate across best_of
                tot_in = in_tokens_est * self.cfg.best_of
                tot_out = trial_out_est * self.cfg.best_of
                # Aggregation estimate: include the sum of expected trial outputs across models.
                # Attribute aggregation tokens only to the aggregator model.
                agg_runs = (self.cfg.aggregator_model_key is not None) and (self.cfg.best_of > 0) and (len(self.cfg.models) > 0)
                agg_in = 0
                agg_out = 0
                if agg_runs and mk == self.cfg.aggregator_model_key:
                    # Sum expected trial outputs across models (best_of per model)
                    total_trial_outputs_est = 0
                    for _mk, _m in self.cfg.models.items():
                        total_trial_outputs_est += self._trial_output_tokens_est(_m) * self.cfg.best_of
                    reserved_out = self._agg_output_tokens_est(m)
                    # Limit by context window after reserving output room; add structural overhead
                    agg_in = min(
                        max(0, m.context_window - reserved_out),
                        total_trial_outputs_est + DEFAULT_AGG_INPUT_OVERHEAD_TOKENS,
                    )
                    agg_out = reserved_out
                # Convert to cost
                cost_trials = m.pricing.compute_cost(tot_in, tot_out)
                cost_agg = m.pricing.compute_cost(agg_in, agg_out) if (mk == self.cfg.aggregator_model_key and agg_runs) else 0.0
                acc = totals_by_model.setdefault(mk, {"in": 0.0, "out": 0.0, "cost": 0.0})
                acc["in"] += tot_in + (agg_in if (mk == self.cfg.aggregator_model_key and agg_runs) else 0.0)
                acc["out"] += tot_out + (agg_out if (mk == self.cfg.aggregator_model_key and agg_runs) else 0.0)
                acc["cost"] += cost_trials + cost_agg

        # Print summary
        self.logger.info("Dry-run estimation complete.\n")
        total_cost = 0.0
        for mk, stats in totals_by_model.items():
            self.logger.info(
                f"Model '{mk}': est input {int(stats['in'])} tok, output {int(stats['out'])} tok, est cost {human_usd(stats['cost'])}"
            )
            total_cost += stats["cost"]
        self.logger.info(f"Estimated TOTAL cost: {human_usd(total_cost)}")
        if oversize_files_by_model:
            self.logger.warning("Files exceeding context window (skipped if running):")
            for mk, arr in oversize_files_by_model.items():
                self.logger.warning(f"  {mk}: {len(arr)} files (showing up to 10)")
                for s in arr[:10]:
                    self.logger.warning(f"    - {s}")

        # Write dry-run JSON
        dry_path = self.cfg.state_dir / f"dryrun_{self.run_id}.json"
        ensure_dirs(self.cfg.state_dir)
        with dry_path.open("w", encoding="utf-8") as f:
            json.dump({"models": totals_by_model, "oversize": oversize_files_by_model}, f, indent=2)
        self.logger.info(f"Dry-run report written to {dry_path}")

    async def _run_real(self, files: List[Path], resume: bool, thinking_budget_override: Optional[int]) -> None:
        # Install signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, lambda s=sig: self._on_signal(s))

        # Prepare records and tasks
        # If resume: not implemented across restarts with prior run_id; we resume within this process on interrupts.
        # This run will produce a new run_id.
        # Scan files, create rows
        trial_tasks: List[asyncio.Task] = []
        files_info: List[Tuple[FileInfo, int]] = []  # (fi, file_id)

        ensure_dirs(self.cfg.output_dir)
        # Add files to DB
        for path in files:
            lang = detect_language(path)
            if not lang:
                continue
            text = read_text_file(path)
            rel = sanitize_rel_path(path)
            fi = FileInfo(path=path, language=lang, size_bytes=len(text.encode("utf-8")), num_lines=text.count("\n") + 1, rel_path=rel)
            sha1 = sha1_bytes(text.encode("utf-8"))
            skip_reason = None
            file_id = self.db.add_file(self.run_id, fi, sha1, skip_reason)
            files_info.append((fi, file_id))

        # For each file, create trials per model that pass context window checks, then aggregator per file
        # Keep in-memory mapping of futures per file to await before aggregation
        per_file_futures: Dict[int, List[asyncio.Future]] = {}
        for fi, file_id in files_info:
            raw_text = read_text_file(fi.path)
            drafts_meta: List[Tuple[str, asyncio.Future, int]] = []  # (model_key, future, trial_idx)
            for mk, m in self.cfg.models.items():
                temps = self._model_temp_schedule(mk)
                for idx in range(self.cfg.best_of):
                    temperature = temps[idx % len(temps)]
                    system_prompt, user_prompt = build_analysis_prompts(
                        fi.path,
                        fi.language,
                        raw_text,
                        trial_index=idx + 1,
                        best_of=self.cfg.best_of,
                    )
                    combined_input = combine_system_and_user(system_prompt, user_prompt)
                    # Estimate to check context window
                    estimator = TokenEstimator(m)
                    est_in = None
                    if m.provider == "gemini":
                        client_wrapper = self.model_clients.get(mk)
                        if client_wrapper and hasattr(client_wrapper, "client"):
                            gemini_sdk_client = client_wrapper.client
                            if gemini_sdk_client:
                                est_in = await estimator.count_gemini_tokens(gemini_sdk_client, combined_input, is_dry_run=False)
                    if est_in is None:
                        est_in = estimator.estimate(combined_input)
                    est_out = self._trial_output_tokens_est(m)
                    if est_in + est_out > m.context_window:
                        self.logger.warning(f"[{mk}] Skipping over-large file for model (context {m.context_window}): {fi.rel_path}")
                        continue
                    # Create trial record
                    trial_id = self.db.add_trial(self.run_id, file_id, mk, idx + 1, temperature)
                    # Schedule
                    fut = asyncio.create_task(
                        self._run_single_trial(
                            fi,
                            file_id,
                            trial_id,
                            mk,
                            system_prompt,
                            user_prompt,
                            temperature,
                            thinking_budget_override,
                        )
                    )
                    drafts_meta.append((mk, fut, idx + 1))
                    trial_tasks.append(fut)
            # Store futures
            per_file_futures[file_id] = [f for (_, f, _) in drafts_meta]

            # Schedule aggregator after all futures complete (always, if configured)
            agg_key = self.cfg.aggregator_model_key
            if agg_key:
                self.logger.info(
                    f"Scheduling aggregation for {fi.rel_path} with aggregator='{agg_key}' and {len(drafts_meta)} scheduled drafts"
                )
                agg_future = asyncio.create_task(self._run_aggregate_when_ready(fi, file_id, drafts_meta, agg_key, raw_text))
                trial_tasks.append(agg_future)

        # Wait for all tasks
        # Also print progress every time a request completes (handled inside trial/aggregate funcs)
        await asyncio.gather(*trial_tasks, return_exceptions=True)

        self.db.end_run(self.run_id, "completed")
        self.logger.info("Run completed.")

        # Generate index and summary
        await self._write_summary_and_index()

    def _on_signal(self, sig: signal.Signals) -> None:
        # Hard exit immediately without attempting cleanup; artifacts already written atomically
        # Use POSIX exit code 130 for SIGINT, 143 for SIGTERM
        code = 130 if sig == signal.SIGINT else 143
        os._exit(code)

    async def _run_single_trial(
        self,
        fi: FileInfo,
        file_id: int,
        trial_id: int,
        model_key: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        thinking_budget_override: Optional[int],
    ) -> None:
        m = self.cfg.models[model_key]
        client = self.model_clients[model_key]
        limiter = self.limiters[model_key]
        sem = self.semaphores[model_key]
        # Update status
        self.db.set_trial_status(trial_id, "running", started_at=now_ts())

        # Budget guard (soft): don't schedule new if over budget
        if self.cfg.max_usd_total is not None:
            totals = self.db.totals(self.run_id)
            if totals["cost_usd"] >= self.cfg.max_usd_total:
                self.logger.error(f"Max total budget reached ({human_usd(self.cfg.max_usd_total)}). Skipping trial {trial_id}.")
                self.db.set_trial_status(trial_id, "skipped", ended_at=now_ts(), error="budget_exceeded")
                return

        # Rate limit and concurrency
        async with sem:
            await limiter.acquire()
            # Build per-call prompt with markers and exact template
            full_text = combine_system_and_user(system_prompt, user_prompt)
            # Thinking budget if Gemini
            thinking_budget = thinking_budget_override if (m.provider == "gemini" and thinking_budget_override is not None) else m.thinking_budget

            # Pick timeout: per-model override or global default
            timeout_val = (m.request_timeout_s if m.request_timeout_s is not None else REQUEST_TIMEOUT_S)
            result = await client.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                thinking_budget=thinking_budget,
                max_output_tokens=m.max_output_tokens,
                timeout_s=timeout_val,
            )

        self._live_counters["completed_requests"] += 1
        if result.ok:
            self.db.append_trial_output(
                trial_id,
                full_text,
                result.text,
                result.input_tokens,
                result.output_tokens,
                result.cached_input_tokens,
                result.cost_usd,
            )
            out_path = self.trial_output_path(model_key, fi, trial_id)
            try:
                write_text_atomic(out_path, result.text)
                status_kwargs = {
                    "ended_at": now_ts(),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cached_input_tokens": result.cached_input_tokens,
                    "cost_usd": result.cost_usd,
                    "output_path": str(out_path),
                }
            except Exception as exc:
                self.logger.error(f"Failed to write trial output for {fi.rel_path} ({model_key} trial {trial_id}): {exc}")
                status_kwargs = {
                    "ended_at": now_ts(),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cached_input_tokens": result.cached_input_tokens,
                    "cost_usd": result.cost_usd,
                    "output_path": None,
                    "error": "write_failed",
                }
            self.db.set_trial_status(trial_id, "done", **status_kwargs)
            self._live_counters["completed_trials"] += 1
            self._print_progress_line()
        else:
            # Persist prompt so failures are inspectable in DB
            self.db.append_trial_output(
                trial_id,
                full_text,
                result.text,
                0,
                0,
                0,
                0.0,
            )
            self.db.set_trial_status(
                trial_id,
                "failed",
                ended_at=now_ts(),
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=0,
                cost_usd=0.0,
                error=result.error or "unknown_error",
            )
            self._print_progress_line()

    async def _run_aggregate_when_ready(
        self,
        fi: FileInfo,
        file_id: int,
        drafts_meta: List[Tuple[str, asyncio.Future, int]],
        aggregator_model_key: str,
        raw_text: str,
    ) -> None:
        # Wait for all trials for this file (including possibly from multiple models)
        # We collect drafts that succeeded; ignore failed/skipped
        collected: List[Tuple[str, str, int]] = []
        self.logger.info(
            f"Aggregator waiting for {len(drafts_meta)} trial(s) to complete for {fi.rel_path}"
        )
        for mk, fut, t_idx in drafts_meta:
            try:
                await fut
            except Exception:
                pass
        # Load completed trial texts from DB rows (prefer in-DB text, fall back to disk)
        rows = self.db.list_completed_trials_for_file(self.run_id, file_id)
        for r in rows:
            text = r["output_text"] if r["output_text"] is not None else ""
            if not text and r["output_path"]:
                try:
                    text = Path(r["output_path"]).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = ""
            if text:
                collected.append((r["model_key"], text, r["trial_index"]))
        # Allow aggregation even with a single draft; skip only if none
        self.logger.info(f"Aggregation ready: collected {len(collected)} draft(s) for {fi.rel_path}")
        if len(collected) < 1:
            self.logger.warning(f"No drafts collected for file {fi.rel_path}; skipping aggregation.")
            return

        aggr_id = self.db.add_aggregate(self.run_id, file_id, aggregator_model_key)
        self.db.set_aggregate_status(aggr_id, "running", started_at=now_ts())

        # Build aggregation prompt and perform preflight checks
        try:
            system_prompt, user_prompt = build_aggregation_prompts(fi.path, collected, best_of=self.cfg.best_of)
            agg_prompt = combine_system_and_user(system_prompt, user_prompt)
            m = self.cfg.models[aggregator_model_key]
            client = self.model_clients[aggregator_model_key]
            limiter = self.limiters[aggregator_model_key]
            sem = self.semaphores[aggregator_model_key]

            # Context window check (hard guard)
            estimator = TokenEstimator(m)
            est_in = None
            if m.provider == "gemini":
                client_wrapper = self.model_clients.get(aggregator_model_key)
                if client_wrapper and hasattr(client_wrapper, "client"):
                    gemini_sdk_client = client_wrapper.client
                    if gemini_sdk_client:
                        est_in = await estimator.count_gemini_tokens(gemini_sdk_client, agg_prompt, is_dry_run=False)
            if est_in is None:
                est_in = estimator.estimate(agg_prompt)

            est_out = self._agg_output_tokens_est(m)
            if est_in + est_out > m.context_window:
                # Do not attempt an over-window call; record and notify clearly
                msg = (
                    f"[{aggregator_model_key}] Aggregation input (~{est_in} in + {est_out} out) exceeds context window "
                    f"({m.context_window}) for file {fi.rel_path}. Reduce best_of and/or number of models, or use an aggregator "
                    f"with a larger window."
                )
                self.logger.error(msg)
                # Persist minimal record
                with contextlib.suppress(Exception):
                    self.db.append_aggregate_output(
                        aggr_id,
                        agg_prompt,
                        "",
                        "",
                        0,
                        0,
                        0,
                        0.0,
                    )
                self.db.set_aggregate_status(
                    aggr_id,
                    "skipped",
                    ended_at=now_ts(),
                    input_tokens=0,
                    output_tokens=0,
                    cached_input_tokens=0,
                    cost_usd=0.0,
                    error="context_window_exceeded",
                )
                return
        except Exception as e:
            # Catch any setup/formatting/token-estimation issues and mark aggregate failed
            self.db.set_aggregate_status(
                aggr_id,
                "failed",
                ended_at=now_ts(),
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=0,
                cost_usd=0.0,
                error=f"setup_error:{type(e).__name__}: {e}",
            )
            self.logger.error(f"[{aggregator_model_key}] Aggregation setup failed for {fi.rel_path}: {e}")
            return

        # Rate limit + concurrency — rely on client-level retries for consistency
        async with sem:
            await limiter.acquire()
            full_prompt = agg_prompt
            thinking_budget = m.thinking_budget
            try:
                timeout_val = (m.request_timeout_s if m.request_timeout_s is not None else REQUEST_TIMEOUT_S)
                result = await client.generate(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,  # keep aggregator conservative
                    thinking_budget=thinking_budget,
                    max_output_tokens=m.max_output_tokens,
                    timeout_s=timeout_val,
                )
            except Exception as e:
                # Ensure we mark aggregate as failed if the request itself errors out
                self._live_counters["completed_requests"] += 1
                with contextlib.suppress(Exception):
                    self.db.append_aggregate_output(
                        aggr_id,
                        full_prompt,
                        "",
                        "",
                        0,
                        0,
                        0,
                        0.0,
                    )
                self.db.set_aggregate_status(
                    aggr_id,
                    "failed",
                    ended_at=now_ts(),
                    input_tokens=0,
                    output_tokens=0,
                    cached_input_tokens=0,
                    cost_usd=0.0,
                    error=f"request_error:{type(e).__name__}: {e}",
                )
                self.logger.error(f"[{aggregator_model_key}] Aggregation request failed: {e}")
                return

        self._live_counters["completed_requests"] += 1
        if result.ok:
            json_sidecar = self._extract_json_findings(result.text)
            json_text = json.dumps(json_sidecar, indent=2)
            self.db.append_aggregate_output(
                aggr_id,
                full_prompt,
                result.text,
                json_text,
                result.input_tokens,
                result.output_tokens,
                result.cached_input_tokens,
                result.cost_usd,
            )
            md_path = self.aggregate_md_path(fi)
            json_path = self.aggregate_json_path(fi)
            try:
                write_text_atomic(md_path, result.text)
                write_json_atomic(json_path, json_sidecar)
                status_kwargs = {
                    "ended_at": now_ts(),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cached_input_tokens": result.cached_input_tokens,
                    "cost_usd": result.cost_usd,
                    "output_path": str(md_path),
                    "json_path": str(json_path),
                }
            except Exception as exc:
                self.logger.error(f"Failed to write aggregate outputs for {fi.rel_path}: {exc}")
                status_kwargs = {
                    "ended_at": now_ts(),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cached_input_tokens": result.cached_input_tokens,
                    "cost_usd": result.cost_usd,
                    "output_path": None,
                    "json_path": None,
                    "error": "write_failed",
                }
            self.db.set_aggregate_status(aggr_id, "done", **status_kwargs)
            self._live_counters["completed_aggregates"] += 1
            self.logger.info(f"Aggregation completed for {fi.rel_path}; wrote {md_path}")
            self._print_progress_line()
        else:
            self.db.append_aggregate_output(
                aggr_id,
                full_prompt,
                result.text,
                "",
                0,
                0,
                0,
                0.0,
            )
            self.db.set_aggregate_status(
                aggr_id,
                "failed",
                ended_at=now_ts(),
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=0,
                cost_usd=0.0,
                error=result.error or "unknown_error",
            )
            self.logger.error(f"Aggregation failed for {fi.rel_path}: {result.error or 'unknown_error'}")
            self._print_progress_line()

    def _extract_json_findings(self, md_text: str) -> Dict[str, Any]:
        # Look for fenced JSON code block after "Machine-Readable Findings (JSON)" section
        code_block_re = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
        m = code_block_re.search(md_text)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict) and "findings" in obj and isinstance(obj["findings"], list):
                    return obj
            except Exception:
                pass
        # Fallback minimal structure
        return {"findings": [], "note": "JSON block not found or invalid in aggregator output"}

    def _print_progress_line(self) -> None:
        totals = self.db.totals(self.run_id)
        per_model = self.db.per_model_totals(self.run_id)
        parts = [f"Requests: {self._live_counters['completed_requests']}", f"Trials: {self._live_counters['completed_trials']}", f"Aggregates: {self._live_counters['completed_aggregates']}"]
        in_str = human_tokens(totals['input_tokens'])
        if totals['cached_input_tokens'] > 0:
            in_str += f" ({human_tokens(totals['cached_input_tokens'])} cached)"
        parts.append(f"In: {in_str}")
        parts.append(f"Out: {human_tokens(totals['output_tokens'])}")
        parts.append(f"Cost: {human_usd(totals['cost_usd'])}")
        # Per model short
        model_summ = []
        for mk, s in per_model.items():
            model_summ.append(f"{mk}={human_usd(s['cost_usd'])}")
        parts.append(" | " + ", ".join(sorted(model_summ)))
        self.logger.info("Progress — " + " | ".join(parts))

    async def _write_summary_and_index(self) -> None:
        run_row = self.db.get_run(self.run_id)
        totals = self.db.totals(self.run_id)
        per_model = self.db.per_model_totals(self.run_id)

        summary_json = {
            "run_id": self.run_id,
            "started_at": run_row["started_at"] if run_row else None,
            "ended_at": run_row["ended_at"] if run_row else None,
            "status": run_row["status"] if run_row else None,
            "totals": totals,
            "per_model": per_model,
        }
        write_json_atomic(self.summary_json_path(), summary_json)

        lines = [
            f"# Run Summary — {self.run_id}",
            "",
            f"- Status: {summary_json['status']}",
            f"- Started at: {summary_json['started_at']}",
            f"- Ended at: {summary_json['ended_at']}",
            f"- Input tokens: {totals['input_tokens']:,d} ({totals['cached_input_tokens']:,d} cached)",
            f"- Output tokens: {totals['output_tokens']:,d}",
            f"- Total cost: {human_usd(totals['cost_usd'])}",
            "",
            "## Per-model",
        ]
        for mk, s in sorted(per_model.items()):
            lines.append(
                f"- {mk}: in {s['input_tokens']:,d} (cached {s['cached_input_tokens']:,d}) out {s['output_tokens']:,d} cost {human_usd(s['cost_usd'])}"
            )
        write_text_atomic(self.summary_md_path(), "\n".join(lines))
        self.logger.info(f"Wrote run summary artifacts in {self.run_root()}")

        summary_text = (
            f"\nFinal Summary — Run ID: {self.run_id}\n"
            f"  Total Cost: {human_usd(totals['cost_usd'])}\n"
            f"  Total Tokens: In={human_tokens(totals['input_tokens'])} ({human_tokens(totals['cached_input_tokens'])} cached), Out={human_tokens(totals['output_tokens'])}\n"
            "  Cost Breakdown:"
        )
        for mk, s in sorted(per_model.items()):
            summary_text += f"\n    - {mk}: {human_usd(s['cost_usd'])} (In: {human_tokens(s['input_tokens'])}, Cached: {human_tokens(s['cached_input_tokens'])}, Out: {human_tokens(s['output_tokens'])})"
        self.logger.info(summary_text)

        aggregate_rows = self.db.list_aggregates_for_run(self.run_id)
        index_items = []
        for row in aggregate_rows:
            if not row["output_path"]:
                continue
            try:
                rel = Path(row["output_path"]).relative_to(self.run_root())
            except Exception:
                rel = Path(row["output_path"]).name
            index_items.append((rel.as_posix(), row["aggregator_model_key"], row["status"]))
        index_items.sort(key=lambda x: x[0])
        list_items = "".join(
            f'<li><a href="{path}">{path}</a> — {model} ({status})</li>' for path, model, status in index_items
        )
        run_index_html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Bug Finder Results — {self.run_id}</title></head>
<body>
<h1>Bug Finder Results — {self.run_id}</h1>
<p>Total cost: {human_usd(totals['cost_usd'])}, tokens in: {totals['input_tokens']:,d} ({totals['cached_input_tokens']:,d} cached), out: {totals['output_tokens']:,d}</p>
<h2>Aggregated Reports</h2>
<ul>
{list_items}
</ul>
</body></html>"""
        write_text_atomic(self.run_index_path(), run_index_html)
        self.logger.info(f"Wrote run index: {self.run_index_path()}")

        manifest = {
            "run_id": self.run_id,
            "started_at": summary_json["started_at"],
            "ended_at": summary_json["ended_at"],
            "status": summary_json["status"],
            "files": [],
        }
        files_rows = self.db.list_files_for_run(self.run_id)
        trials_rows = self.db.list_trials_for_run(self.run_id)
        aggregates_rows = self.db.list_aggregates_for_run(self.run_id)
        trials_by_file: Dict[int, List[Dict[str, Any]]] = {}
        for t in trials_rows:
            trials_by_file.setdefault(t["file_id"], []).append(
                {
                    "trial_id": t["id"],
                    "model_key": t["model_key"],
                    "trial_index": t["trial_index"],
                    "status": t["status"],
                    "output_path": t["output_path"],
                }
            )
        aggregates_by_file: Dict[int, List[Dict[str, Any]]] = {}
        for a in aggregates_rows:
            aggregates_by_file.setdefault(a["file_id"], []).append(
                {
                    "aggregate_id": a["id"],
                    "aggregator_model_key": a["aggregator_model_key"],
                    "status": a["status"],
                    "output_path": a["output_path"],
                    "json_path": a["json_path"],
                }
            )
        for f_row in files_rows:
            manifest["files"].append(
                {
                    "file_id": f_row["id"],
                    "rel_path": f_row["rel_path"],
                    "language": f_row["language"],
                    "sha1": f_row["sha1"],
                    "size_bytes": f_row["size_bytes"],
                    "num_lines": f_row["num_lines"],
                    "trials": trials_by_file.get(f_row["id"], []),
                    "aggregates": aggregates_by_file.get(f_row["id"], []),
                }
            )
        write_json_atomic(self.manifest_path(), manifest)
        self.logger.info(f"Wrote manifest: {self.manifest_path()}")

        runs_rows = self.db.list_runs()
        table_rows = []
        for run in runs_rows:
            totals_row = self.db.totals(run["id"])
            link = f"{run['id']}/index_{run['id']}.html"
            table_rows.append(
                f"<tr><td><a href=\"{link}\">{run['id']}</a></td><td>{run['started_at']}</td><td>{run['status']}</td><td>{human_usd(totals_row['cost_usd'])}</td></tr>"
            )
        runs_index_html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Bug Finder Runs</title></head>
<body>
<h1>Bug Finder Runs</h1>
<table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">
<thead><tr><th>Run</th><th>Started</th><th>Status</th><th>Total Cost</th></tr></thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</body></html>"""
        write_text_atomic(self.runs_index_path(), runs_index_html)
        write_text_atomic(
            self.top_index_path(),
            "<!doctype html><html><head><meta http-equiv=\"refresh\" content=\"0; url=runs/index.html\"></head><body>Redirecting to runs index…</body></html>",
        )
        self.logger.info(f"Updated runs index at {self.runs_index_path()} and top-level redirect")

        # Try to create a symlink 'latest' -> this run. If symlinks are unsupported
        # or creation fails (e.g., permissions), we simply skip creating 'latest'.
        latest_link = self.latest_root()
        symlink_ok = False
        try:
            os.symlink(self.run_root(), latest_link, target_is_directory=True)
            symlink_ok = True
            self.logger.info(f"Updated 'latest' symlink -> {self.run_root()}")
        except Exception as exc:
            # Keep it quiet; we'll just point users to the run folder.
            self.logger.debug(f"Could not create 'latest' symlink: {exc}")

        # Concise outro to help users find the main artifact (.html)
        if symlink_ok:
            self.logger.info(
                f"Results: {latest_link}/index_{self.run_id}.html | All runs: {self.runs_index_path()}"
            )
        else:
            self.logger.info(
                f"Results: {self.run_root()}/index_{self.run_id}.html | All runs: {self.runs_index_path()}"
            )


# ------------------------------- CLI ------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bug_finder.py", description="Analyze C/C++ files with Gemini and/or OpenAI-compatible LLMs (best-of-N, aggregation, cost tracking).")
    p.add_argument("paths", nargs="+", help="Paths to files or directories to scan")
    p.add_argument("--config", "-c", default=str(DEFAULT_CONFIG_PATH), help="YAML config file (default bugfinder.yml)")
    p.add_argument("--models", "-m", nargs="*", help="Subset of model keys to use (defaults to all defined in config)")
    p.add_argument("--aggregator", "-a", help="Model key to use as aggregator (default: first model in config)")
    p.add_argument("--best-of", "-N", type=int, help="Best-of-N per file per model (default from config, typically 3)")
    p.add_argument("--dry-run", action="store_true", help="Estimate costs and tokens without running")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run (same process); in this version, this flag is a no-op placeholder")
    p.add_argument("--thinking-budget", type=int, help="Gemini-only: override thinking budget per request (-1 dynamic, 0 off)")
    p.add_argument("--max-usd", type=float, help="Stop scheduling new work once total estimated spend reaches this USD amount")
    p.add_argument("--concurrency", help="Per-model concurrency like modelA=3,modelB=5 (overrides config)")
    p.add_argument("--rpm", help="Per-model requests-per-minute like modelA=60,modelB=100 (overrides config)")
    p.add_argument("--include", nargs="*", help="Glob patterns to include (in addition to config)")
    p.add_argument("--exclude", nargs="*", help="Glob patterns to exclude (in addition to config)")
    p.add_argument("--output-dir", help=f"Output directory (default {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--state-dir", help=f"State directory (default {DEFAULT_STATE_DIR})")
    p.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
    return p


def parse_kv_map(s: Optional[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not s:
        return out
    for part in s.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = int(v.strip())
        except Exception:
            continue
    return out


def configure_logging(level: str, state_dir: Path) -> logging.Logger:
    ensure_dirs(state_dir / "logs")
    logger = logging.getLogger("bug_finder")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Prevent duplicate emission via root handlers if other libs configured logging
    # Clear any existing handlers (in case of re-entry) and disable propagation.
    if logger.handlers:
        logger.handlers.clear()
    logger.propagate = False
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)
    # File
    fh = logging.FileHandler(state_dir / "logs" / f"run_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)
    # Optionally quiet noisy Gemini SDK logs while keeping httpx visible
    with contextlib.suppress(Exception):
        logging.getLogger("google_genai").setLevel(logging.WARNING)
        logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    return logger


# ------------------------------- Main -----------------------------------------


async def main_async(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Load config
    cfg = parse_yaml_config(Path(args.config)) if args.config else parse_yaml_config(DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else Path(args.config))

    # Apply CLI overrides
    if args.models:
        # Filter to only specified
        cfg.models = {k: v for k, v in cfg.models.items() if k in set(args.models)}
        if not cfg.models:
            print("No matching models after filtering by --models", file=sys.stderr)
            return 2
    if args.aggregator:
        cfg.aggregator_model_key = args.aggregator
    if args.best_of:
        cfg.best_of = int(args.best_of)
    if args.max_usd is not None:
        cfg.max_usd_total = float(args.max_usd)
    if args.concurrency:
        conc = parse_kv_map(args.concurrency)
        for k, v in conc.items():
            if k in cfg.models:
                cfg.models[k].concurrency = v
    if args.rpm:
        rpm_ov = parse_kv_map(args.rpm)
        for k, v in rpm_ov.items():
            if k in cfg.models:
                cfg.models[k].rate_limit.rpm = v
    if args.include:
        cfg.include_globs.extend(args.include)
    if args.exclude:
        cfg.exclude_globs.extend(args.exclude)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.state_dir:
        cfg.state_dir = Path(args.state_dir)

    logger = configure_logging(args.log_level, cfg.state_dir)

    # Validate aggregator
    if cfg.aggregator_model_key and cfg.aggregator_model_key not in cfg.models:
        logger.error(f"Aggregator model '{cfg.aggregator_model_key}' not found in models")
        return 2

    # Prepare files
    input_paths = find_paths_from_cli(args.paths)
    if not input_paths:
        logger.error("No valid input paths provided")
        return 2
    files = scan_source_files(input_paths, cfg.include_globs, cfg.exclude_globs, follow_symlinks=cfg.follow_symlinks)
    if not files:
        logger.error("No source files found after applying filters")
        return 2
    logger.info(f"Found {len(files)} source files to consider")

    orch = Orchestrator(cfg, logger)
    await orch.setup()
    try:
        await orch.run(files, dry_run=args.dry_run, resume=args.resume, thinking_budget_override=args.thinking_budget)
    finally:
        with contextlib.suppress(Exception):
            await orch.close()
    return 0


def main() -> None:
    if sys.version_info < (3, 9):
        print("Python 3.9+ required", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
