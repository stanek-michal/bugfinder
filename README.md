# Bug Finder — Multi-LLM Static Analysis for Existing Codebases

`bug_finder.py` scans existing source files with two LLM backends (Gemini + an OpenAI-compatible endpoint) and merges their findings into one ranked report. Fast to start, highly configurable, with precise **token/cost** tracking.

---

## TL;DR — 60-second Quickstart

1. **Install & set keys**

```bash
python -m venv .venv && source .venv/bin/activate
pip install pyyaml openai google-genai tiktoken

# Gemini (required)
export GEMINI_API_KEY=...

# OpenAI-compatible (defaults to AWS Bedrock's gpt-oss-120b)
# Bedrock setup (required):
#  - In your AWS account, switch region to us-west-2 (currently the only region for gpt-oss-120b).
#  - Bedrock → API keys → Generate a short-lived key (~12 hours).
#  - Make sure your run finishes within that window.
#  - Paste the key into OPENAI_API_KEY before running:
export OPENAI_API_KEY=...
```

> **OpenAI-compatible requirement:** the model must support the `reasoning_effort` API parameter. Bug Finder sets it to **high** (max) on purpose. Any OpenAI-compatible host (Azure, Bedrock, self-hosted gateways, etc.) works if it honors that field.

2. **(Optional) Dry-run to estimate cost**

```bash
./bug_finder.py path/to/code --dry-run
```

3. **Run**

```bash
./bug_finder.py path/to/code
# or a single file:
./bug_finder.py src/foo/bar.c
```

**You’ll get**

* Per-file trial reviews + one **aggregated** report ranked by severity
* Costs & tokens per model, plus a final breakdown
* HTML index: `llm-outputs/runs/<run_id>/index_<run_id>.html`
* Inspectable state in `.bugfinder/` (SQLite DB + logs) and raw drafts in `llm-outputs/models/*`

---

## Cost & runtime heads-up

* With defaults (Gemini `gemini-3-pro` + Bedrock `gpt-oss-120b`, **2 tries/model**, max reasoning), expect **~$0.40–$1.00 per source file** depending on length.
* The script can run **4–10 minutes per file**. Start with a small set of files. If running over SSH, use `tmux`/`screen` so it isn’t interrupted.
* Use `--dry-run` first for a precise estimate; it accounts for **tiered pricing**, **cached tokens**, and context limits.
* You can cap spend with `--max-usd`.

### Why it costs time & money (by design)

* We **always read the whole file**—no shortcuts—so the model sees full context.
* We **max out reasoning** (`reasoning_effort=high`, and generous reasoning tokens). The more it thinks, the more subtle/egregious bugs it catches.
* Combined, this often beats typical coding/web agents that avoid full files and keep reasoning short due to context/cost constraints. Bug Finder trades speed/cost for **substantially better bug-finding power** on a per-file basis.

---

## How it works (high-level)

* Two backends run **the same prompt** on **the same file**; **multiple tries per model** surface different issues (LLMs are probabilistic). Backends can be different vendors or both Gemini.
* An **aggregation** pass consolidates/derisks duplicates and sorts by severity; JSON is emitted for automation.

### Limitations

Bug Finder intentionally analyzes **one source file at a time**. It **does not** read headers/other files as it hard to do reliably without bloating the context and increasing costs. Instead it infers missing definitions from names/types. It’s excellent at classic language/UB pitfalls, concurrency/memory errors, error-handling faults **within that file/component**.

It will **not** flag issues requiring cross-file usage context.

---

## Configuration (`bugfinder.yml`)

* **Backends**

  * `models.gemini`: `provider: gemini`, `model_name: gemini-2.5-pro`, **max reasoning** enabled.
  * `models.gpt-oss`: `provider: openai_compat` (routes to any OpenAI-compatible API). Default `base_url` is **Bedrock** (`us-west-2`) with `openai.gpt-oss-120b-1:0`.
    **Requires** support for `reasoning_effort` (Bug Finder sets `high`).
* **best_of: 2** ← default number of tries per model (recommended).
* **aggregator_model_key**: model used to merge trials (defaults to `gemini`).
* **Globs**: `include_globs`/`exclude_globs` to target files (defaults are C/C++).
* **Rate & concurrency**: per-model `rpm` and `concurrency`.
* **Pricing**: per-token or **tiered** (supports discounted cached tokens).
* **output_dir / state_dir**: defaults to `llm-outputs` and `.bugfinder`.
* **Budgets**: `max_usd_total`, optional per-model caps.

> Bedrock note: run in **us-west-2** and place the short-lived API key into `OPENAI_API_KEY`.

## Adapting to Other Languages

Bug Finder defaults to C/C++. To analyze other languages:

1. Update `include_globs` (and optionally `exclude_globs`) in `bugfinder.yml` so your target extensions are scheduled.
2. Extend `SUPPORTED_LANGS` in `bug_finder.py` to map new extensions to a language key, and add matching guidance in `LANG_SPECIFIC` so the prompt reflects language-specific concerns.
3. If the new language has different bug patterns, tailor `CORE_ANALYSIS_DIRECTIVE` (same file) with any additional checks or terminology.
4. Re-run `./bug_finder.py --dry-run ...` to confirm files are detected before launching a full run.

---

## Usage (CLI)

```bash
./bug_finder.py PATH... [options]
```

Common flags:

* **Selection**

  * `--include "src/**/*.c" "drivers/**/*.cpp"`
  * `--exclude "third_party/**" "build/**"`
  * `--models gemini gpt-oss`
  * `--aggregator gemini`
* **Quality / Cost**

  * `--best-of 2`
  * `--max-usd 10`
  * `--rpm gemini=5,gpt-oss=2`
  * `--concurrency gemini=2,gpt-oss=1`
  * `--thinking-budget 32768` (Gemini 2.5; `-1` dynamic.)
  * `--thinking-level high` (Gemini 3; Pro supports `low|high`, Flash supports `minimal|low|medium|high`)
* **Ops**

  * `--dry-run`
  * `--output-dir results/`  `--state-dir .state/`
  * `--log-level DEBUG`

Examples:

```bash
# Scan a repo with defaults
./bug_finder.py .

# Gemini only, best-of 2, cap spend at $5
./bug_finder.py . --models gemini --best-of 2 --max-usd 5

# Tighten targets
./bug_finder.py . --include "**/*.cc" --exclude "vendor/**" "third_party/**"
```

---

## Outputs & where to click

- `llm-outputs/runs/<run_id>/index_<run_id>.html` — entry point with links to per-file aggregated Markdown.
- `llm-outputs/runs/<run_id>/manifest.json` — compact catalog of all files with links to trials and aggregates (useful for tooling).
- `llm-outputs/runs/index.html` — all runs; top-level `llm-outputs/index.html` redirects here. `llm-outputs/latest/` may point to the newest run.
- `.bugfinder/state.sqlite` — durable state (runs, files, trials, aggregates, tokens, cost). Logs in `.bugfinder/logs/*`.

Results layout for a single run:

- Root: `llm-outputs/runs/<run_id>/`
- Trials per model: `models/<model_key>/<mirrored_rel_dir>/<file>__trial-<n>.md`
- Final reports: `aggregated/<mirrored_rel_dir>/<file>.md` and optional `aggregated/<mirrored_rel_dir>/<file>.json`

Path mirroring: `<mirrored_rel_dir>` reproduces the source file’s relative path under the run folder (files outside the CWD are placed under `external/...`). This makes it easy to map outputs back to inputs at a glance.

---

## Repo structure (expected)

```
bug_finder.py
bugfinder.yml
llm-outputs/
  runs/<run_id>/
  latest/
.bugfinder/
  state.sqlite
  logs/
```

---

## Optional: OpenTelemetry Tracing

Trace LLM calls with Jaeger for deeper visibility:

1. Install the latest OpenTelemetry CLI packages, e.g. `pip install opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation`.
2. Export telemetry environment variables:

   ```bash
   export OTEL_SERVICE_NAME=bug-finder
   export OTEL_TRACES_EXPORTER=otlp
   export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
   ```

3. Run the Jaeger all-in-one binary (default settings) and open `http://localhost:16686/` to confirm the UI loads.
4. Wrap Bug Finder with the OpenTelemetry launcher:

   ```bash
   opentelemetry-instrument python3 bug_finder.py <codebase_dir>
   ```

Jaeger will populate live traces showing request timing, full prompts, and responses.
