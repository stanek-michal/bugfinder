#!/usr/bin/env python3
"""
Standalone script to implement easy fixes from BUGFINDER markdown reports.

What it does:
- Recursively traverses ./BUGFINDER_REPORTS looking for .md files
- Each .md file contains bug reports for a single source file in the repo root
- Bugs are identified by lines like: #### F10: Some Title
- For each bug ID (F#) found, runs: agent -p "<prompt>"
  - The agent is instructed to assess if the fix is simple (<=10 LOC diff)
  - If simple: apply the fix and commit to git, then output DONE
  - If not simple or invalid bug: output NOT_SIMPLE_OR_INVALID
- Appends results to ./BUGFINDER_FIXES.txt in the format:
  <md_rel_path> - F# - DONE|NOT_SIMPLE_OR_INVALID
- Also prints results to stdout
- Skips duplicates already present in BUGFINDER_FIXES.txt (append-only)

Timeout:
- Each agent subprocess has a 10 minute timeout.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BUGFINDER_REPORTS_DIRNAME = "BUGFINDER_REPORTS"
FIXES_FILENAME = "BUGFINDER_FIXES.txt"
AGENT_CMD = "agent"
MODEL = "gpt-5.2-high"
AGENT_TIMEOUT_S = 10 * 60

# Headers start with 3+ hashes. Allow arbitrary text between hashes and the bug id,
# but require a space before "F#" and either a space OR ":" immediately after it.
# Examples matched:
# - "#### F10: Title"
# - "#### ID: F1 — Title"  (space after F1)
BUG_ID_RE = re.compile(r"^#{3,}.*?\sF(\d+)(?:\s|:)", re.MULTILINE)

@dataclass(frozen=True)
class FixKey:
    md_rel_path: str  
    bug_id: str  # e.g. F10


def find_md_files(root: Path) -> list[Path]:
    md_files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".md"):
                md_files.append(Path(dirpath) / name)
    return sorted(md_files)


def parse_bug_ids(md_text: str) -> list[str]:
    # Preserve first-seen order while de-duping
    seen: set[str] = set()
    bug_ids: list[str] = []
    for m in BUG_ID_RE.finditer(md_text):
        bug_id = f"F{int(m.group(1))}"
        if bug_id not in seen:
            seen.add(bug_id)
            bug_ids.append(bug_id)
    return bug_ids


def load_existing_fixes(fixes_file: Path) -> set[FixKey]:
    existing: set[FixKey] = set()
    if not fixes_file.exists():
        return existing

    try:
        for raw_line in fixes_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(" - ")
            if len(parts) != 3:
                continue
            md_rel_path, bug_id, verdict = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not md_rel_path or not bug_id:
                continue
            if not bug_id.startswith("F"):
                continue
            if verdict not in {"DONE", "NOT_SIMPLE_OR_INVALID"}:
                continue
            existing.add(FixKey(md_rel_path=md_rel_path, bug_id=bug_id))
    except Exception as e:
        print(f"Warning: Could not read existing fixes from {fixes_file}: {e}", file=sys.stderr)
    return existing


def build_agent_prompt(md_rel_path: str, bug_id: str) -> str:
    # Keep prompt free of double quotes to avoid confusion in shell usage examples.
    source_rel_path = md_rel_path[:-3] if md_rel_path.endswith(".md") else md_rel_path

    return (
        "You are an expert software engineer tasked with fixing a bug from an AI-generated bug report.\n"
        "\n"
        "Context:\n"
        "- Your current working directory is the repository root.\n"
        f"- There is a directory named {BUGFINDER_REPORTS_DIRNAME} in the current directory.\n"
        "- The markdown bug reports live under that directory.\n"
        "- The markdown report path I provide is relative to BUGFINDER_REPORTS.\n"
        "- Each markdown report corresponds to ONE source code file in the repo root at the same mirrored path,\n"
        "  which is the markdown path with the trailing .md removed.\n"
        "\n"
        "Your task:\n"
        f"- Fix ONLY bug {bug_id} in the markdown report {md_rel_path}, and only if you consider it a valid bug with a simple fix.\n"
        "- Ignore all other bugs in that report.\n"
        "- Read the markdown report section for that bug, then inspect the corresponding source code to understand the problem and validate if the bug report is correct.\n"
        f"- Source code file to inspect: {source_rel_path}\n"
        "- The original bug reports may contain false positives (invalid bugs). If you deem it as such after inspection of the code, skip the fix.\n"
        "- The bug reports may contain incorrect or imprecise line numbers - do not rely on them, search the file yourself.\n"
        "\n"
        "Rules for fixing:\n"
        "- A fix is considered simple if it can be done within around 10 lines of code diff (both added and removed lines combined).\n"
        "- If the fix is simple and straightforward like this: apply the fix to the source file, then commit the change to git (do NOT push to origin).\n"
        "- The commit should include the bug identifier number and source code file name in the commit log (at the end).\n"
        "- If the fix is NOT simple (larger diff, not immediately straighforward), or you believe the bug report is invalid/false positive: do NOT make any changes.\n"
        "- The goal is that a developer on the team should be able to review the diff and understand it quite quickly, pretty much immediately. Anything more complex than that is considered a not simple fix and will be deferred for later.\n"
        "\n"
        "What the format of your final response (after doing the work) should look like (IMPORTANT):\n"
        "- You must output exactly ONE word only: DONE or NOT_SIMPLE_OR_INVALID.\n"
        "- Respond with DONE if you successfully applied the fix and committed it.\n"
        "- Respond with NOT_SIMPLE_OR_INVALID if the fix is too complex, not straightforward, or the bug is not real.\n"
        "- No additional commentary or justification in your final output, only one word - DONE or NOT_SIMPLE_OR_INVALID.\n"
    )


def run_agent(prompt: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        [AGENT_CMD, "-f", "-p", prompt, "--model", MODEL],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=AGENT_TIMEOUT_S,
        check=False,
    )
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def main() -> int:
    repo_root = Path.cwd()
    reports_root = repo_root / BUGFINDER_REPORTS_DIRNAME
    fixes_file = repo_root / FIXES_FILENAME

    if not reports_root.exists() or not reports_root.is_dir():
        print(f"Error: {BUGFINDER_REPORTS_DIRNAME}/ not found in repo root: {reports_root}", file=sys.stderr)
        return 1

    existing = load_existing_fixes(fixes_file)

    md_files = find_md_files(reports_root)
    if not md_files:
        print(f"No .md files found under {reports_root}")
        return 0

    print(f"Found {len(md_files)} markdown report(s) under {BUGFINDER_REPORTS_DIRNAME}/")
    print(f"Skipping {len(existing)} already-processed (path, bug) pair(s) from {FIXES_FILENAME}")

    appended = 0
    warnings = 0

    for md_path in md_files:
        try:
            md_text = md_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            warnings += 1
            print(f"Warning: Could not read {md_path}: {e}", file=sys.stderr)
            continue

        bug_ids = parse_bug_ids(md_text)
        if not bug_ids:
            continue

        md_rel_path = md_path.relative_to(reports_root).as_posix()
        for bug_id in bug_ids:
            key = FixKey(md_rel_path=md_rel_path, bug_id=bug_id)
            if key in existing:
                continue

            prompt = build_agent_prompt(md_rel_path=md_rel_path, bug_id=bug_id)
            print(f"{md_rel_path} {bug_id}: running agent (timeout {AGENT_TIMEOUT_S}s)...")

            try:
                _rc, stdout, stderr = run_agent(prompt)
            except FileNotFoundError:
                print(
                    f"Error: '{AGENT_CMD}' command not found on PATH. Install/enable it and re-run.",
                    file=sys.stderr,
                )
                return 1
            except subprocess.TimeoutExpired:
                warnings += 1
                print(f"Warning: agent timed out for {md_rel_path} {bug_id}", file=sys.stderr)
                continue
            except Exception as e:
                warnings += 1
                print(f"Warning: agent failed for {md_rel_path} {bug_id}: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            verdict = (stdout or "").strip()
            if verdict not in {"DONE", "NOT_SIMPLE_OR_INVALID"}:
                warnings += 1
                err_preview = (stderr or "").strip()
                if err_preview:
                    err_preview = err_preview[:500]
                    print(
                        f"Warning: Unexpected agent output for {md_rel_path} {bug_id}: {verdict!r}. "
                        f"stderr: {err_preview!r}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Warning: Unexpected agent output for {md_rel_path} {bug_id}: {verdict!r}",
                        file=sys.stderr,
                    )
                continue

            result_line = f"{md_rel_path} - {bug_id} - {verdict}"
            print(result_line)

            try:
                with fixes_file.open("a", encoding="utf-8") as f:
                    f.write(result_line + "\n")
            except Exception as e:
                warnings += 1
                print(f"Warning: Could not append to {fixes_file}: {e}", file=sys.stderr)
                continue

            existing.add(key)
            appended += 1

    print()
    print("Done.")
    print(f"Appended {appended} fix result(s) to {FIXES_FILENAME}")
    if warnings:
        print(f"Warnings: {warnings}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
