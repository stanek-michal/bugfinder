#!/usr/bin/env python3
"""
Standalone validator for BUGFINDER markdown reports.

What it does:
- Recursively traverses ./BUGFINDER_REPORTS looking for .md files
- Each .md file contains bug reports for a single source file in the repo root
- Bugs are identified by lines like: #### F10: Some Title
- For each bug ID (F#) found, runs: agent -p "<prompt>"
  - The agent is instructed to validate ONLY that bug and output one word: VALID or INVALID
- Appends results to ./BUGFINDER_VALIDATION.txt in the format:
  <md_rel_path> - F# - VALID|INVALID
- Skips duplicates already present in BUGFINDER_VALIDATION.txt (append-only)

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
VALIDATION_FILENAME = "BUGFINDER_VALIDATION.txt"
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
class ValidationKey:
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


def load_existing_validations(validation_file: Path) -> set[ValidationKey]:
    existing: set[ValidationKey] = set()
    if not validation_file.exists():
        return existing

    try:
        for raw_line in validation_file.read_text(encoding="utf-8").splitlines():
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
            if verdict not in {"VALID", "INVALID"}:
                continue
            existing.add(ValidationKey(md_rel_path=md_rel_path, bug_id=bug_id))
    except Exception as e:
        print(f"Warning: Could not read existing validations from {validation_file}: {e}", file=sys.stderr)
    return existing


def build_agent_prompt(md_rel_path: str, bug_id: str) -> str:
    # Keep prompt free of double quotes to avoid confusion in shell usage examples.
    source_rel_path = md_rel_path[:-3] if md_rel_path.endswith(".md") else md_rel_path

    return (
        "You are an expert software engineer validating an AI-generated bug report.\n"
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
        f"- Validate ONLY bug {bug_id} in the markdown report {md_rel_path}.\n"
        "- Ignore all other bugs in that report.\n"
        "- Read the markdown report section for that bug, then independently inspect the corresponding source code.\n"
        f"- Source code file to inspect: {source_rel_path}\n"
        "- Read surrounding code and any relevant references in the repo to decide if the bug is real.\n"
        "- The original bug reports may contain false positives.\n"
        "- The bug reports may contain incorrect or imprecise line numbers - do not rely on them, search the file yourself.\n"
        "\n"
        "What the format of your response should look like (IMPORTANT):\n"
        "- You must output exactly ONE word only: VALID or INVALID.\n"
        "- Respond with VALID if the bug is confirmed OR if you are unsure if it is a real bug / lack context (err on the side of VALID).\n"
        "- Respond with INVALID if you are certain it is a false positive and does not need fixing.\n"
        "No additional commentary or justification in your final output, only one word - VALID or INVALID.\n"
    )


def run_agent(prompt: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        [AGENT_CMD, "-p", prompt, "--model", MODEL],
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
    validation_file = repo_root / VALIDATION_FILENAME

    if not reports_root.exists() or not reports_root.is_dir():
        print(f"Error: {BUGFINDER_REPORTS_DIRNAME}/ not found in repo root: {reports_root}", file=sys.stderr)
        return 1

    existing = load_existing_validations(validation_file)

    md_files = find_md_files(reports_root)
    if not md_files:
        print(f"No .md files found under {reports_root}")
        return 0

    print(f"Found {len(md_files)} markdown report(s) under {BUGFINDER_REPORTS_DIRNAME}/")
    print(f"Skipping {len(existing)} already-validated (path, bug) pair(s) from {VALIDATION_FILENAME}")

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
            key = ValidationKey(md_rel_path=md_rel_path, bug_id=bug_id)
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
            if verdict not in {"VALID", "INVALID"}:
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

            line = f"{md_rel_path} - {bug_id} - {verdict}\n"
            try:
                with validation_file.open("a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                warnings += 1
                print(f"Warning: Could not append to {validation_file}: {e}", file=sys.stderr)
                continue

            existing.add(key)
            appended += 1

    print()
    print("Done.")
    print(f"Appended {appended} validation line(s) to {VALIDATION_FILENAME}")
    if warnings:
        print(f"Warnings: {warnings}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


