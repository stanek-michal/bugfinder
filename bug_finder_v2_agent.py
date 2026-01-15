#!/usr/bin/env python3
"""
Run an agent-based bugfinder over a repo folder.

What it does:
- Recursively scans a chosen folder within the repo root
- Filters files using include/exclude globs (defaults baked in)
- Runs `agent` per file with a prompt to find bugs scoped to that file
- Writes one markdown report per file under BUGFINDER_REPORTS/
- Logs progress line-by-line

Notes:
- No retries if the agent output is malformed
- No parallelism (intended for overnight runs)
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BUGFINDER_REPORTS_DIRNAME = "BUGFINDER_REPORTS"
AGENT_CMD = "claude"
DEFAULT_MODEL = "gpt-5.2-high"
DEFAULT_TIMEOUT_S = 10 * 60

DEFAULT_INCLUDE_GLOBS = [
    "**/*.c",
    "**/*.cc",
    "**/*.cpp",
    "**/*.cxx",
]
DEFAULT_EXCLUDE_GLOBS = [
    ".git/**",
    "build/**",
    "Build/**",
    "Dependencies/**",
    "Python/**",
    "dist/**",
    "out/**",
    "target/**",
    "vendor/**",
    "third_party/**",
]
DEFAULT_FOLLOW_SYMLINKS = True

SUPPORTED_EXTS = {
    ".c",
#    ".h",
#    ".hh",
#    ".hpp",
#    ".hxx",
    ".cc",
    ".cpp",
    ".cxx",
}


@dataclass(frozen=True)
class ScanConfig:
    include_globs: list[str]
    exclude_globs: list[str]
    follow_symlinks: bool


def resolve_scan_config(args: argparse.Namespace) -> ScanConfig:
    include_globs = args.include_glob or DEFAULT_INCLUDE_GLOBS
    exclude_globs = args.exclude_glob or DEFAULT_EXCLUDE_GLOBS
    follow_symlinks = DEFAULT_FOLLOW_SYMLINKS if args.follow_symlinks is None else args.follow_symlinks

    return ScanConfig(
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        follow_symlinks=follow_symlinks,
    )


def is_excluded(rel_posix_path: str, exclude_globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel_posix_path, pat) for pat in exclude_globs)


def is_included(rel_posix_path: str, include_globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel_posix_path, pat) for pat in include_globs)


def discover_files(scan_root: Path, repo_root: Path, config: ScanConfig) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=config.follow_symlinks):
        dir_path = Path(dirpath)
        rel_dir = dir_path.relative_to(repo_root).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        # Prune excluded directories to reduce traversal.
        pruned_dirnames: list[str] = []
        for d in dirnames:
            rel = f"{rel_dir}/{d}" if rel_dir else d
            if not is_excluded(rel, config.exclude_globs):
                pruned_dirnames.append(d)
        dirnames[:] = pruned_dirnames

        for name in filenames:
            full = dir_path / name
            if not config.follow_symlinks and full.is_symlink():
                continue

            rel = full.relative_to(repo_root).as_posix()
            if is_excluded(rel, config.exclude_globs):
                continue
            if not is_included(rel, config.include_globs):
                continue
            if full.suffix.lower() not in SUPPORTED_EXTS:
                continue
            files.append(full)
    return files


def build_agent_prompt(target_rel_path: str) -> str:
    return (
        "You are an expert code reviewer for systems code. Your goal is to find real, impactful defects and bugs.\n"
        "\n"
        "Scope and context:\n"
        "- Your current working directory is the repository root.\n"
        f"- Target source file: {target_rel_path}\n"
        "- You may read other files in the repo to confirm suspected issues in the target file.\n"
        "- Report ONLY bugs that pertain to the target file (do not report bugs that belong only to other files).\n"
        "\n"
        "Safety constraints (STRICT):\n"
        "- Do NOT modify, create, or delete any files in the repository.\n"
        "- Do NOT run any git commands that modify state (no commit, checkout, stash, reset, branch, merge, rebase).\n"
        "- Do NOT use `cd` to change directories.\n"
        "- You are READ-ONLY. You may only read files and run non-destructive commands.\n"
        "\n"
        "Evidence and precision:\n"
        "- Do NOT include line numbers (you do not know them).\n"
        "- Use code snippets as evidence and always name the function(s) involved.\n"
        "- If you cite a snippet from a different file, include its repo-relative path in the evidence.\n"
        "- It is fine to include longer snippets if required to understand the bug, do not overly shorten\n"
        "\n"
        "Return a Markdown report with this exact structure and section titles:\n"
        f"# File Review — {target_rel_path}\n"
        "\n"
        "## Summary\n"
        "Provide a 3–7 sentence summary of the most significant risks and areas to review.\n"
        "\n"
        "## Findings\n"
        "Organize by severity: Critical, High, Medium, Low. For each finding, include:\n"
        "- ID: F{index}\n"
        "- Title: Summary of the finding\n"
        "- Severity: Critical|High|Medium|Low\n"
        "- Confidence: 0.0–1.0\n"
        "- Evidence: All relevant code snippets, function names, and file paths when applicable\n"
        "- Why it’s a bug: Explain the defect clearly and fully. Prioritize clarity and cleanliness of the report, do not overly shorten it. Describe premise of the code and what it tries to achieve, and what is wrong. Do not assume the reader is an expert in that part of the code.\n"
        "- Fix: Practical remediation steps or patterns\n"
        "- Tests: Assertions/unit/integration tests to catch or reproduce\n"
        "\n"
        "## False Positives or Uncertain\n"
        "List any items you flagged but consider likely false positives, with rationale.\n"
        "\n"
        "## Notes\n"
        "Any caveats about missing context or environment dependencies.\n"
        "\n"
        "-- DO NOT MODIFY THE SECTION TITLES ABOVE. USE CLEAR MARKDOWN STRUCTURE. NO OTHER COMMENTS. --\n"
    )


def run_agent(prompt: str, model: str, timeout_s: int) -> tuple[int, str, str]:
    completed = subprocess.run(
        [AGENT_CMD, "-p", prompt, "--model", model, "--dangerously-skip-permissions"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def write_report(output_root: Path, target_rel_path: str, content: str) -> Path:
    out_path = output_root / f"{target_rel_path}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-based bugfinder for a repo folder.")
    parser.add_argument(
        "folder",
        help="Folder to scan (relative to repo root or absolute path inside repo).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name for agent (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"Agent timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--output-dir",
        default=BUGFINDER_REPORTS_DIRNAME,
        help=f"Output directory for reports (default: {BUGFINDER_REPORTS_DIRNAME}).",
    )
    parser.add_argument(
        "--include-glob",
        action="append",
        default=None,
        help="Include glob pattern (can be repeated). Overrides defaults.",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=None,
        help="Exclude glob pattern (can be repeated). Overrides defaults.",
    )
    parser.add_argument(
        "--file-list",
        default=None,
        help="Path to a text file with one repo-relative file path per line. Skips discovery.",
    )
    parser.add_argument(
        "--follow-symlinks",
        dest="follow_symlinks",
        action="store_true",
        default=None,
        help="Follow symlinks while scanning.",
    )
    parser.add_argument(
        "--no-follow-symlinks",
        dest="follow_symlinks",
        action="store_false",
        default=None,
        help="Do not follow symlinks while scanning.",
    )
    return parser.parse_args()


def main() -> int:
    repo_root = Path.cwd().resolve()
    args = parse_args()

    raw_scan_root = Path(args.folder)
    scan_root = (raw_scan_root if raw_scan_root.is_absolute() else repo_root / raw_scan_root).resolve()
    if not scan_root.exists() or not scan_root.is_dir():
        print(f"Error: folder not found or not a directory: {scan_root}", file=sys.stderr)
        return 1
    if scan_root != repo_root and repo_root not in scan_root.parents:
        print(f"Error: folder must be within repo root: {repo_root}", file=sys.stderr)
        return 1

    config = resolve_scan_config(args)
    output_root = repo_root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    if args.file_list:
        file_list_path = Path(args.file_list)
        lines = file_list_path.read_text(encoding="utf-8").strip().splitlines()
        files = [repo_root / line.strip() for line in lines if line.strip()]
    else:
        files = discover_files(scan_root=scan_root, repo_root=repo_root, config=config)
    if not files:
        print("No matching source files found.")
        return 0

    scan_rel = scan_root.relative_to(repo_root).as_posix()
    print(f"Scanning {len(files)} file(s) under {scan_rel or '.'}")

    warnings = 0
    for idx, full_path in enumerate(files, start=1):
        rel_path = full_path.relative_to(repo_root).as_posix()
        print(f"[{idx}/{len(files)}] {rel_path}...")

        prompt = build_agent_prompt(rel_path)
        try:
            _rc, stdout, stderr = run_agent(prompt, model=args.model, timeout_s=args.timeout_s)
        except FileNotFoundError:
            print(f"Error: '{AGENT_CMD}' command not found on PATH. Install/enable it and re-run.", file=sys.stderr)
            return 1
        except subprocess.TimeoutExpired:
            warnings += 1
            print(f"Warning: agent timed out for {rel_path}", file=sys.stderr)
            continue
        except Exception as e:
            warnings += 1
            print(f"Warning: agent failed for {rel_path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        if not stdout.strip():
            warnings += 1
            err_preview = (stderr or "").strip()
            if err_preview:
                err_preview = err_preview[:500]
                print(f"Warning: empty agent output for {rel_path}. stderr: {err_preview!r}", file=sys.stderr)
            else:
                print(f"Warning: empty agent output for {rel_path}.", file=sys.stderr)

        write_report(output_root=output_root, target_rel_path=rel_path, content=stdout)

    print()
    print("Done.")
    if warnings:
        print(f"Warnings: {warnings}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
