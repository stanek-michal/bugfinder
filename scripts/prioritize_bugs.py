#!/usr/bin/env python3
"""
Standalone program to prioritize bug reports using Gemini API.

Walks a given directory recursively, collects all .md files (bug reports),
joins them with newlines, and calls Gemini to identify the top 5 most
critical bugs that need immediate attention.

Usage:
    python prioritize_bugs.py <directory>

Requires:
    - GEMINI_API_KEY environment variable to be set
    - google-genai package installed
"""

import os
import sys
from pathlib import Path
from google import genai as google_genai
from google.genai import types as google_types


# Model configuration
MODEL_NAME = "gemini-3-flash-preview"
THINKING_LEVEL = "low"
TEMPERATURE = 0.7
TIMEOUT_S = 300

# System prompt for bug prioritization
SYSTEM_PROMPT = """You are an expert software engineer and bug triage specialist. Your task is to analyze a collection of bug reports and identify the most critical issues that require immediate developer attention.

From the bug reports provided, identify and highlight the **Top 5 Most Critical Bugs** based on your own independent assessment. For each bug you select, consider:

1. **Severity**: How serious and how obvious is the impact?
2. **Confidence**: How likely is this to be a real, valid bug vs. a false positive? Note that the bug reports are AI-generated and are likely to be false positives due to frequently incorrectassumptions about the rest of the codebase or environment.
3. **ROI**: Is this a quick, high-impact fix or a complex undertaking?
4. **Clarity**: Is the bug clearly defined with an obvious fix path?

**Important**: Ignore any confidence scores or severity ratings in the original bug reports. Perform your own fresh assessment based on the actual code issues described.

For each of your top 5 picks, provide:
- The bug location (file and bug ID number) 
- A brief description of the issue
- Why you ranked it as critical (your reasoning)

Focus on bugs that are:
- No-brainer fixes with high certainty
- High severity with clear impact
- Low effort to fix but high value
- Security or data integrity related

Be decisive and opinionated. The developer reading this has limited time and needs to know exactly where to focus first."""


def find_md_files(directory: Path) -> list[Path]:
    """Recursively find all .md files in the given directory."""
    md_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".md"):
                md_files.append(Path(root) / filename)
    return sorted(md_files)


def read_and_join_files(files: list[Path]) -> str:
    """Read all files and join their contents with newlines."""
    contents = []
    for filepath in files:
        try:
            text = filepath.read_text(encoding="utf-8")
            # Add a header to identify which file the content came from
            header = f"\n{'='*60}\n# Source: {filepath.name}\n{'='*60}\n"
            contents.append(header + text)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
    return "\n".join(contents)


def call_gemini(client: google_genai.Client, user_prompt: str) -> dict:
    """Call Gemini API with the bug reports and return the response."""
    # Build thinking config for Gemini 3
    thinking_config = google_types.ThinkingConfig(thinking_level=THINKING_LEVEL)

    # Build generation config
    gen_config = google_types.GenerateContentConfig(
        temperature=TEMPERATURE,
        thinking_config=thinking_config,
        system_instruction=SYSTEM_PROMPT,
    )

    # Make the API call
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=gen_config,
    )

    # Extract response data
    text = getattr(resp, "text", "") or ""
    usage = getattr(resp, "usage_metadata", None)

    result = {"text": text}
    if usage is not None:
        result["prompt_tokens"] = int(getattr(usage, "prompt_token_count", 0) or 0)
        result["output_tokens"] = int(getattr(usage, "candidates_token_count", 0) or 0)
        result["thinking_tokens"] = int(getattr(usage, "thoughts_token_count", 0) or 0)

    return result


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        print(f"  directory: Path to directory containing bug report .md files", file=sys.stderr)
        return 1

    directory = Path(sys.argv[1])

    if not directory.exists():
        print(f"Error: Directory does not exist: {directory}", file=sys.stderr)
        return 1

    if not directory.is_dir():
        print(f"Error: Path is not a directory: {directory}", file=sys.stderr)
        return 1

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Bug Report Prioritizer")
    print("=" * 60)
    print(f"Directory: {directory.resolve()}")
    print(f"Model: {MODEL_NAME}")
    print(f"Thinking level: {THINKING_LEVEL}")
    print()

    # Find all .md files
    print("Searching for bug report .md files...")
    md_files = find_md_files(directory)

    if not md_files:
        print("No .md files found in directory.")
        return 0

    print(f"Found {len(md_files)} .md file(s):")
    for f in md_files:
        print(f"  - {f.name}")
    print()

    # Read and join all files
    print("Reading and joining bug reports...")
    blob = read_and_join_files(md_files)
    print(f"Total text size: {len(blob):,} characters")
    print()

    # Initialize Gemini client
    print("Initializing Gemini client...")
    client = google_genai.Client(api_key=api_key)

    # Call Gemini
    print(f"Calling {MODEL_NAME} to analyze bug reports...")
    print("(This may take a moment...)")
    print()

    try:
        result = call_gemini(client, blob)

        print("=" * 60)
        print("TOKEN USAGE")
        print("=" * 60)
        if "prompt_tokens" in result:
            print(f"Input tokens: {result['prompt_tokens']:,}")
            print(f"Output tokens: {result['output_tokens']:,}")
            print(f"Thinking tokens: {result['thinking_tokens']:,}")
        print()

        print("=" * 60)
        print("TOP 5 CRITICAL BUGS")
        print("=" * 60)
        print()
        print(result["text"])

        return 0

    except Exception as e:
        print(f"Error calling Gemini API: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

