#!/usr/bin/env python3
"""
Standalone test program to count tokens using the Gemini API.

Walks a given directory recursively, collects all .md files,
joins them with newlines, and calls count_tokens via the Gemini API.

Usage:
    python count_tokens_test.py <directory>

Requires:
    - GEMINI_API_KEY environment variable to be set
    - google-genai package installed
"""

import os
import sys
from pathlib import Path
from google import genai as google_genai


# Default model for token counting
DEFAULT_MODEL = "gemini-2.0-flash"


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
            contents.append(text)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
    return "\n".join(contents)


def count_tokens(client: google_genai.Client, model: str, text: str) -> dict:
    """Count tokens using the Gemini API."""
    resp = client.models.count_tokens(model=model, contents=text)
    return {
        "total_tokens": getattr(resp, "total_tokens", None),
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory> [model]", file=sys.stderr)
        print(f"  directory: Path to directory containing .md files", file=sys.stderr)
        print(f"  model: Optional Gemini model name (default: {DEFAULT_MODEL})", file=sys.stderr)
        return 1

    directory = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

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
    print("Gemini Token Counter")
    print("=" * 60)
    print(f"Directory: {directory.resolve()}")
    print(f"Model: {model}")
    print()

    # Find all .md files
    print("Searching for .md files...")
    md_files = find_md_files(directory)

    if not md_files:
        print("No .md files found in directory.")
        return 0

    print(f"Found {len(md_files)} .md file(s):")
    for f in md_files:
        print(f"  - {f}")
    print()

    # Read and join all files
    print("Reading and joining files...")
    blob = read_and_join_files(md_files)
    print(f"Total text size: {len(blob):,} characters")
    print()

    # Initialize Gemini client
    print("Initializing Gemini client...")
    client = google_genai.Client(api_key=api_key)

    # Count tokens
    print("Counting tokens via Gemini API...")
    try:
        result = count_tokens(client, model, blob)
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Files processed: {len(md_files)}")
        print(f"Total characters: {len(blob):,}")
        print(f"Total tokens: {result['total_tokens']:,}")
        if result['total_tokens'] and len(blob) > 0:
            chars_per_token = len(blob) / result['total_tokens']
            print(f"Characters per token: {chars_per_token:.2f}")
        return 0

    except Exception as e:
        print(f"Error counting tokens: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

