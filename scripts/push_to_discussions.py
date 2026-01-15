#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


RATE_LIMIT_SLEEP = 0.0
_LAST_CALL_TS = 0.0


def load_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Install with: python3 -m pip install pyyaml"
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")
    return data


def run_gh_graphql(query: str, variables: dict) -> dict:
    global _LAST_CALL_TS
    args = [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={query}",
    ]
    for key, value in (variables or {}).items():
        if value is None:
            continue
        args.extend(["-f", f"{key}={value}"])

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        if RATE_LIMIT_SLEEP > 0:
            elapsed = time.monotonic() - _LAST_CALL_TS
            if elapsed < RATE_LIMIT_SLEEP:
                time.sleep(RATE_LIMIT_SLEEP - elapsed)

        proc = subprocess.run(args, capture_output=True, text=True)
        _LAST_CALL_TS = time.monotonic()
        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse gh output as JSON: {e}") from e

        error_text = (proc.stderr.strip() or proc.stdout.strip()).lower()
        if "submitted too quickly" in error_text or "secondary rate limit" in error_text:
            sleep_s = min(2 ** attempt, 30)
            time.sleep(sleep_s)
            continue

        raise RuntimeError(f"gh api graphql failed: {proc.stderr.strip() or proc.stdout.strip()}")

    raise RuntimeError("gh api graphql failed: exceeded retry limit after rate limiting")


def get_repo_and_category_ids(owner: str, repo: str, category_name: str) -> tuple:
    query = """
    query($owner: String!, $name: String!) {
      repository(owner: $owner, name: $name) {
        id
        discussionCategories(first: 100) {
          nodes { id name }
        }
      }
    }
    """
    data = run_gh_graphql(query, {"owner": owner, "name": repo})
    repo_node = data.get("data", {}).get("repository")
    if not repo_node:
        raise RuntimeError("Repository not found or access denied.")
    repo_id = repo_node["id"]
    categories = repo_node.get("discussionCategories", {}).get("nodes", [])
    cat_id = None
    for cat in categories:
        if cat.get("name") == category_name:
            cat_id = cat.get("id")
            break
    if not cat_id:
        available = ", ".join([c.get("name", "") for c in categories])
        raise RuntimeError(
            f"Category '{category_name}' not found. Available: {available}"
        )
    return repo_id, cat_id


def find_discussion_by_title(owner: str, repo: str, title: str) -> tuple:
    """
    Search for a discussion by exact title.
    Returns (discussion_id, url) if found, (None, None) otherwise.
    """
    query = """
    query($queryString: String!) {
      search(query: $queryString, type: DISCUSSION, first: 10) {
        nodes {
          ... on Discussion {
            id
            title
            url
          }
        }
      }
    }
    """
    safe_title = title.replace('"', '\\"')
    q = f'repo:{owner}/{repo} in:title "{safe_title}"'
    data = run_gh_graphql(query, {"queryString": q})
    nodes = data.get("data", {}).get("search", {}).get("nodes", []) or []
    for node in nodes:
        if node.get("title") == title:
            return node.get("id"), node.get("url")
    return None, None


def create_discussion(repo_id: str, category_id: str, title: str, body: str) -> tuple:
    query = """
    mutation($repoId: ID!, $catId: ID!, $title: String!, $body: String!) {
      createDiscussion(input: {
        repositoryId: $repoId,
        categoryId: $catId,
        title: $title,
        body: $body
      }) {
        discussion { id url }
      }
    }
    """
    data = run_gh_graphql(
        query, {"repoId": repo_id, "catId": category_id, "title": title, "body": body}
    )
    discussion = (
        data.get("data", {})
        .get("createDiscussion", {})
        .get("discussion", {})
    )
    if not discussion:
        raise RuntimeError("Failed to create discussion (no discussion returned).")
    return discussion["id"], discussion.get("url", "")


def update_discussion(discussion_id: str, body: str) -> str:
    """
    Update only the body of an existing discussion.
    Returns the discussion URL.
    """
    query = """
    mutation($discussionId: ID!, $body: String!) {
      updateDiscussion(input: {
        discussionId: $discussionId,
        body: $body
      }) {
        discussion { id url }
      }
    }
    """
    data = run_gh_graphql(query, {"discussionId": discussion_id, "body": body})
    discussion = (
        data.get("data", {})
        .get("updateDiscussion", {})
        .get("discussion", {})
    )
    if not discussion:
        raise RuntimeError("Failed to update discussion (no discussion returned).")
    return discussion.get("url", "")


def posix_path(path: Path) -> str:
    return path.as_posix()


def strip_prefix(value: str, prefix: str) -> str:
    if prefix and value.startswith(prefix):
        return value[len(prefix):]
    return value


def gather_markdown_files(run_dir: Path) -> list:
    return sorted([p for p in run_dir.rglob("*.md") if p.is_file()])


def build_title(file_path: Path, title_strip_prefix: str) -> str:
    base = posix_path(file_path)
    base = strip_prefix(base, title_strip_prefix)
    base = base.lstrip("/")
    if base.endswith(".md"):
        base = base[:-3]
    return base


def build_source_rel(file_path: Path, local_prefix: str, repo_path_prefix: str) -> str:
    base = posix_path(file_path)
    base = strip_prefix(base, local_prefix)
    base = base.lstrip("/")
    if base.endswith(".md"):
        base = base[:-3]
    if repo_path_prefix:
        base = f"{repo_path_prefix.rstrip('/')}/{base}"
    return base


def build_source_url(owner: str, repo: str, branch: str, source_rel: str) -> str:
    return f"https://github.com/{owner}/{repo}/blob/{branch}/{source_rel}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Push Bugfinder results to GitHub Discussions via gh CLI."
    )
    parser.add_argument("run_name", help="Bugfinder run folder name")
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to YAML config (default: config.yml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without creating/updating discussions",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Upsert mode: update existing discussions or create if not found",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    github_cfg = config.get("github", {})
    bugfinder_cfg = config.get("bugfinder", {})
    output_cfg = config.get("output", {})

    owner = github_cfg.get("owner")
    repo = github_cfg.get("repo")
    branch = github_cfg.get("branch", "main")
    category_name = github_cfg.get("category_name")
    local_prefix = github_cfg.get("local_prefix", "")
    repo_path_prefix = github_cfg.get("repo_path_prefix", "")

    runs_dir = bugfinder_cfg.get("runs_dir")
    aggregated_dir_name = bugfinder_cfg.get("aggregated_dir_name", "aggregated")

    title_strip_prefix = output_cfg.get("title_strip_prefix", "")
    skip_if_exists = bool(output_cfg.get("skip_if_exists", True))
    global RATE_LIMIT_SLEEP
    RATE_LIMIT_SLEEP = float(output_cfg.get("rate_limit_sleep_seconds", 0.0) or 0.0)

    missing = [k for k, v in {
        "github.owner": owner,
        "github.repo": repo,
        "github.category_name": category_name,
        "bugfinder.runs_dir": runs_dir,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required config fields: {', '.join(missing)}")

    if "{run_name}" in local_prefix:
        local_prefix = local_prefix.replace("{run_name}", args.run_name)
    if "{run_name}" in title_strip_prefix:
        title_strip_prefix = title_strip_prefix.replace("{run_name}", args.run_name)

    run_dir = Path(runs_dir) / args.run_name / aggregated_dir_name
    if not run_dir.exists():
        raise RuntimeError(f"Run aggregated dir not found: {run_dir}")

    md_files = gather_markdown_files(run_dir)
    if not md_files:
        print(f"No markdown files found under {run_dir}")
        return 0

    created = 0
    updated = 0
    skipped = 0

    if not args.dry_run:
        repo_id, category_id = get_repo_and_category_ids(owner, repo, category_name)

    for md_path in md_files:
        title = build_title(md_path, title_strip_prefix)
        source_rel = build_source_rel(md_path, local_prefix, repo_path_prefix)
        source_url = build_source_url(owner, repo, branch, source_rel)
        content = md_path.read_text(encoding="utf-8")
        body = f"Source: {source_url}\n\n{content}"

        if args.dry_run:
            if args.update:
                print(f"DRY-RUN: would upsert discussion: {title}")
            else:
                print(f"DRY-RUN: would create discussion: {title}")
            print(f"DRY-RUN: source link: {source_url}")
            created += 1
            continue

        if args.update:
            # Upsert mode: update if exists, create if not
            discussion_id, discussion_url = find_discussion_by_title(owner, repo, title)
            if discussion_id:
                discussion_url = update_discussion(discussion_id, body)
                updated += 1
                print(f"UPDATED: {title} -> {discussion_url}")
            else:
                discussion_id, discussion_url = create_discussion(
                    repo_id, category_id, title, body
                )
                created += 1
                print(f"CREATED: {title} -> {discussion_url}")
        else:
            # Default mode: create, optionally skip if exists
            if skip_if_exists:
                discussion_id, _ = find_discussion_by_title(owner, repo, title)
                if discussion_id:
                    print(f"SKIP (exists): {title}")
                    skipped += 1
                    continue

            discussion_id, discussion_url = create_discussion(
                repo_id, category_id, title, body
            )
            created += 1
            print(f"CREATED: {title} -> {discussion_url}")

    if args.dry_run:
        print(f"Done (dry-run). Would process: {created}")
    else:
        print(f"Done. Created: {created}, Updated: {updated}, Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
