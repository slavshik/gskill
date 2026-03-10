"""Initial skill generation skill file I/O."""

import base64
import json
import os
import re
import urllib.request
from pathlib import Path

import openai


def _make_skill_name(repo: str) -> str:
    """Sanitize a repo short name into a valid skill name.

    Rules: lowercase, only [a-z0-9-], collapse/strip hyphens, max 64 chars.
    """
    name = repo.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = name.strip("-")
    return name[:64]


def _fetch_readme(owner: str, repo: str, max_chars: int = 3000) -> str:
    """Fetch the README from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gskill/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            content = base64.b64decode(data["content"]).decode(
                "utf-8", errors="replace"
            )
            return content[:max_chars]
    except Exception:
        return ""


def _fetch_file(owner: str, repo: str, path: str, max_chars: int = 2000) -> str:
    """Fetch a specific file from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gskill/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            if data.get("encoding") == "base64":
                content = base64.b64decode(data["content"]).decode(
                    "utf-8", errors="replace"
                )
                return content[:max_chars]
    except Exception:
        pass
    return ""


def _read_local_file(path: Path, max_chars: int = 3000) -> str:
    """Read a local file, returning up to max_chars."""
    try:
        return path.read_text(errors="replace")[:max_chars]
    except Exception:
        return ""


def _gather_local_context(repo_path: Path) -> tuple[str, str, str]:
    """Gather README, config files, and directory structure from a local repo.

    Returns:
        Tuple of (readme, extra_context, repo_name).
    """
    repo_name = repo_path.resolve().name

    # Try README variants
    readme = ""
    for name in ["README.md", "readme.md", "README.rst", "README.txt", "README"]:
        candidate = repo_path / name
        if candidate.exists():
            readme = _read_local_file(candidate, max_chars=4000)
            break

    # Try CLAUDE.md for extra agent-specific context
    claude_md = repo_path / "CLAUDE.md"
    claude_context = ""
    if claude_md.exists():
        claude_context = _read_local_file(claude_md, max_chars=4000)

    # Gather config files (broader set for JS/TS/Python/etc.)
    extra_context = ""
    config_candidates = [
        "package.json",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "Makefile",
        "pytest.ini",
        "tsconfig.json",
        "tsconfig.base.json",
        ".eslintrc.js",
        "eslint.config.cjs",
        "jest.config.js",
        "jest.config.cjs",
        "jest.config.ts",
        "vitest.config.ts",
        "CONTRIBUTING.md",
    ]
    files_added = 0
    for name in config_candidates:
        fpath = repo_path / name
        if fpath.exists():
            content = _read_local_file(fpath, max_chars=2000)
            if content:
                extra_context += f"\n\n### {name}\n```\n{content}\n```"
                files_added += 1
                if files_added >= 4:
                    break

    if claude_context:
        extra_context += f"\n\n### CLAUDE.md (existing agent guidance)\n```\n{claude_context}\n```"

    return readme, extra_context, repo_name


def generate_initial_skill(
    repo_url: str,
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    """Generate an initial SKILL.md for the repo via static analysis.

    Fetches the README and common config files, then asks a model to synthesize
    repo-specific guidance for a coding agent.

    Args:
        repo_url: Full GitHub URL, e.g. 'https://github.com/pallets/jinja'.
        model: Model to use. Defaults to GSKILL_SKILL_MODEL env var, then 'gpt-5.2'.
        base_url: OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL env var.

    Returns:
        Skill content as a string (YAML frontmatter + markdown body).
    """
    # Parse owner/repo from URL
    parts = repo_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    skill_name = _make_skill_name(repo)

    readme = _fetch_readme(owner, repo)

    # Try to grab common config files for test/build info
    extra_context = ""
    for candidate in [
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "Makefile",
        "pytest.ini",
    ]:
        content = _fetch_file(owner, repo, candidate, max_chars=1500)
        if content:
            extra_context += f"\n\n### {candidate}\n```\n{content}\n```"
            break  # one is enough

    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    resolved_model = model or os.environ.get("GSKILL_SKILL_MODEL")

    if resolved_base_url and not resolved_model:
        raise ValueError(
            "A custom base URL is set but no skill model was specified. "
            "Use --skill-model or set GSKILL_SKILL_MODEL to the model name your local backend serves."
        )

    resolved_model = resolved_model or "gpt-5.2"

    # Strip LiteLLM-style provider prefix (e.g. "openai/gpt-5.2" -> "gpt-5.2").
    # The OpenAI client talks to the endpoint directly, so the prefix is meaningless
    # and confuses non-OpenAI backends like Ollama.
    if "/" in resolved_model:
        resolved_model = resolved_model.split("/", 1)[1]

    client_kwargs: dict = {}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    if not os.environ.get("OPENAI_API_KEY") and resolved_base_url:
        client_kwargs["api_key"] = "none"

    client = openai.OpenAI(**client_kwargs)
    try:
        message = client.chat.completions.create(
            model=resolved_model,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are generating a SKILL.md for the '{repo}' repository.
This skill file will be injected into the system prompt of a coding agent that must
solve GitHub issues by modifying source files in a Docker container at /testbed.

Repository URL: {repo_url}

README (may be truncated):
{readme}
{extra_context}

Output a complete SKILL.md starting with YAML frontmatter, then the body. Use exactly this structure:

---
name: {skill_name}
description: <one-sentence description, max 1024 characters, no angle-bracket XML tags, stating what the skill covers and when to use it>
---

<body: 400-800 words covering the five sections below>

The body must cover:

1. **Test commands**: The exact command(s) to run the test suite (e.g., `pytest`, `tox`, `make test`).
   If there are relevant flags or test file patterns, include them.
2. **Code structure**: Key directories and files an agent should know about.
3. **Conventions**: Code style, naming patterns, or idioms specific to this project.
4. **Common pitfalls**: Mistakes an agent typically makes on this repo and how to avoid them.
5. **Workflow**: Recommended steps to diagnose and fix an issue (reproduce, patch, verify).

Constraints:
- The `name` field must be exactly: {skill_name}
- The `description` must be non-empty, at most 1024 characters, and must not contain angle-bracket XML tags.
- Be specific and actionable. Write for an AI agent, not a human developer.
- Do NOT include generic advice that applies to all Python projects.
- Focus on what is distinctive about {repo}.""",
                }
            ],
        )
    except openai.APIStatusError as exc:
        endpoint = resolved_base_url or "https://api.openai.com"
        raise RuntimeError(
            f"Skill generation failed — HTTP {exc.status_code} from {endpoint!r} "
            f"with model {resolved_model!r}: {exc.message}"
        ) from exc
    except openai.APIConnectionError as exc:
        endpoint = resolved_base_url or "https://api.openai.com"
        raise RuntimeError(
            f"Skill generation failed — could not connect to {endpoint!r}: {exc}"
        ) from exc

    content = message.choices[0].message.content
    if not content:
        raise RuntimeError(
            f"Skill generation failed — model {resolved_model!r} returned an empty response "
            "(the model may have invoked a tool instead of generating text, or the response was filtered)"
        )
    return content


def _resolve_model_and_client(
    model: str | None, base_url: str | None
) -> tuple[str, "openai.OpenAI"]:
    """Resolve model name and create an OpenAI client."""
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    resolved_model = model or os.environ.get("GSKILL_SKILL_MODEL")

    if resolved_base_url and not resolved_model:
        raise ValueError(
            "A custom base URL is set but no skill model was specified. "
            "Use --skill-model or set GSKILL_SKILL_MODEL to the model name your local backend serves."
        )

    resolved_model = resolved_model or "gpt-5.2"

    if "/" in resolved_model:
        resolved_model = resolved_model.split("/", 1)[1]

    client_kwargs: dict = {}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    if not os.environ.get("OPENAI_API_KEY") and resolved_base_url:
        client_kwargs["api_key"] = "none"

    return resolved_model, openai.OpenAI(**client_kwargs)


def generate_local_skill(
    repo_path: str,
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    """Generate a SKILL.md from a local repository path.

    Reads README, CLAUDE.md, config files, and directory structure from the
    local filesystem, then asks a model to synthesize repo-specific guidance.

    Args:
        repo_path: Path to the local repository root.
        model: Model to use. Defaults to GSKILL_SKILL_MODEL env var, then 'gpt-5.2'.
        base_url: OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL env var.

    Returns:
        Skill content as a string (YAML frontmatter + markdown body).
    """
    path = Path(repo_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    readme, extra_context, repo_name = _gather_local_context(path)
    skill_name = _make_skill_name(repo_name)

    resolved_model, client = _resolve_model_and_client(model, base_url)

    # Detect language from config files
    is_js = (path / "package.json").exists()
    is_python = (path / "pyproject.toml").exists() or (path / "setup.py").exists()
    lang_hint = "JavaScript/TypeScript" if is_js else "Python" if is_python else "this"

    try:
        message = client.chat.completions.create(
            model=resolved_model,
            max_tokens=3000,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are generating a SKILL.md for the '{repo_name}' repository.
This skill file will be used by a Claude coding agent (claude.ai/code) working
directly in this repository on the local filesystem.

Repository name: {repo_name}

README (may be truncated):
{readme}
{extra_context}

Output a complete SKILL.md starting with YAML frontmatter, then the body. Use exactly this structure:

---
name: {skill_name}
description: <one-sentence description, max 1024 characters, no angle-bracket XML tags, stating what the skill covers and when to use it>
---

<body: 600-1200 words covering the sections below>

The body must cover:

1. **Test commands**: The exact command(s) to run the test suite. Include both root-level and per-package
   commands if this is a monorepo. Mention relevant flags, config files, or test file patterns.
2. **Build & dev commands**: How to build, start dev server, lint, format. Include common flags.
3. **Code structure**: Key directories, packages, and files an agent should know about.
   For monorepos, describe the package layout and how packages relate.
4. **Conventions**: Code style, naming patterns, file organization idioms specific to this project.
5. **Common pitfalls**: Mistakes an agent typically makes on this repo and how to avoid them.
6. **Workflow**: Recommended steps to diagnose and fix an issue (reproduce, patch, verify).

Constraints:
- The `name` field must be exactly: {skill_name}
- The `description` must be non-empty, at most 1024 characters, and must not contain angle-bracket XML tags.
- Be specific and actionable. Write for an AI coding agent, not a human developer.
- Do NOT include generic advice that applies to all {lang_hint} projects.
- Focus on what is distinctive about {repo_name}.
- If CLAUDE.md content is provided above, incorporate and expand on its guidance rather than duplicating it.""",
                }
            ],
        )
    except openai.APIStatusError as exc:
        endpoint = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
        raise RuntimeError(
            f"Skill generation failed — HTTP {exc.status_code} from {endpoint!r} "
            f"with model {resolved_model!r}: {exc.message}"
        ) from exc
    except openai.APIConnectionError as exc:
        endpoint = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
        raise RuntimeError(
            f"Skill generation failed — could not connect to {endpoint!r}: {exc}"
        ) from exc

    content = message.choices[0].message.content
    if not content:
        raise RuntimeError(
            f"Skill generation failed — model {resolved_model!r} returned an empty response"
        )
    return content


def save_skill(skill: str, repo_name: str, output_dir: str = ".claude/skills") -> Path:
    """Write skill to {output_dir}/{short_repo_name}/SKILL.md.

    Args:
        skill: Skill content string.
        repo_name: 'owner/repo' or plain 'repo' name.
        output_dir: Base directory for skills (default: .claude/skills).

    Returns:
        Path to the written file.
    """
    short_name = repo_name.split("/")[-1]
    path = Path(output_dir) / short_name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(skill)
    return path
