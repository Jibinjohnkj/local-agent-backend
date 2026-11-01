"""Property 46: Cross-platform transcript equivalence (task 22.15).

Validates Requirements 17.3.

A fixed, derandomized corpus of scenario scripts runs on both Session_Stores through the real
`TaskRunner`. Each run becomes a normalized transcript: the Status_Event and Token_Event sequence,
the final Message_History, and the Task as an A2A client sees it, with generated IDs, timestamps,
and durations replaced by placeholders. A corpus of Backend_Config texts adds the config validation
results (the loaded model, or every issue path and message).

The transcripts are folded into one SHA-256 digest. The test writes it to
`artifacts/transcripts-<os>-<arch>.sha256`. CI uploads that file from each runner, and the
`equivalence` job requires all five runners to report the same first field. Nothing platform
specific may reach the digest: no paths, no process or thread IDs, no wall-clock times. The corpus
is the same on every runner because `derandomize=True` seeds Hypothesis from the test's source.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from scenarios import (
    ScenarioScript,
    StoreKind,
    build_transcript,
    open_store,
    run_scenario,
    scenario_scripts,
    transcript_json,
)

from local_agent.config.loader import ConfigError, load_config_from_text

# Pinned, so the corpus does not change with the HYPOTHESIS_PROFILE (dev 100, ci 200).
CORPUS_EXAMPLES = 100
STORES: tuple[StoreKind, ...] = ("memory", "sqlite")
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

ENV = {
    "AGENT_BEARER_TOKEN": "b" * 40,
    "PG_MCP_TOKEN": "pg-secret-token",
    "LLM_MODEL": "ollama_chat/llama3.1",
}

# Backend_Config texts with platform-neutral inputs: no Path_Fields, so nothing resolves against
# the working directory or HOME, and every environment name is upper case.
CONFIG_CORPUS: tuple[str, ...] = (
    "llm:\n  model: ollama_chat/llama3.1\n",
    "llm:\n  model: ${LLM_MODEL}\n  api_base: http://localhost:11434\n",
    (
        "server:\n  host: 127.0.0.1\n  port: 10000\n"
        '  cors_allowed_origins: ["http://localhost:5173"]\n'
        "llm:\n  model: ollama_chat/llama3.1\n"
        "orchestration:\n  approval_timeout_s: 600\n"
        "mcp_servers:\n"
        "  - name: fs\n    transport: stdio\n    command: npx\n"
        '    args: ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/me/notes"]\n'
        '    env: { NODE_OPTIONS: "--max-old-space-size=512" }\n'
        "  - name: pg\n    transport: sse\n    url: http://localhost:8931/sse\n"
        '    headers: { Authorization: "Bearer ${PG_MCP_TOKEN}" }\n'
        'permissions:\n  fs: ["write_*", "delete_*", "move_file"]\n'
        "storage:\n  backend: memory\n"
    ),
    (
        "server:\n  host: 0.0.0.0\n  port: 8080\n  bearer_token: ${AGENT_BEARER_TOKEN}\n"
        "llm:\n  model: m\n"
    ),
    "server:\n  host: 0.0.0.0\n  port: 8080\nllm:\n  model: m\n",
    "server:\n  host: 192.168.1.10\n  bearer_token: short\nllm:\n  model: m\n",
    "server:\n  port: 70000\nllm:\n  model: m\n",
    "llm:\n  model: ${MISSING_MODEL}\n",
    "llm:\n  model: ''\n",
    "llm:\n  model: m\nstorage:\n  backend: postgres\n",
    (
        "llm:\n  model: m\n"
        "mcp_servers:\n"
        "  - name: dup\n    transport: stdio\n    command: a\n"
        "  - name: dup\n    transport: stdio\n    command: b\n"
        "  - name: bad name!\n    transport: stdio\n    command: c\n"
        "  - name: nocmd\n    transport: stdio\n"
        "  - name: odd\n    transport: ftp\n"
        "permissions:\n  ghost: ['x*']\n  dup: ['']\n"
    ),
    "- not\n- a\n- mapping\n",
    "llm: [unclosed\n",
    "llm:\n  model: m\nunknown_section: 1\n",
)


def config_result(text: str) -> dict[str, Any]:
    """What loading `text` yields: the validated model, or every issue as (path, message)."""
    try:
        config = load_config_from_text(text, ENV, "config.yaml", base_dir="/base")
    except ConfigError as error:
        return {"error": [[issue.path, issue.message] for issue in error.issues]}
    return {"config": config.model_dump(mode="json")}


def os_label() -> str:
    """`<os>-<arch>` for the digest file name, for example `linux-arm64` or `windows-x86_64`."""
    system = {"win32": "windows", "darwin": "macos"}.get(
        sys.platform, "linux" if sys.platform.startswith("linux") else sys.platform
    )
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{arch}"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_digest(entries: dict[str, str]) -> str:
    """One digest over `entries` (a key to a transcript text), independent of insertion order."""
    lines = (f"{key} {sha256_hex(text)}" for key, text in sorted(entries.items()))
    return sha256_hex("\n".join(lines))


# Feature: local-agent-backend, Property 46: Cross-platform transcript equivalence
def test_transcripts_are_equivalent_across_platforms() -> None:
    entries: dict[str, str] = {}
    runs = 0
    temp_root = Path(tempfile.gettempdir()).as_posix()

    @settings(
        derandomize=True,
        database=None,
        max_examples=CORPUS_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(script=scenario_scripts())
    def run_corpus(script: ScenarioScript) -> None:
        nonlocal runs
        runs += 1
        for kind in STORES:
            with open_store(kind) as env:
                text = transcript_json(build_transcript(run_scenario(script, env)))
            # Normalization must leave nothing of this machine in the transcript.
            assert temp_root not in text.replace("\\\\", "/")
            entries[f"{sha256_hex(repr(script))}/{kind}"] = text

    run_corpus()
    assert runs >= CORPUS_EXAMPLES

    for index, config_text in enumerate(CONFIG_CORPUS):
        result = json.dumps(
            config_result(config_text), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        assert temp_root not in result
        entries[f"config/{index:02d}"] = result

    digest = corpus_digest(entries)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    target = ARTIFACTS_DIR / f"transcripts-{os_label()}.sha256"
    target.write_text(f"{digest}  transcripts\n", encoding="utf-8", newline="\n")
    assert target.read_text(encoding="utf-8").split()[0] == digest
