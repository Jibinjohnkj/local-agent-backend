"""Windows-only process tests (task 22.11).

These run the real `oscompat.process` layer (`build_spawn_spec`, `spawn`, `terminate_tree`) against
the Windows APIs: job objects, `CREATE_SUSPENDED`, the hidden console behind `CTRL_BREAK_EVENT`,
`taskkill`, and `cmd.exe`. Every test is skipped on other systems (`windows_only`).

* `test_ctrl_break_reaches_tree` (17.11): the `ctrl_break` helper ends a three-level tree without
  stdin closing and without a force step, which confirms that `AttachConsole` reaches the hidden
  console of the child.
* `test_job_assignment_before_resume` (17.10, 17.11): with `--spawn-first` the server starts its
  child as its first instruction, yet the child and grandchild are in the job. Only an assignment
  before the primary thread resumes can guarantee that. The server ignores the graceful signal, so
  the job kill ends the tree.
* `test_taskkill_fallback` (17.11, 17.12): job creation is forced to fail. The warning is logged,
  `termination_plan` picks `taskkill /T`, and a tree that ignores `CTRL_BREAK_EVENT` is still gone.
* `test_batch_args_literal`, `test_batch_args_rejected` (17.8, 17.9): the metacharacters of
  `cmd.exe` reach a C runtime program as the original text through `tests/fixtures/echo_args.cmd`,
  and `%` and `"` are rejected before anything starts.
* `test_npx_resolution` (17.4): a bare `npx` resolves to `npx.cmd` through the child's `PATH` and
  `PATHEXT`, never to the extensionless script that ships next to it.
* `test_env_case_merge` (17.22): a configured `Path` replaces the inherited `PATH`, and the child's
  raw environment block holds exactly one entry for it.
* `test_reference_parser_matches_python_argv` (17.7): the `winargv_ref` oracle of the property
  tests agrees with what `python.exe` really reads, for a fixed corpus.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pytest
from winargv_ref import parse_command_line

from integration_harness import FIXTURES_DIR, PIDS, wait_until_gone
from local_agent.config.models import StdioServerConfig
from local_agent.oscompat import process as process_module
from local_agent.oscompat.process import (
    REAP_TIMEOUT_S,
    STDIN_WAIT_S,
    Resolved,
    SpawnRejected,
    SpawnSpec,
    Step,
    TreeHandle,
    build_command_line,
    build_spawn_spec,
    spawn,
    system_tree_ops,
    terminate_tree,
)

# Same marker as `windows_only` in tests/conftest.py. It cannot be imported here: inside
# tests/integration the module name `conftest` is this directory's own conftest.py.
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific OS behaviour")

SPAWNING_SERVER: Final = FIXTURES_DIR / "mcp_spawning_server.py"
STUBBORN_TREE: Final = FIXTURES_DIR / "stubborn_tree.py"
ECHO_ARGS_CMD: Final = FIXTURES_DIR / "echo_args.cmd"
PLATFORM: Final = "win32"
SERVER_NAME: Final = "win-test"
GRACE_S: Final = 1.0
PUBLISH_TIMEOUT_S: Final = 15.0
PROGRAM_TIMEOUT_S: Final = 30.0
# 17.12 bound plus a margin for the slow Windows runners the other integration tests allow for.
GONE_TIMEOUT_S: Final = STDIN_WAIT_S + GRACE_S + REAP_TIMEOUT_S + 2.0

ARGV_CODE: Final = "import sys, json; print(json.dumps(sys.argv[1:]))"


# --- helpers ------------------------------------------------------------------------------------


class RecordingLog:
    """A `ProcessLogger` that keeps every record, so a test can look for a warning."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, object]]] = []

    def events(self, level: str | None = None) -> list[str]:
        return [event for lvl, event, _ in self.records if level in (None, lvl)]

    def fields(self, event: str) -> dict[str, object]:
        return next(fields for _, name, fields in self.records if name == event)

    def debug(self, event: str, **fields: object) -> None:
        self.records.append(("debug", event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.records.append(("info", event, fields))

    def warning(self, event: str, **fields: object) -> None:
        self.records.append(("warning", event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.records.append(("error", event, fields))


async def make_spec(
    tmp_path: Path,
    command: str,
    args: Sequence[str] = (),
    env: dict[str, str] | None = None,
) -> SpawnSpec:
    """The production `SpawnSpec` of a `stdio` server, built for Windows from the real `environ`."""
    cfg = StdioServerConfig.model_validate(
        {
            "name": "win",
            "transport": "stdio",
            "command": command,
            "args": list(args),
            "env": env or {},
        }
    )
    return await build_spawn_spec(cfg, str(tmp_path), PLATFORM)


async def eventually(check: Callable[[], bool], timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not check():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout_s} s waiting for {what}")
        await asyncio.sleep(0.05)


def read_spawning_tree(pid_file: Path) -> list[int] | None:
    """`[server, grandchild, great-grandchild]` from the spawning server's pid file, once known."""
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    descendants = [int(pid) for pid in data.get("descendants", ())]
    if "server" not in data or len(descendants) != 2:
        return None
    return [int(data["server"]), *descendants]


def read_stubborn_tree(pid_file: Path, depth: int) -> list[int] | None:
    """The pids `stubborn_tree.py` has written, once all `depth + 1` of them are there."""
    try:
        lines = pid_file.read_text(encoding="utf-8").split()
    except OSError:
        return None
    return [int(line) for line in lines] if len(lines) == depth + 1 else None


async def wait_for_tree(handle: TreeHandle, read: Callable[[], list[int] | None]) -> list[int]:
    """Wait until the server has published its whole tree. Returns every pid, the leader's too."""
    found: list[list[int]] = []

    def ready() -> bool:
        pids = read()
        if pids is not None:
            found.append(pids)
        return pids is not None

    await eventually(ready, PUBLISH_TIMEOUT_S, "the server to publish its process tree")
    pids = sorted({handle.pid, *found[-1]})
    PIDS.record(*pids)
    return pids


async def assert_all_gone(pids: Sequence[int], timeout_s: float = GONE_TIMEOUT_S) -> None:
    leftovers = await asyncio.to_thread(wait_until_gone, list(pids), timeout_s)
    assert leftovers == [], f"processes of the tree are still alive: {leftovers}"


async def run_to_completion(spec: SpawnSpec) -> str:
    """Start `spec`, read its stdout until it closes, stop the tree, and return the text."""
    handle = await spawn(spec, server=SERVER_NAME)
    PIDS.record(handle.pid)
    output = bytearray()
    try:
        async with asyncio.timeout(PROGRAM_TIMEOUT_S):
            async for chunk in handle.stdout:
                output.extend(chunk)
    finally:
        await terminate_tree(handle, GRACE_S)
    return output.decode("utf-8")


# --- CTRL_BREAK_EVENT ---------------------------------------------------------------------------


async def test_ctrl_break_reaches_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.json"
    spec = await make_spec(
        tmp_path,
        sys.executable,
        [str(SPAWNING_SERVER), "--no-initialize", "--pid-file", str(pid_file)],
    )
    handle = await spawn(spec, server=SERVER_NAME)
    PIDS.record(handle.pid)
    ops = system_tree_ops()
    try:
        pids = await wait_for_tree(handle, lambda: read_spawning_tree(pid_file))
        assert len(pids) >= 3
        assert wait_until_gone(pids, 0) == pids  # every process of the tree is alive
        assert not ops.group_empty(handle)

        # stdin stays open and no force step runs: the event alone must end every process. The
        # helper's own status is not asserted, because the helper shares the console it signals.
        reason = await ops.graceful(handle)
        leftovers = await asyncio.to_thread(wait_until_gone, pids, 5.0)
        assert leftovers == [], (
            f"CTRL_BREAK_EVENT did not reach {leftovers} (helper said {reason!r})"
        )
        await eventually(lambda: ops.group_empty(handle), 2.0, "the job to report no process")

        report = await terminate_tree(handle, GRACE_S)
        assert report.forced is False
        assert report.survivors == ()
    finally:
        await terminate_tree(handle, 0.0)


# --- job assignment before resume ---------------------------------------------------------------


async def test_job_assignment_before_resume(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.json"
    spec = await make_spec(
        tmp_path,
        sys.executable,
        [
            str(SPAWNING_SERVER),
            "--spawn-first",
            "--ignore-graceful",
            "--no-initialize",
            "--pid-file",
            str(pid_file),
        ],
    )
    assert spec.creationflags & process_module.CREATE_SUSPENDED
    log = RecordingLog()
    handle = await spawn(spec, server=SERVER_NAME, log=log)
    PIDS.record(handle.pid)
    try:
        assert handle.job_assigned is True
        assert "job_object_unavailable" not in log.events()
        pids = await wait_for_tree(handle, lambda: read_spawning_tree(pid_file))

        # The first thing the server does is start its child. That child, and the grandchild after
        # it, are in the job only because the server was assigned before its thread resumed.
        members = set(await system_tree_ops().survivors(handle))
        assert set(pids) <= members, f"processes outside the job: {sorted(set(pids) - members)}"

        started = time.monotonic()
        report = await terminate_tree(handle, GRACE_S)
        elapsed = time.monotonic() - started
    finally:
        await terminate_tree(handle, 0.0)

    assert Step.FORCE_JOB in report.steps
    assert Step.FORCE_TASKKILL not in report.steps
    assert report.forced is True
    assert report.survivors == ()
    assert elapsed <= STDIN_WAIT_S + GRACE_S + REAP_TIMEOUT_S + 1.0
    await assert_all_gone(pids)


# --- taskkill fallback --------------------------------------------------------------------------


async def test_taskkill_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    depth = 2
    pid_file = tmp_path / "stubborn.pids"
    monkeypatch.setattr(
        process_module, "_create_job", lambda: (None, "CreateJobObjectW failed: forced by the test")
    )
    spec = await make_spec(
        tmp_path, sys.executable, [str(STUBBORN_TREE), str(pid_file), str(depth)]
    )
    log = RecordingLog()
    handle = await spawn(spec, server=SERVER_NAME, log=log)
    PIDS.record(handle.pid)
    try:
        assert handle.job_assigned is False
        assert "job_object_unavailable" in log.events("warning")
        assert "forced by the test" in str(log.fields("job_object_unavailable")["reason"])
        pids = await wait_for_tree(handle, lambda: read_stubborn_tree(pid_file, depth))
        assert len(pids) >= depth + 1

        report = await terminate_tree(handle, GRACE_S)
    finally:
        await terminate_tree(handle, 0.0)

    # The tree ignores CTRL_BREAK_EVENT and never reads stdin, so only the force step ends it.
    assert Step.FORCE_TASKKILL in report.steps
    assert Step.FORCE_JOB not in report.steps
    assert report.forced is True
    assert report.survivors == ()
    await assert_all_gone(pids)


# --- batch files --------------------------------------------------------------------------------

ARGS_ENV: Final = {"ECHO_ARGS_PYTHON": sys.executable}


@pytest.mark.parametrize(
    "args",
    [
        ["a&b"],
        ["x|y"],
        ["<z>"],
        ["c^d"],
        ["sp ace"],
        ["trail\\"],
        ["a&b", "x|y", "<z>", "c^d", "sp ace", "trail\\"],
    ],
    ids=["ampersand", "pipe", "redirects", "caret", "space", "trailing-backslash", "all-together"],
)
async def test_batch_args_literal(tmp_path: Path, args: list[str]) -> None:
    spec = await make_spec(tmp_path, str(ECHO_ARGS_CMD), args, ARGS_ENV)
    assert spec.application is not None
    assert spec.application.lower().endswith("cmd.exe")

    output = await run_to_completion(spec)

    assert json.loads(output) == args


@pytest.mark.parametrize(
    ("args", "position", "char"),
    [
        (["50%"], 1, "%"),
        (['say "hi"'], 1, '"'),
        (["fine", "%PATH%"], 2, "%"),
        (["ok", "ok", 'q"'], 3, '"'),
    ],
)
async def test_batch_args_rejected(
    tmp_path: Path, args: list[str], position: int, char: str
) -> None:
    with pytest.raises(SpawnRejected) as caught:
        await make_spec(tmp_path, str(ECHO_ARGS_CMD), args, ARGS_ENV)

    assert f"argument {position}" in caught.value.reason
    assert repr(char) in caught.value.reason


# --- npx.cmd resolution -------------------------------------------------------------------------


def make_npx_directory(directory: Path) -> None:
    """`npx` (the shell script), `npx.cmd`, and `npx.bat`; each batch file says which one ran."""
    directory.mkdir()
    (directory / "npx").write_bytes(b"#!/bin/sh\nexit 1\n")
    (directory / "npx.cmd").write_bytes(b"@echo off\r\necho ran-npx-cmd\r\n")
    (directory / "npx.bat").write_bytes(b"@echo off\r\necho ran-npx-bat\r\n")


@pytest.mark.parametrize(
    ("pathext", "expected"),
    [(".COM;.EXE;.CMD;.BAT", "ran-npx-cmd"), (".COM;.EXE;.BAT;.CMD", "ran-npx-bat")],
)
async def test_npx_resolution(tmp_path: Path, pathext: str, expected: str) -> None:
    bin_dir = tmp_path / "bin"
    make_npx_directory(bin_dir)

    # The configured PATH and PATHEXT belong to the child, and they alone decide the search.
    spec = await make_spec(tmp_path, "npx", [], {"PATH": str(bin_dir), "PATHEXT": pathext})
    assert spec.application is not None
    assert spec.application.lower().endswith("cmd.exe")
    assert spec.command_line is not None
    assert f'npx.{expected.rsplit("-", 1)[1]}"' in spec.command_line.lower()

    output = await run_to_completion(spec)

    assert output.strip() == expected


async def test_npx_is_not_found_when_the_childs_pathext_lacks_batch_suffixes(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    make_npx_directory(bin_dir)

    with pytest.raises(SpawnRejected) as caught:
        await make_spec(tmp_path, "npx", [], {"PATH": str(bin_dir), "PATHEXT": ".COM;.EXE"})

    assert "npx" in caught.value.reason
    assert str(bin_dir) in caught.value.reason


# --- environment names --------------------------------------------------------------------------

ENV_PROBE: Final = """\
import ctypes, json
kernel32 = ctypes.WinDLL("kernel32")
kernel32.GetEnvironmentStringsW.restype = ctypes.c_void_p
address = kernel32.GetEnvironmentStringsW()
entries = []
while True:
    entry = ctypes.wstring_at(address)
    if not entry:
        break
    entries.append(entry)
    address += (len(entry) + 1) * 2
print(json.dumps([entry for entry in entries if entry.upper().startswith("PATH=")]))
"""


@pytest.mark.parametrize("name", ["Path", "path", "PATH"])
async def test_env_case_merge(tmp_path: Path, name: str) -> None:
    assert "PATH" in {key.upper() for key in os.environ}, "the runner has no PATH to replace"
    configured = str(tmp_path / "configured-bin")
    probe = tmp_path / "env_probe.py"
    probe.write_text(ENV_PROBE, encoding="utf-8")

    spec = await make_spec(tmp_path, sys.executable, [str(probe)], {name: configured})

    assert [key for key in spec.env if key.upper() == "PATH"] == [name]
    output = await run_to_completion(spec)

    # The raw environment block, not `os.environ`, which folds spellings together.
    assert json.loads(output) == [f"{name}={configured}"]


# --- the reference parser against the real runtime ----------------------------------------------

ARGV_CORPUS: Final[list[list[str]]] = [
    [],
    ["plain"],
    ["two words", "tab\tseparated"],
    ['say "hi"', '"'],
    ["\\", "trail\\", "trail\\\\", "dir with space\\"],
    ['a\\"b', '\\\\"', 'x\\\\\\"y'],
    ['""', '" "', 'a"b c"d'],
    ["a&b", "x|y", "<z>", "c^d", "%PATH%", "!x!", "(a)"],
    ["", "after empty", ""],
    ["caf\u00e9", "\u20ac", "\U0001f600", "a\u00a0b", "\u3000"],
    ["C:\\Program Files\\My App\\", "C:\\a b\\c d.txt"],
]


@pytest.mark.parametrize(
    "args", ARGV_CORPUS, ids=[f"corpus-{index}" for index in range(len(ARGV_CORPUS))]
)
def test_reference_parser_matches_python_argv(args: list[str]) -> None:
    line = build_command_line(
        Resolved(sys.executable, "native"), ["-c", ARGV_CODE, *args], os.environ["SYSTEMROOT"]
    )

    done = subprocess.run(
        line,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=PROGRAM_TIMEOUT_S,
        check=True,
    )

    assert json.loads(done.stdout.decode("ascii")) == args
    assert parse_command_line(line) == [sys.executable, "-c", ARGV_CODE, *args]
