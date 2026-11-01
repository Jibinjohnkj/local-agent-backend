"""Unit tests for the process helpers (Requirements 17.4, 17.6, 17.11).

Covers the pure parts of `oscompat/process.py` and the CTRL_BREAK helper `oscompat/ctrl_break.py`:

- `termination_plan` templates for every platform and job outcome (17.11);
- the `PATHEXT` default when the child environment has none (17.4);
- a `.ps1` script is found but rejected as not launchable (17.6);
- the `SpawnRejected` message for a command that resolves to nothing (17.6);
- the helper's command-line handling, and its Windows call order (17.11), which only runs on
  Windows.

The resolution rules take the platform as an argument, so the Windows cases run on every host.
"""

import asyncio
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

from conftest import posix_only, windows_only
from local_agent.config.models import StdioServerConfig
from local_agent.oscompat import _win32, ctrl_break
from local_agent.oscompat.process import (
    DEFAULT_PATHEXT,
    NotFound,
    Resolved,
    SpawnRejected,
    Step,
    build_spawn_spec,
    resolve_command,
    termination_plan,
)


def _files(*existing: str) -> Callable[[str], bool]:
    """A fake file system: only the listed paths are files."""
    present = frozenset(existing)
    return present.__contains__


def _stdio(command: str, *args: str, cwd: str | None = None) -> StdioServerConfig:
    return StdioServerConfig(
        name="srv", transport="stdio", command=command, args=list(args), cwd=cwd
    )


def _resolve_win(
    command: str,
    *existing: str,
    path_value: str = "C:\\bin",
    pathext_value: str | None = None,
) -> Resolved | NotFound:
    return resolve_command(
        command,
        platform="win32",
        path_value=path_value,
        pathext_value=pathext_value,
        base_dir="C:\\work",
        home="C:\\Users\\u",
        is_file=_files(*existing),
    )


# --- termination_plan (17.11) ----------------------------------------------------------------


class TestTerminationPlan:
    @pytest.mark.parametrize("platform", ["linux", "darwin"])
    @pytest.mark.parametrize("job_assigned", [True, False])
    def test_posix_kills_the_process_group_whatever_the_job_flag(
        self, platform: str, job_assigned: bool
    ) -> None:
        assert termination_plan(platform, job_assigned) == (
            Step.CLOSE_STDIN,
            Step.GRACEFUL,
            Step.WAIT_GRACE,
            Step.FORCE_GROUP,
            Step.REAP,
        )

    def test_windows_with_a_job_object_terminates_the_job(self) -> None:
        assert termination_plan("win32", True) == (
            Step.CLOSE_STDIN,
            Step.GRACEFUL,
            Step.WAIT_GRACE,
            Step.FORCE_JOB,
            Step.REAP,
        )

    def test_windows_without_a_job_object_falls_back_to_taskkill(self) -> None:
        assert termination_plan("win32", False) == (
            Step.CLOSE_STDIN,
            Step.GRACEFUL,
            Step.WAIT_GRACE,
            Step.FORCE_TASKKILL,
            Step.REAP,
        )

    @pytest.mark.parametrize(
        ("platform", "job_assigned"),
        [("linux", False), ("darwin", False), ("win32", True), ("win32", False)],
    )
    def test_every_plan_has_one_force_step_between_the_wait_and_the_reap(
        self, platform: str, job_assigned: bool
    ) -> None:
        plan = termination_plan(platform, job_assigned)
        forces = {Step.FORCE_GROUP, Step.FORCE_JOB, Step.FORCE_TASKKILL}
        assert [step for step in plan if step in forces] == [plan[3]]
        assert plan[0] is Step.CLOSE_STDIN
        assert plan[-1] is Step.REAP


# --- PATHEXT default (17.4) ------------------------------------------------------------------


class TestPathextDefault:
    def test_default_value_lists_the_four_launchable_extensions_in_order(self) -> None:
        assert DEFAULT_PATHEXT == ".COM;.EXE;.BAT;.CMD"

    @pytest.mark.parametrize(
        ("suffix", "kind"),
        [(".COM", "native"), (".EXE", "native"), (".BAT", "batch"), (".CMD", "batch")],
    )
    def test_unset_pathext_probes_each_default_extension(self, suffix: str, kind: str) -> None:
        found = _resolve_win("tool", f"C:\\bin\\tool{suffix}", pathext_value=None)
        assert found == Resolved(f"C:\\bin\\tool{suffix}", kind)  # type: ignore[arg-type]

    def test_unset_pathext_tries_extensions_in_default_order(self) -> None:
        found = _resolve_win("tool", "C:\\bin\\tool.CMD", "C:\\bin\\tool.EXE", "C:\\bin\\tool.BAT")
        assert found == Resolved("C:\\bin\\tool.EXE", "native")

    def test_unset_pathext_does_not_try_other_extensions(self) -> None:
        found = _resolve_win("tool", "C:\\bin\\tool.PS1", "C:\\bin\\tool.VBS", "C:\\bin\\tool.JS")
        assert found == NotFound("tool", ("C:\\bin",))

    def test_unset_pathext_matches_the_explicit_default_value(self) -> None:
        files = ("C:\\bin\\tool.BAT", "C:\\bin\\tool.CMD")
        assert _resolve_win("tool", *files, pathext_value=None) == _resolve_win(
            "tool", *files, pathext_value=DEFAULT_PATHEXT
        )

    def test_a_set_pathext_replaces_the_default(self) -> None:
        files = ("C:\\bin\\tool.EXE", "C:\\bin\\tool.CMD")
        found = _resolve_win("tool", *files, pathext_value=".CMD")
        assert found == Resolved("C:\\bin\\tool.CMD", "batch")

    def test_a_command_with_a_default_extension_is_probed_as_written(self) -> None:
        found = _resolve_win("npx.cmd", "C:\\bin\\npx.cmd", "C:\\bin\\npx.cmd.EXE")
        assert found == Resolved("C:\\bin\\npx.cmd", "batch")

    def test_pathext_is_not_read_on_posix(self) -> None:
        found = resolve_command(
            "tool",
            platform="linux",
            path_value="/usr/bin",
            pathext_value=".EXE",
            base_dir="/work",
            home="/home/u",
            is_file=_files("/usr/bin/tool.EXE"),
        )
        assert found == NotFound("tool", ("/usr/bin",))

    def test_build_spawn_spec_uses_the_default_when_the_child_env_has_no_pathext(self) -> None:
        spec = asyncio.run(
            build_spawn_spec(
                _stdio("tool"),
                "C:\\cfg",
                "win32",
                environ={"PATH": "C:\\bin", "SYSTEMROOT": "C:\\Windows"},
                is_file=_files("C:\\bin\\tool.EXE"),
            )
        )
        assert spec.application == "C:\\bin\\tool.EXE"
        assert "PATHEXT" not in spec.env

    def test_build_spawn_spec_honours_a_pathext_set_in_the_backend_environment(self) -> None:
        environ = {"PATH": "C:\\bin", "PATHEXT": ".CMD", "SYSTEMROOT": "C:\\Windows"}
        with pytest.raises(SpawnRejected):
            asyncio.run(
                build_spawn_spec(
                    _stdio("tool"),
                    "C:\\cfg",
                    "win32",
                    environ=environ,
                    is_file=_files("C:\\bin\\tool.EXE"),
                )
            )


# --- .ps1 is not launchable (17.6) -----------------------------------------------------------


class TestPowerShellScriptRejected:
    def test_a_ps1_path_command_is_found_but_not_launchable(self) -> None:
        outcome = _resolve_win("C:\\tools\\x.ps1", "C:\\tools\\x.ps1")
        assert isinstance(outcome, NotFound)
        assert outcome.found == "C:\\tools\\x.ps1"
        assert "not a launchable executable" in outcome.message
        assert "C:\\tools\\x.ps1" in outcome.message

    def test_a_ps1_listed_in_pathext_is_found_first_and_rejected(self) -> None:
        outcome = _resolve_win(
            "tool",
            "C:\\bin\\tool.PS1",
            "C:\\bin\\tool.EXE",
            pathext_value=".PS1;.EXE",
        )
        assert isinstance(outcome, NotFound)
        assert outcome.found == "C:\\bin\\tool.PS1"

    def test_a_bare_ps1_name_is_not_resolved_through_path(self) -> None:
        outcome = _resolve_win("tool.ps1", "C:\\bin\\tool.ps1")
        assert isinstance(outcome, NotFound)
        assert outcome.found is None

    def test_the_ps1_suffix_is_matched_without_regard_to_case(self) -> None:
        outcome = _resolve_win("C:\\tools\\X.PS1", "C:\\tools\\X.PS1")
        assert isinstance(outcome, NotFound)
        assert outcome.found == "C:\\tools\\X.PS1"

    def test_other_non_launchable_suffixes_and_no_suffix_are_rejected_too(self) -> None:
        for command in ("C:\\tools\\x.vbs", "C:\\tools\\x.js", "C:\\tools\\x."):
            outcome = _resolve_win(command, command)
            assert isinstance(outcome, NotFound), command
            assert outcome.found == command

    def test_posix_launches_a_ps1_file_like_any_other_executable(self) -> None:
        outcome = resolve_command(
            "/opt/tools/x.ps1",
            platform="linux",
            path_value="/usr/bin",
            pathext_value=None,
            base_dir="/work",
            home="/home/u",
            is_file=_files("/opt/tools/x.ps1"),
        )
        assert outcome == Resolved("/opt/tools/x.ps1", "native")

    def test_build_spawn_spec_rejects_a_ps1_target_with_the_reason(self) -> None:
        with pytest.raises(SpawnRejected) as caught:
            asyncio.run(
                build_spawn_spec(
                    _stdio("C:\\tools\\x.ps1"),
                    "C:\\cfg",
                    "win32",
                    environ={"PATH": "C:\\bin", "SYSTEMROOT": "C:\\Windows"},
                    is_file=_files("C:\\tools\\x.ps1"),
                )
            )
        assert "not a launchable executable" in caught.value.reason
        assert "C:\\tools\\x.ps1" in caught.value.reason


# --- SpawnRejected for an unresolved command (17.6) -------------------------------------------


class TestUnresolvedCommand:
    def test_posix_message_names_the_command_and_every_searched_directory(self) -> None:
        with pytest.raises(SpawnRejected) as caught:
            asyncio.run(
                build_spawn_spec(
                    _stdio("no-such-server"),
                    "/etc/app",
                    "linux",
                    environ={"PATH": "/usr/bin:/opt/bin", "HOME": "/home/u"},
                    is_file=_files(),
                )
            )
        expected = "no executable found for command 'no-such-server' (searched: /usr/bin, /opt/bin)"
        assert caught.value.reason == expected
        assert str(caught.value) == expected

    def test_windows_message_names_the_command_and_every_searched_directory(self) -> None:
        with pytest.raises(SpawnRejected) as caught:
            asyncio.run(
                build_spawn_spec(
                    _stdio("nope"),
                    "C:\\cfg",
                    "win32",
                    environ={"PATH": "C:\\bin;D:\\tools", "SYSTEMROOT": "C:\\Windows"},
                    is_file=_files(),
                )
            )
        assert caught.value.reason == (
            "no executable found for command 'nope' (searched: C:\\bin, D:\\tools)"
        )

    def test_an_empty_path_says_that_nothing_was_searched(self) -> None:
        with pytest.raises(SpawnRejected) as caught:
            asyncio.run(
                build_spawn_spec(
                    _stdio("nope"),
                    "/etc/app",
                    "linux",
                    environ={"HOME": "/home/u"},
                    is_file=_files(),
                )
            )
        assert caught.value.reason == "no executable found for command 'nope' (searched: (none))"

    def test_a_path_command_is_reported_resolved_with_the_directory_it_probed(self) -> None:
        with pytest.raises(SpawnRejected) as caught:
            asyncio.run(
                build_spawn_spec(
                    _stdio("bin/server"),
                    "/etc/app",
                    "linux",
                    environ={"PATH": "/usr/bin", "HOME": "/home/u"},
                    is_file=_files("/usr/bin/bin/server"),
                )
            )
        assert caught.value.reason == (
            "no executable found for command '/etc/app/bin/server' (searched: /etc/app/bin)"
        )

    def test_the_reason_is_the_resolvers_message(self) -> None:
        outcome = resolve_command(
            "nope",
            platform="linux",
            path_value="/usr/bin:/opt/bin",
            pathext_value=None,
            base_dir="/etc/app",
            home=None,
            is_file=_files(),
        )
        assert isinstance(outcome, NotFound)
        assert outcome.message == (
            "no executable found for command 'nope' (searched: /usr/bin, /opt/bin)"
        )

    def test_spawn_rejected_is_an_exception_that_carries_its_reason(self) -> None:
        error = SpawnRejected("because")
        assert isinstance(error, Exception)
        assert error.reason == "because"
        assert str(error) == "because"


# --- ctrl_break helper (17.11) ----------------------------------------------------------------


class TestCtrlBreakCommandLine:
    @pytest.mark.parametrize(
        "argv",
        [[], ["abc"], ["0"], ["-1"], ["1", "2"], [""], ["1.5"], ["\uff11\uff12"], [" 7"]],
    )
    def test_anything_but_one_positive_integer_is_a_usage_error(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert ctrl_break.main(argv) == ctrl_break.EXIT_USAGE
        assert "usage: python -m local_agent.oscompat.ctrl_break <pid>" in capsys.readouterr().err

    def test_exit_statuses_are_distinct(self) -> None:
        statuses = {
            ctrl_break.EXIT_OK,
            ctrl_break.EXIT_WIN32_FAILURE,
            ctrl_break.EXIT_USAGE,
            ctrl_break.EXIT_NOT_WINDOWS,
        }
        assert statuses == {0, 1, 2, 3}

    @posix_only
    def test_off_windows_the_helper_reports_that_it_is_unsupported(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert ctrl_break.main(["1234"]) == ctrl_break.EXIT_NOT_WINDOWS
        assert "only supported on Windows" in capsys.readouterr().err

    @posix_only
    def test_running_the_module_off_windows_exits_with_the_unsupported_status(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "local_agent.oscompat.ctrl_break", "1234"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert done.returncode == ctrl_break.EXIT_NOT_WINDOWS
        assert "only supported on Windows" in done.stderr

    @posix_only
    def test_running_the_module_without_a_pid_exits_with_the_usage_status(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "local_agent.oscompat.ctrl_break"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert done.returncode == ctrl_break.EXIT_USAGE
        assert "usage:" in done.stderr


class _Win32Calls:
    """Replaces the four console bindings in `_win32` and records the calls in order.

    A binding named in `failing` returns 0 (the Windows failure value); the others return 1.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, failing: frozenset[str] = frozenset()):
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        for name in (
            "FreeConsole",
            "AttachConsole",
            "SetConsoleCtrlHandler",
            "GenerateConsoleCtrlEvent",
        ):
            monkeypatch.setattr(_win32, name, self._make(name, failing))

    def _make(self, name: str, failing: frozenset[str]) -> Callable[..., int]:
        def call(*args: Any) -> int:
            self.calls.append((name, args))
            return 0 if name in failing else 1

        return call

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


@windows_only
class TestCtrlBreakOnWindows:
    def test_the_four_calls_run_in_order_against_the_given_pid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        win32 = _Win32Calls(monkeypatch)
        assert ctrl_break.main(["4321"]) == ctrl_break.EXIT_OK
        assert win32.calls == [
            ("FreeConsole", ()),
            ("AttachConsole", (4321,)),
            ("SetConsoleCtrlHandler", (None, True)),
            ("GenerateConsoleCtrlEvent", (_win32.CTRL_BREAK_EVENT, 0)),
        ]

    def test_a_free_console_failure_is_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        win32 = _Win32Calls(monkeypatch, failing=frozenset({"FreeConsole"}))
        assert ctrl_break.main(["4321"]) == ctrl_break.EXIT_OK
        assert win32.names[-1] == "GenerateConsoleCtrlEvent"

    @pytest.mark.parametrize(
        ("failing", "calls_made"),
        [
            ("AttachConsole", 2),
            ("SetConsoleCtrlHandler", 3),
            ("GenerateConsoleCtrlEvent", 4),
        ],
    )
    def test_a_failed_call_stops_the_sequence_and_names_the_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        failing: str,
        calls_made: int,
    ) -> None:
        win32 = _Win32Calls(monkeypatch, failing=frozenset({failing}))
        assert ctrl_break.main(["4321"]) == ctrl_break.EXIT_WIN32_FAILURE
        assert len(win32.calls) == calls_made
        assert f"ctrl_break: {failing} failed" in capsys.readouterr().err

    def test_a_bad_pid_makes_no_windows_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        win32 = _Win32Calls(monkeypatch)
        assert ctrl_break.main(["zero"]) == ctrl_break.EXIT_USAGE
        assert win32.calls == []
