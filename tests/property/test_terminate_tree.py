"""Property test for the process-tree termination schedule (Requirements 17.11, 17.15, 6.11).

`terminate_tree` is driven with a `FakeTree` (the injected `TreeOps`), a manually advanced clock,
and a fake sleep that advances that clock, so a 5 s grace period costs no wall time. Each member
of the fake tree exits at a scripted point:

- after stdin close, `delay` seconds after the `close_stdin` call;
- after the graceful request, `delay` seconds after the `graceful` call, but only when that call
  succeeded (a failed signal reaches nobody);
- never on its own, so only the forced termination removes it.

The forced termination removes every member at once, as `SIGKILL`, `TerminateJobObject`, and
`taskkill /T /F` do. The expected schedule is computed from the script and the start time alone,
never from what the implementation did, and is compared with the recorded OS calls:

- stdin close is the first call;
- the graceful request follows `stdin_wait_s` later, and only when a member is still alive then;
- exactly one force call, `grace` seconds after the graceful request, and only when a member is
  still alive at that moment; it carries the step that matches the platform and job outcome;
- the tree is empty at the end and the report lists no survivors.

Delays are 0 or `k * 0.25 + 0.1`, and every wait length is a multiple of 0.25, so no member exits
within 0.1 s of a deadline and float rounding in the clock cannot decide an example.
"""

import asyncio
from dataclasses import dataclass, field
from typing import cast

import anyio.abc
from hypothesis import given
from hypothesis import strategies as st

from local_agent.oscompat.process import (
    ProcessLogger,
    Step,
    TerminationReport,
    TreeHandle,
    TreeOps,
    terminate_tree,
    termination_plan,
)

# The longest a `_wait_empty` poll may overshoot its deadline is one poll interval (0.05 s).
_POLL_SLACK_S = 0.06
_START = 1_000.0

# Grace by shutdown reason: first Shutdown_Signal, console close, repeated Shutdown_Signal.
_GRACE_BY_REASON = {"signal": 5.0, "console_close": 2.0, "repeated_signal": 0.0}
_STDIN_WAIT_BY_REASON = {"signal": 1.0, "console_close": 0.5, "repeated_signal": 1.0}


class FakeClock:
    """A manually advanced monotonic clock (the `Clock` protocol)."""

    def __init__(self) -> None:
        self._now = _START

    def now_ms(self) -> int:
        return round(self._now * 1000)

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += max(seconds, 0.0)


class FakeSleep:
    """An awaitable `sleep(seconds)` that advances the clock instead of waiting."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    async def __call__(self, seconds: float) -> None:
        self._clock.advance(seconds)


@dataclass
class Member:
    """One process of the fake tree and the point at which it exits."""

    pid: int
    trigger: str  # "stdin", "graceful", or "never"
    delay: float
    exit_at: float | None = None
    killed: bool = False


@dataclass
class RecordingLog:
    """A `ProcessLogger` that keeps `(level, event, fields)` for every record."""

    records: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)

    def debug(self, event: str, **fields: object) -> None:
        self.records.append(("debug", event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.records.append(("info", event, fields))

    def warning(self, event: str, **fields: object) -> None:
        self.records.append(("warning", event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.records.append(("error", event, fields))


class FakeTree:
    """A scripted `TreeOps`: records each OS call with its time and drives member exits."""

    def __init__(self, clock: FakeClock, members: list[Member], *, graceful_ok: bool) -> None:
        self._clock = clock
        self.members = members
        self._graceful_ok = graceful_ok
        # (name, time, tree_empty_at_call_time, step or None)
        self.calls: list[tuple[str, float, bool, Step | None]] = []

    def _empty(self) -> bool:
        now = self._clock.monotonic()
        return all(m.killed or (m.exit_at is not None and m.exit_at <= now) for m in self.members)

    def _record(self, name: str, step: Step | None = None) -> None:
        self.calls.append((name, self._clock.monotonic(), self._empty(), step))

    async def close_stdin(self, handle: TreeHandle) -> None:
        self._record("close_stdin")
        now = self._clock.monotonic()
        for member in self.members:
            if member.trigger == "stdin":
                member.exit_at = now + member.delay

    def group_empty(self, handle: TreeHandle) -> bool:
        return self._empty()

    async def graceful(self, handle: TreeHandle) -> str | None:
        self._record("graceful")
        if not self._graceful_ok:
            return "signal refused"
        now = self._clock.monotonic()
        for member in self.members:
            if member.trigger == "graceful":
                member.exit_at = now + member.delay
        return None

    async def force(self, handle: TreeHandle, step: Step) -> str | None:
        self._record("force", step)
        for member in self.members:
            member.killed = True
        return None

    async def close_streams(self, handle: TreeHandle) -> None:
        self._record("close_streams")

    async def survivors(self, handle: TreeHandle) -> tuple[int, ...]:
        self._record("survivors")
        now = self._clock.monotonic()
        return tuple(
            m.pid
            for m in self.members
            if not (m.killed or (m.exit_at is not None and m.exit_at <= now))
        )

    async def release(self, handle: TreeHandle) -> None:
        self._record("release")


# --- strategies ------------------------------------------------------------------------------

_delays = st.one_of(
    st.just(0.0), st.integers(min_value=0, max_value=32).map(lambda k: k * 0.25 + 0.1)
)
_members = st.lists(
    st.tuples(st.sampled_from(["stdin", "graceful", "never"]), _delays), min_size=0, max_size=6
)


def _expected(
    members: list[tuple[str, float]], *, graceful_ok: bool, stdin_wait: float, grace: float
) -> tuple[bool, bool]:
    """(graceful requested, force needed), computed from the script alone.

    Start is time 0, stdin closes at 0, the graceful request (if any) is at `stdin_wait`.
    """
    alive_after_stdin = any(t != "stdin" or d > stdin_wait for t, d in members)
    if not alive_after_stdin:
        return False, False
    alive_after_grace = any(
        t == "never"
        or (t == "stdin" and d > stdin_wait + grace)
        or (t == "graceful" and (not graceful_ok or d > grace))
        for t, d in members
    )
    return True, alive_after_grace


def _only(
    calls: list[tuple[str, float, bool, Step | None]], name: str
) -> list[tuple[str, float, bool, Step | None]]:
    return [call for call in calls if call[0] == name]


@given(
    platform=st.sampled_from(["win32", "linux", "darwin"]),
    job_assigned=st.booleans(),
    reason=st.sampled_from(sorted(_GRACE_BY_REASON)),
    graceful_ok=st.booleans(),
    script=_members,
)
def test_terminate_tree_schedule(
    platform: str,
    job_assigned: bool,
    reason: str,
    graceful_ok: bool,
    script: list[tuple[str, float]],
) -> None:
    # Feature: local-agent-backend, Property 43: Process-tree termination schedule
    # **Validates: Requirements 17.11, 17.15, 6.11**
    grace = _GRACE_BY_REASON[reason]
    stdin_wait = _STDIN_WAIT_BY_REASON[reason]
    clock = FakeClock()
    members = [Member(pid=100 + i, trigger=t, delay=d) for i, (t, d) in enumerate(script)]
    tree = FakeTree(clock, members, graceful_ok=graceful_ok)
    log = RecordingLog()
    # The fake ops never touch the pipes, so the handle needs no real streams.
    handle = TreeHandle(
        platform=platform,
        pid=100,
        stdin=cast(anyio.abc.ByteSendStream, None),
        stdout=cast(anyio.abc.ByteReceiveStream, None),
        job_assigned=job_assigned,
        server="fake",
    )

    report: TerminationReport = asyncio.run(
        terminate_tree(
            handle,
            grace,
            stdin_wait,
            ops=cast(TreeOps, tree),
            clock=clock,
            sleep=FakeSleep(clock),
            log=cast(ProcessLogger, log),
        )
    )

    want_graceful, want_force = _expected(
        script, graceful_ok=graceful_ok, stdin_wait=stdin_wait, grace=grace
    )
    plan = termination_plan(platform, job_assigned)
    force_step = plan[3]
    assert force_step in {Step.FORCE_GROUP, Step.FORCE_JOB, Step.FORCE_TASKKILL}
    if platform == "win32":
        assert force_step is (Step.FORCE_JOB if job_assigned else Step.FORCE_TASKKILL)
    else:
        assert force_step is Step.FORCE_GROUP

    names = [call[0] for call in tree.calls]
    # Stdin close is first, and the OS calls follow the expected order exactly.
    expected_names = ["close_stdin"]
    expected_names += ["graceful"] if want_graceful else []
    expected_names += ["force"] if want_force else []
    expected_names += ["close_streams", "release"]
    assert names == expected_names

    # Timing: graceful `stdin_wait` after stdin close; force `grace` after graceful.
    start = tree.calls[0][1]
    assert start == _START
    if want_graceful:
        (_, graceful_at, empty_at_graceful, _) = _only(tree.calls, "graceful")[0]
        assert stdin_wait <= graceful_at - start <= stdin_wait + _POLL_SLACK_S
        assert not empty_at_graceful
        if want_force:
            (_, force_at, empty_at_force, step) = _only(tree.calls, "force")[0]
            assert grace <= force_at - graceful_at <= grace + _POLL_SLACK_S
            assert not empty_at_force
            assert step is force_step

    # Report: steps that ran, whether force ran, and no survivors.
    expected_steps = [Step.CLOSE_STDIN]
    expected_steps += [Step.GRACEFUL, Step.WAIT_GRACE] if want_graceful else []
    expected_steps += [force_step] if want_force else []
    expected_steps += [Step.REAP]
    assert report.steps == tuple(expected_steps)
    assert report.forced is want_force
    assert report.survivors == ()
    assert report.pid == 100
    assert "survivors" not in names
    assert tree.calls[-1][2], "the tree must be empty when the last call runs"

    # The stop is logged once, as info, with no survivors.
    stopped = [r for r in log.records if r[1] == "mcp_process_tree_stopped"]
    assert len(stopped) == 1
    assert stopped[0][0] == "info"
    assert stopped[0][2]["survivors"] == []
    assert stopped[0][2]["forced"] is want_force

    # A failed graceful request is logged and does not stop the schedule.
    failures = [r for r in log.records if r[0] == "warning"]
    if want_graceful and not graceful_ok:
        assert len(failures) == 1
        assert failures[0][1] == (
            "ctrl_break_failed" if platform == "win32" else "mcp_process_signal_failed"
        )
    else:
        assert failures == []
