"""Shared pytest configuration: Hypothesis profiles, injectable fake time, platform markers.

Time-dependent code (backoff, retries, approval timeout, retention, tool latency) takes an
injected clock and sleep function. The fakes here let tests run those paths instantly and
deterministically.
"""

import os
import sys

import pytest
from hypothesis import HealthCheck, settings

# --- Hypothesis profiles ---------------------------------------------------------------------
# `ci` is the profile CI selects with HYPOTHESIS_PROFILE=ci. `dev` is the local default and still
# meets the design's floor of 100 examples per property.
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# --- Platform markers ------------------------------------------------------------------------
# Pure-core tests run on every runner and take `platform` as a parameter. Tests that need the
# real OS use these markers.
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific OS behaviour")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific OS behaviour")
macos_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific OS behaviour")
linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux-specific OS behaviour"
)


# --- Injectable fake time --------------------------------------------------------------------
class FakeClock:
    """A manually advanced clock.

    `now_ms()` is wall-clock epoch milliseconds (stored timestamps, Last_Activity). `monotonic()`
    is seconds on a clock that never goes backwards (timeouts and backoff). `advance` moves both
    forward together. `set_wall_ms` jumps only the wall clock, to model a wall-clock change.
    """

    def __init__(
        self, *, start_ms: int = 1_700_000_000_000, start_monotonic: float = 1_000.0
    ) -> None:
        self._wall_ms = start_ms
        self._monotonic = start_monotonic

    def now_ms(self) -> int:
        return self._wall_ms

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("advance() takes a non-negative number of seconds")
        self._monotonic += seconds
        self._wall_ms += round(seconds * 1000)

    def set_wall_ms(self, wall_ms: int) -> None:
        self._wall_ms = wall_ms


class FakeSleep:
    """An awaitable `sleep(seconds)` that advances a `FakeClock` instead of waiting.

    It records every requested delay in `calls`, so tests can assert backoff schedules.
    """

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self._clock.advance(max(seconds, 0.0))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleep(clock: FakeClock) -> FakeSleep:
    return FakeSleep(clock)
