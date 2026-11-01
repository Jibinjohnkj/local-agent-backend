"""Unit tests for the Storage_Lock (Requirements 18.25, 18.26, 18.27, 17.36).

The lock is an OS lock on `<storage_dir>/local-agent.lock`. The cross-process tests start a child
interpreter with `sys.executable`, because the OS lock is what is under test and only a second
process shows that it holds. Child processes import `local_agent` from the repo's `src` directory.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from conftest import posix_only
from local_agent.storage import StorageError
from local_agent.storage.lock import LOCK_FILE_NAME, StorageInUse, StorageLock

PLATFORM = sys.platform
SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")

# Holds the lock, prints READY, then blocks until the parent closes stdin or kills it.
_HOLDER_CODE = """
import sys
from pathlib import Path
from local_agent.storage.lock import StorageLock
lock = StorageLock.acquire(Path(sys.argv[1]), sys.platform)
print("READY", flush=True)
sys.stdin.read()
"""

# Tries to take the lock and prints how long the attempt took in seconds.
_CONTENDER_CODE = """
import sys
import time
from pathlib import Path
from local_agent.storage.lock import StorageInUse, StorageLock
start = time.monotonic()
try:
    StorageLock.acquire(Path(sys.argv[1]), sys.platform)
except StorageInUse as exc:
    print("IN_USE", time.monotonic() - start, exc.path, flush=True)
    sys.exit(3)
print("ACQUIRED", time.monotonic() - start, flush=True)
"""


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = SRC_DIR if not existing else os.pathsep.join([SRC_DIR, existing])
    return env


def _try_in_child(storage_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CONTENDER_CODE, str(storage_dir)],
        capture_output=True,
        text=True,
        env=_child_env(),
        timeout=60,
        check=False,
    )


def _start_holder(storage_dir: Path) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_CODE, str(storage_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if line.strip() != "READY":
        proc.kill()
        _, err = proc.communicate()
        pytest.fail(f"holder child did not acquire the lock: {line!r} {err}")
    return proc


HolderFactory = Callable[[Path], "subprocess.Popen[str]"]


@pytest.fixture
def holder_factory() -> Iterator[HolderFactory]:
    """Yield a function that starts a lock-holding child, and kill every child at teardown."""
    started: list[subprocess.Popen[str]] = []

    def start(storage_dir: Path) -> subprocess.Popen[str]:
        proc = _start_holder(storage_dir)
        started.append(proc)
        return proc

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


# --- acquire / release -----------------------------------------------------------------------


def test_acquire_creates_the_lock_file_in_the_storage_dir(tmp_path: Path) -> None:
    lock = StorageLock.acquire(tmp_path, PLATFORM)
    try:
        assert lock.path == str(tmp_path / LOCK_FILE_NAME)
        assert (tmp_path / "local-agent.lock").is_file()
    finally:
        lock.release()


def test_acquire_release_acquire_succeeds(tmp_path: Path) -> None:
    first = StorageLock.acquire(tmp_path, PLATFORM)
    first.release()

    second = StorageLock.acquire(tmp_path, PLATFORM)
    second.release()


def test_release_is_idempotent(tmp_path: Path) -> None:
    lock = StorageLock.acquire(tmp_path, PLATFORM)
    lock.release()
    lock.release()

    StorageLock.acquire(tmp_path, PLATFORM).release()


def test_acquire_does_not_wait_when_the_lock_is_held_in_process(tmp_path: Path) -> None:
    first = StorageLock.acquire(tmp_path, PLATFORM)
    try:
        start = time.monotonic()
        with pytest.raises(StorageInUse) as excinfo:
            StorageLock.acquire(tmp_path, PLATFORM)
        elapsed = time.monotonic() - start
    finally:
        first.release()

    assert elapsed < 1.0
    assert excinfo.value.path == str(tmp_path / LOCK_FILE_NAME)


def test_storage_in_use_message_names_the_lock_file() -> None:
    error = StorageInUse("/some/dir/local-agent.lock")

    assert error.path == "/some/dir/local-agent.lock"
    assert "/some/dir/local-agent.lock" in str(error)


def test_acquire_with_the_other_platform_raises_storage_error(tmp_path: Path) -> None:
    other = "linux" if sys.platform == "win32" else "win32"

    with pytest.raises(StorageError):
        StorageLock.acquire(tmp_path, other)

    assert not (tmp_path / LOCK_FILE_NAME).exists()


def test_acquire_in_a_missing_directory_raises_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        StorageLock.acquire(tmp_path / "does-not-exist", PLATFORM)


# --- two processes ---------------------------------------------------------------------------


def test_second_process_gets_storage_in_use_without_waiting(tmp_path: Path) -> None:
    lock = StorageLock.acquire(tmp_path, PLATFORM)
    try:
        started = time.monotonic()
        result = _try_in_child(tmp_path)
        total = time.monotonic() - started
    finally:
        lock.release()

    assert result.returncode == 3, result.stderr
    tag, attempt_seconds, path = result.stdout.split()
    assert tag == "IN_USE"
    assert path == str(tmp_path / LOCK_FILE_NAME)
    # The `acquire` call itself returns at once. `total` includes interpreter start-up, so it only
    # has to show that nothing blocked for a long time.
    assert float(attempt_seconds) < 1.0
    assert total < 30.0


def test_second_process_acquires_after_the_first_releases(tmp_path: Path) -> None:
    StorageLock.acquire(tmp_path, PLATFORM).release()

    result = _try_in_child(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[0] == "ACQUIRED"


def test_lock_held_by_a_child_blocks_this_process(
    tmp_path: Path, holder_factory: HolderFactory
) -> None:
    holder = holder_factory(tmp_path)
    assert holder.poll() is None

    began = time.monotonic()
    with pytest.raises(StorageInUse):
        StorageLock.acquire(tmp_path, PLATFORM)

    assert time.monotonic() - began < 1.0


def test_lock_is_dropped_when_the_holder_is_killed(
    tmp_path: Path, holder_factory: HolderFactory
) -> None:
    """The OS drops the lock when the process dies, so no stale-lock cleanup is needed (18.27)."""
    holder = holder_factory(tmp_path)
    with pytest.raises(StorageInUse):
        StorageLock.acquire(tmp_path, PLATFORM)

    holder.kill()
    holder.wait(timeout=30)

    # The lock file is still on disk, and a new Backend takes the lock anyway.
    assert (tmp_path / LOCK_FILE_NAME).exists()
    StorageLock.acquire(tmp_path, PLATFORM).release()


# --- file contents are never read ------------------------------------------------------------


@pytest.mark.parametrize(
    "contents",
    [
        b"\xff\xfe\x00 not a pid \x80\x81",
        b"not-a-number\n",
        b"999999999\n",
        b"",
        b"x" * 100_000,
    ],
    ids=["binary-garbage", "text-garbage", "dead-pid", "empty", "large"],
)
def test_garbage_in_the_lock_file_does_not_stop_the_lock(tmp_path: Path, contents: bytes) -> None:
    (tmp_path / LOCK_FILE_NAME).write_bytes(contents)

    lock = StorageLock.acquire(tmp_path, PLATFORM)
    lock.release()


def test_a_live_pid_in_the_lock_file_does_not_stop_the_lock(tmp_path: Path) -> None:
    """A file naming a running process is not a lock. Only the OS lock decides (18.26)."""
    (tmp_path / LOCK_FILE_NAME).write_text(f"{os.getpid()}\n", encoding="ascii")

    lock = StorageLock.acquire(tmp_path, PLATFORM)
    lock.release()


def test_acquire_writes_the_pid_for_diagnostics(tmp_path: Path) -> None:
    (tmp_path / LOCK_FILE_NAME).write_bytes(b"old garbage that is longer than a pid\n")

    lock = StorageLock.acquire(tmp_path, PLATFORM)
    try:
        text = (tmp_path / LOCK_FILE_NAME).read_text(encoding="ascii")
    finally:
        lock.release()

    assert text == f"{os.getpid()}\n"


# --- permissions (POSIX) ---------------------------------------------------------------------


@pytest.fixture
def open_umask() -> Iterator[None]:
    """Run under `umask 000`, so a mode set by the code cannot be narrowed by the umask."""
    previous = os.umask(0)
    try:
        yield
    finally:
        os.umask(previous)


@posix_only
@pytest.mark.usefixtures("open_umask")
def test_new_lock_file_is_0600(tmp_path: Path) -> None:
    lock = StorageLock.acquire(tmp_path, PLATFORM)
    try:
        mode = stat.S_IMODE((tmp_path / LOCK_FILE_NAME).stat().st_mode)
    finally:
        lock.release()

    assert mode == 0o600


@posix_only
@pytest.mark.usefixtures("open_umask")
def test_existing_broad_lock_file_is_narrowed_to_0600(tmp_path: Path) -> None:
    path = tmp_path / LOCK_FILE_NAME
    path.write_bytes(b"")
    path.chmod(0o644)

    lock = StorageLock.acquire(tmp_path, PLATFORM)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    finally:
        lock.release()

    assert mode == 0o600


@posix_only
def test_lock_file_stays_0600_after_release(tmp_path: Path) -> None:
    StorageLock.acquire(tmp_path, PLATFORM).release()

    assert stat.S_IMODE((tmp_path / LOCK_FILE_NAME).stat().st_mode) == 0o600
