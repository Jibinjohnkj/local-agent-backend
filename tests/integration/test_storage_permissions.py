"""Per-OS access rules on the Storage_Directory of a real Backend process (task 22.18).

* `test_storage_modes_posix` (`posix_only`; 16.8, 17.36): under `umask 000` a fresh start gives the
  directory mode `0700` and the database, `-wal`, `-shm`, and lock files mode `0600`. A restart over
  a directory and files that were loosened to `0755`/`0644` (and `0777`/`0666`) tightens them
  again, so the result does not depend on the umask or on what was there before.
* `test_storage_dacl_windows` (`windows_only`; 16.8, 17.37): the directory has a protected DACL with
  exactly one allow entry for the current user SID, marked `OI|CI`. The database and lock files
  carry a protected, non-inheritable entry. The `-wal` and `-shm` files that SQLite creates later
  inherit exactly one entry for the user. A pre-existing directory that inherited `Users` entries
  is cleaned up the same way. Windows-only; verified here through collection, ruff, and
  `mypy --platform win32`.
* `test_storage_restrict_failure` (16.9, 18.28): when the OS refuses to restrict the directory or
  the database, the Backend exits with status 2 and prints a report that names `storage.path`, the
  path, and the OS reason. No MCP_Server starts and the port stays unbound. The OS call is made to
  fail in the child through a `sitecustomize.py` on `PYTHONPATH` (`os.chmod` on POSIX,
  `SetNamedSecurityInfoW` on Windows), so the Backend has no hook for it.
"""

from __future__ import annotations

import errno
import os
import socket
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Final

import pytest

from integration_harness import (
    DATABASE_NAME,
    BackendLauncher,
    BackendProcess,
    echo_server_entry,
)
from local_agent.oscompat.permissions import current_user_sid, describe_access
from local_agent.storage.lock import LOCK_FILE_NAME

pytestmark = pytest.mark.integration

# Same definitions as `posix_only` / `windows_only` in tests/conftest.py, which this directory's
# own conftest.py shadows under the module name `conftest`.
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific OS behaviour")
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific OS behaviour")

EXIT_CONFIG: Final = 2
EXIT_TIMEOUT_S: Final = 30.0
WAL_NAME: Final = f"{DATABASE_NAME}-wal"
SHM_NAME: Final = f"{DATABASE_NAME}-shm"
SIDECAR_NAMES: Final = (WAL_NAME, SHM_NAME)
FILE_NAMES: Final = (DATABASE_NAME, WAL_NAME, SHM_NAME, LOCK_FILE_NAME)

DIR_MODE: Final = 0o700
FILE_MODE: Final = 0o600

# Windows access-control constants (winnt.h).
ACCESS_ALLOWED_ACE_TYPE: Final = 0x00
OBJECT_INHERIT_ACE: Final = 0x01
CONTAINER_INHERIT_ACE: Final = 0x02
INHERIT_FLAGS: Final = OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE
# S-1-5-32-545, BUILTIN\Users: revision 1, two sub-authorities, authority 5, then 32 and 545.
USERS_SID: Final = (
    b"\x01\x02\x00\x00\x00\x00\x00\x05" + (32).to_bytes(4, "little") + (545).to_bytes(4, "little")
)

# The child replaces one OS call so that it fails for a path that ends with `LA_FAIL_SUFFIX`.
FAIL_SUFFIX_ENV: Final = "LOCAL_AGENT_IT_FAIL_SUFFIX"
SITECUSTOMIZE_POSIX: Final = f'''\
"""Test-only: make `os.chmod` fail with EPERM for one path (task 22.18)."""

import errno
import os

_suffix = os.environ["{FAIL_SUFFIX_ENV}"]
_real_chmod = os.chmod


def _chmod(path, *args, **kwargs):
    if os.fspath(path).replace("\\\\", "/").endswith(_suffix):
        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
    return _real_chmod(path, *args, **kwargs)


os.chmod = _chmod
'''
SITECUSTOMIZE_WINDOWS: Final = f'''\
"""Test-only: make `SetNamedSecurityInfoW` return ERROR_ACCESS_DENIED for one path (task 22.18)."""

import os

from local_agent.oscompat import _win32

_suffix = os.environ["{FAIL_SUFFIX_ENV}"]
_real = _win32.SetNamedSecurityInfoW
_ERROR_ACCESS_DENIED = 5


def _set_named_security_info(path, *args):
    if str(path).replace("\\\\", "/").endswith(_suffix):
        return _ERROR_ACCESS_DENIED
    return _real(path, *args)


_win32.SetNamedSecurityInfoW = _set_named_security_info
'''


# --- Helpers ------------------------------------------------------------------------------------


@pytest.fixture
def open_umask() -> Iterator[None]:
    """Run the test under `umask 000`; the Backend child inherits it."""
    previous = os.umask(0)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _mode_report(storage_dir: Path) -> dict[str, str]:
    """The octal mode of the directory and of each storage file; a missing file is `missing`."""
    report = {".": oct(_mode(storage_dir))}
    for name in FILE_NAMES:
        path = storage_dir / name
        report[name] = oct(_mode(path)) if path.exists() else "missing"
    return report


def _assert_user_only_modes(storage_dir: Path) -> None:
    expected = {".": oct(DIR_MODE), **{name: oct(FILE_MODE) for name in FILE_NAMES}}
    assert _mode_report(storage_dir) == expected


def _port_refuses(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(2.0)
        return sock.connect_ex(("127.0.0.1", port)) != 0


# --- POSIX --------------------------------------------------------------------------------------


@posix_only
@pytest.mark.parametrize(
    ("loose_dir_mode", "loose_file_mode"),
    [(0o755, 0o644), (0o777, 0o666)],
    ids=["755-644", "777-666"],
)
def test_storage_modes_posix(
    backend_launcher: BackendLauncher,
    open_umask: None,
    loose_dir_mode: int,
    loose_file_mode: int,
) -> None:
    storage_dir = backend_launcher.storage_dir

    # A fresh start under umask 000: the Backend creates the directory and every file.
    first = backend_launcher.launch()
    assert _mode_report(storage_dir)[WAL_NAME] != "missing", "no -wal file while the Backend runs"
    _assert_user_only_modes(storage_dir)
    assert first.stop() == 0

    # Loosen everything, including empty sidecars that are already there at the next start.
    for name in SIDECAR_NAMES:
        if not (storage_dir / name).exists():
            (storage_dir / name).write_bytes(b"")
    for name in FILE_NAMES:
        (storage_dir / name).chmod(loose_file_mode)
    storage_dir.chmod(loose_dir_mode)
    assert _mode(storage_dir) == loose_dir_mode

    second = backend_launcher.launch()
    _assert_user_only_modes(storage_dir)
    assert second.stop() == 0


# --- Windows ------------------------------------------------------------------------------------


def _assert_one_user_ace(path: Path, *, protected: bool, inheritable: bool | None) -> None:
    """`path` holds exactly one allow entry, for the current user; check protection and `OI|CI`."""
    report = describe_access(str(path), "win32")
    assert report.dacl_present, f"{path} has a NULL DACL"
    if protected:
        assert report.protected, f"{path} DACL is not protected"
    assert len(report.aces) == 1, f"{path} has {len(report.aces)} entries: {report.aces}"
    ace = report.aces[0]
    assert ace.ace_type == ACCESS_ALLOWED_ACE_TYPE
    assert ace.sid == current_user_sid(), f"{path} entry is not for the current user"
    if inheritable is True:
        assert ace.flags & INHERIT_FLAGS == INHERIT_FLAGS, f"{path} entry is not OI|CI"
    elif inheritable is False:
        assert ace.flags & INHERIT_FLAGS == 0, f"{path} entry is inheritable"


def _assert_dacls(storage_dir: Path) -> None:
    _assert_one_user_ace(storage_dir, protected=True, inheritable=True)
    for name in (DATABASE_NAME, LOCK_FILE_NAME):
        _assert_one_user_ace(storage_dir / name, protected=True, inheritable=False)
    # SQLite creates these after the Backend restricted the directory; they inherit its entry.
    for name in SIDECAR_NAMES:
        assert (storage_dir / name).exists(), f"no {name} while the Backend runs"
        _assert_one_user_ace(storage_dir / name, protected=False, inheritable=None)


def _share_with_users(parent: Path) -> None:
    """Give `BUILTIN\\Users` an inheritable entry on `parent`, so new children inherit it."""
    subprocess.run(
        ["icacls", str(parent), "/grant", "*S-1-5-32-545:(OI)(CI)M"],
        check=True,
        capture_output=True,
    )


@windows_only
@pytest.mark.parametrize("preexisting", [False, True], ids=["fresh", "inherited-users"])
def test_storage_dacl_windows(
    backend_launcher: BackendLauncher, tmp_path: Path, preexisting: bool
) -> None:
    storage_dir = backend_launcher.storage_dir
    if preexisting:
        parent = tmp_path / "shared"
        parent.mkdir()
        _share_with_users(parent)
        storage_dir = parent / "storage"
        storage_dir.mkdir()
        (storage_dir / DATABASE_NAME).write_bytes(b"")
        for path in (storage_dir, storage_dir / DATABASE_NAME):
            inherited = describe_access(str(path), "win32")
            assert any(ace.sid == USERS_SID for ace in inherited.aces), (
                f"setup: {path} did not inherit a Users entry"
            )

    backend = backend_launcher.launch(storage_dir=storage_dir)
    _assert_dacls(storage_dir)
    assert backend.stop() == 0


# --- A refused restriction ----------------------------------------------------------------------


def _failure_hook(tmp_path: Path, suffix: str) -> dict[str, str]:
    """Environment for a child whose OS permission call fails for paths ending with `suffix`."""
    hook_dir = tmp_path / "restrict-hook"
    hook_dir.mkdir(exist_ok=True)
    source = SITECUSTOMIZE_WINDOWS if sys.platform == "win32" else SITECUSTOMIZE_POSIX
    (hook_dir / "sitecustomize.py").write_text(source, encoding="utf-8")
    inherited = os.environ.get("PYTHONPATH")
    python_path = os.pathsep.join([str(hook_dir), *([inherited] if inherited else [])])
    return {"PYTHONPATH": python_path, FAIL_SUFFIX_ENV: suffix}


def _os_reason() -> str:
    """The text that the failing OS call reports, as the Backend prints it."""
    return "Access is denied" if sys.platform == "win32" else os.strerror(errno.EPERM)


def _run_to_exit(process: BackendProcess) -> int:
    try:
        return process.wait(EXIT_TIMEOUT_S)
    finally:
        process.close()


@pytest.mark.parametrize("target", ["directory", "database"])
def test_storage_restrict_failure(
    backend_launcher: BackendLauncher, tmp_path: Path, target: str
) -> None:
    storage_dir = tmp_path / "perm-storage"
    failing_path = storage_dir if target == "directory" else storage_dir / DATABASE_NAME
    echo_log = tmp_path / "echo-requests.jsonl"

    process = backend_launcher.launch(
        storage_dir=storage_dir,
        mcp_servers=[echo_server_entry(request_log=echo_log)],
        env=_failure_hook(tmp_path, "/" + failing_path.name),
        wait=False,
    )

    assert _run_to_exit(process) == EXIT_CONFIG, process.stderr_text()
    report = process.stderr_text()
    assert "invalid configuration" in report
    assert "storage.path" in report
    assert str(failing_path) in report
    assert _os_reason().lower() in report.lower()
    assert not echo_log.exists(), "an MCP_Server was started after a storage failure"
    assert _port_refuses(process.port), "the Backend bound its port after a storage failure"
    if target == "directory":
        # The directory is restricted before anything else is created in it.
        assert not (storage_dir / DATABASE_NAME).exists()
        assert not (storage_dir / LOCK_FILE_NAME).exists()
