"""Storage_Lock: one Backend per Storage_Directory (18.25-18.27).

The lock is an OS lock on a separate file, `<storage_dir>/local-agent.lock`. It is never taken on
the database file, because that would interfere with SQLite's own locking. `StorageLock.acquire`
runs before the database is opened, so a second Backend never touches a database that another
process is using.

POSIX takes `fcntl.flock(fd, LOCK_EX | LOCK_NB)`. Windows takes `msvcrt.locking(fd, LK_NBLCK, 1)`
on byte 0. Neither call waits: a held lock raises `StorageInUse` at once (18.26). The descriptor
stays open for the life of the process, and the OS drops the lock when the process ends, including
after `SIGKILL` or `TerminateProcess`. No stale-lock cleanup exists or is needed (18.27).

After locking, the Backend writes its PID into the file for people to read. Nothing reads it back,
so a lock file with garbage contents still locks.

`platform` takes `sys.platform` values, like `oscompat/permissions.py`: `"win32"` selects the
Windows implementation and every other value selects the POSIX one. The platform-specific calls sit
behind `sys.platform == "win32"` checks, so mypy and ruff see only the code that matches the host.
Asking for the other platform's implementation raises `StorageError`.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Final, Self

from local_agent.oscompat.permissions import PermissionSetupError, restrict_file
from local_agent.storage import StorageError

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

__all__ = ["LOCK_FILE_NAME", "StorageInUse", "StorageLock"]

LOCK_FILE_NAME: Final = "local-agent.lock"

_FILE_MODE: Final = 0o600


class StorageInUse(Exception):
    """Another Backend process holds the Storage_Lock (18.26).

    `path` is the lock file. The caller reports the database path, which lives next to it.
    """

    path: str

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path

    def __str__(self) -> str:
        return f"storage is in use: {self.path}"


# --- Platform calls ------------------------------------------------------------------------------

if sys.platform == "win32":

    def _try_lock(fd: int) -> bool:
        """Lock byte 0 without waiting. Any `OSError` means another process holds it."""
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

    def _restrict_new_file(fd: int, path: str, *, existed: bool) -> None:
        # A new file inherits the directory's protected DACL. A file that was already there may
        # carry another ACL, so the DACL is applied again explicitly (17.37).
        if existed:
            restrict_file(path, "win32")

else:

    def _try_lock(fd: int) -> bool:
        """Take the exclusive flock without waiting. `BlockingIOError` means it is held."""
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

    def _restrict_new_file(fd: int, path: str, *, existed: bool) -> None:
        # `os.open` mode is filtered by the umask and ignored for an existing file, so set 0600
        # on the descriptor itself (17.36).
        del existed
        try:
            os.fchmod(fd, _FILE_MODE)
        except OSError as exc:
            raise PermissionSetupError(path, exc.strerror or type(exc).__name__) from exc


def _write_pid(fd: int) -> None:
    """Record the PID for diagnostics. A failure here never matters, so it is swallowed."""
    with contextlib.suppress(OSError):
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))


class StorageLock:
    """A held Storage_Lock. Build one with `StorageLock.acquire`."""

    __slots__ = ("_fd", "_path")

    def __init__(self, fd: int, path: str) -> None:
        self._fd: int | None = fd
        self._path = path

    @property
    def path(self) -> str:
        """The lock file path."""
        return self._path

    @classmethod
    def acquire(cls, storage_dir: Path, platform: str) -> Self:
        """Lock `<storage_dir>/local-agent.lock` or raise. Never waits (18.26).

        Raises `StorageInUse` when another process holds the lock, `PermissionSetupError` when the
        file cannot be restricted to the current user, and `StorageError` when `platform` does not
        match the host or the file cannot be opened.
        """
        path = str(storage_dir / LOCK_FILE_NAME)
        if (platform == "win32") != (sys.platform == "win32"):
            raise StorageError(path, f"the {platform} lock needs a matching host")

        existed = os.path.exists(path)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(path, flags, _FILE_MODE)
        except OSError as exc:
            raise StorageError(path, exc.strerror or type(exc).__name__) from exc

        try:
            _restrict_new_file(fd, path, existed=existed)
            if not _try_lock(fd):
                raise StorageInUse(path)
        except BaseException:
            os.close(fd)
            raise
        _write_pid(fd)
        return cls(fd, path)

    def release(self) -> None:
        """Unlock and close the file. Safe to call twice; process exit does the same (18.27)."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        with contextlib.suppress(OSError):
            _unlock(fd)
        with contextlib.suppress(OSError):
            os.close(fd)
