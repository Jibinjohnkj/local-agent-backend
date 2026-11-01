"""Storage_Database plumbing: open, migrate, lock, recover, and retain.

This module owns the two storage exceptions. `storage/database.py` and `session/store.py` re-export
them, so every layer catches the same classes. It imports nothing from the rest of the package,
which lets `oscompat/paths.py` raise `StorageError` without an import cycle.
"""

from __future__ import annotations

__all__ = ["StaleState", "StorageError"]


class StorageError(Exception):
    """A storage operation failed (18.24, 18.30).

    `path` is the file or directory involved. `reason` is the OS or SQLite reason and must already
    be redacted by the caller (16.11), because it is logged and shown as is.
    """

    path: str
    reason: str

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(path, reason)
        self.path = path
        self.reason = reason

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


class StaleState(Exception):
    """A compare-and-set on a Task_State lost: the Task was not in an expected state (18.30).

    The transaction has been rolled back. Raised when `StateChange.expected_from` does not match
    the committed state, so two writers can never interleave a transition.
    """
