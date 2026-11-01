"""Unit tests for storage permissions (Requirements 16.8, 16.9, 17.36, 17.37).

POSIX tests run on macOS and Linux and check the real mode bits. Windows tests check the real DACL
and run on Windows only. The failure tests run on both: they replace `os.chmod` (POSIX) or
`_win32.SetNamedSecurityInfoW` (Windows) so the call fails and check the `PermissionSetupError`.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from conftest import posix_only, windows_only
from local_agent.oscompat.permissions import (
    STORAGE_FILES,
    PermissionSetupError,
    describe_access,
    restrict_dir,
    restrict_file,
    restrict_storage,
)

PLATFORM = sys.platform

_BROAD_DIR_MODE = 0o755
_BROAD_FILE_MODE = 0o644


@pytest.fixture
def open_umask() -> Iterator[None]:
    """Run the test under `umask 000`, so a mode set by the code cannot be narrowed by the umask."""
    previous = os.umask(0)
    try:
        yield
    finally:
        os.umask(previous)


def _make_storage(root: Path) -> tuple[Path, list[Path]]:
    """A `0755` directory holding every storage file, each created with mode `0644`."""
    directory = root / "storage"
    directory.mkdir()
    os.chmod(directory, _BROAD_DIR_MODE)
    files: list[Path] = []
    for name in STORAGE_FILES:
        file = directory / name
        file.write_bytes(b"data")
        os.chmod(file, _BROAD_FILE_MODE)
        files.append(file)
    return directory, files


# --- POSIX: modes ----------------------------------------------------------------------------


@posix_only
@pytest.mark.usefixtures("open_umask")
class TestPosixModes:
    def test_restrict_dir_narrows_an_existing_0755_directory_to_0700(self, tmp_path: Path) -> None:
        directory, _ = _make_storage(tmp_path)
        assert stat.S_IMODE(directory.stat().st_mode) == _BROAD_DIR_MODE

        restrict_dir(str(directory), PLATFORM)

        assert stat.S_IMODE(directory.stat().st_mode) == 0o700

    def test_restrict_file_narrows_an_existing_0644_file_to_0600(self, tmp_path: Path) -> None:
        _, files = _make_storage(tmp_path)
        assert stat.S_IMODE(files[0].stat().st_mode) == _BROAD_FILE_MODE

        restrict_file(str(files[0]), PLATFORM)

        assert stat.S_IMODE(files[0].stat().st_mode) == 0o600

    def test_restrict_storage_fixes_the_directory_and_every_existing_file(
        self, tmp_path: Path
    ) -> None:
        directory, files = _make_storage(tmp_path)

        restrict_storage(str(directory), PLATFORM)

        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        for file in files:
            assert stat.S_IMODE(file.stat().st_mode) == 0o600, file.name

    def test_restrict_storage_skips_files_that_do_not_exist(self, tmp_path: Path) -> None:
        directory = tmp_path / "storage"
        directory.mkdir()
        os.chmod(directory, _BROAD_DIR_MODE)
        database = directory / STORAGE_FILES[0]
        database.write_bytes(b"data")
        os.chmod(database, _BROAD_FILE_MODE)

        restrict_storage(str(directory), PLATFORM)

        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(database.stat().st_mode) == 0o600
        assert sorted(entry.name for entry in directory.iterdir()) == [STORAGE_FILES[0]]

    def test_describe_access_reports_the_mode_bits(self, tmp_path: Path) -> None:
        directory, files = _make_storage(tmp_path)

        before = describe_access(str(directory), PLATFORM)
        restrict_storage(str(directory), PLATFORM)
        after_dir = describe_access(str(directory), PLATFORM)
        after_file = describe_access(str(files[0]), PLATFORM)

        assert before.mode == _BROAD_DIR_MODE
        assert after_dir.mode == 0o700
        assert after_file.mode == 0o600
        assert after_dir.aces == ()


# --- Windows: DACLs --------------------------------------------------------------------------

if sys.platform == "win32":
    from local_agent.oscompat import _win32
    from local_agent.oscompat.permissions import AccessReport, current_user_sid

    _INHERITED_ACE = 0x10

    def _assert_single_user_ace(report: AccessReport, *, flags: int | None) -> None:
        """The DACL is protected and holds one allow entry for the current user."""
        assert report.dacl_present
        assert report.protected
        assert len(report.aces) == 1
        ace = report.aces[0]
        assert ace.ace_type == _win32.ACCESS_ALLOWED_ACE_TYPE
        assert ace.sid == current_user_sid()
        assert ace.mask & _win32.FILE_ALL_ACCESS == _win32.FILE_ALL_ACCESS
        if flags is not None:
            assert ace.flags == flags

    @windows_only
    class TestWindowsDacl:
        def test_directory_has_a_protected_dacl_with_one_inheritable_user_entry(
            self, tmp_path: Path
        ) -> None:
            directory, _ = _make_storage(tmp_path)

            restrict_dir(str(directory), PLATFORM)

            _assert_single_user_ace(
                describe_access(str(directory), PLATFORM),
                flags=_win32.OBJECT_INHERIT_ACE | _win32.CONTAINER_INHERIT_ACE,
            )

        def test_file_has_a_protected_dacl_with_one_user_entry_and_no_inheritance_flags(
            self, tmp_path: Path
        ) -> None:
            _, files = _make_storage(tmp_path)

            restrict_file(str(files[0]), PLATFORM)

            _assert_single_user_ace(describe_access(str(files[0]), PLATFORM), flags=0)

        def test_restrict_storage_covers_the_directory_and_every_existing_file(
            self, tmp_path: Path
        ) -> None:
            directory, files = _make_storage(tmp_path)

            restrict_storage(str(directory), PLATFORM)

            _assert_single_user_ace(
                describe_access(str(directory), PLATFORM),
                flags=_win32.OBJECT_INHERIT_ACE | _win32.CONTAINER_INHERIT_ACE,
            )
            for file in files:
                _assert_single_user_ace(describe_access(str(file), PLATFORM), flags=0)

        def test_file_created_later_inherits_the_directory_entry(self, tmp_path: Path) -> None:
            directory, _ = _make_storage(tmp_path)
            restrict_dir(str(directory), PLATFORM)

            later = directory / "local-agent.db-wal-created-later"
            later.write_bytes(b"data")

            report = describe_access(str(later), PLATFORM)
            assert report.dacl_present
            assert not report.protected
            assert len(report.aces) == 1
            ace = report.aces[0]
            assert ace.ace_type == _win32.ACCESS_ALLOWED_ACE_TYPE
            assert ace.sid == current_user_sid()
            assert ace.mask & _win32.FILE_ALL_ACCESS == _win32.FILE_ALL_ACCESS
            assert ace.flags & _INHERITED_ACE

        def test_current_user_sid_is_a_well_formed_sid(self) -> None:
            sid = current_user_sid()

            assert sid[0] == 1  # SID revision
            assert len(sid) == 8 + 4 * sid[1]  # header plus one DWORD per sub-authority


# --- Both platforms: injected failure --------------------------------------------------------


def _inject_failure(monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the OS call that sets permissions fail. Returns the reason the error must carry."""
    if sys.platform == "win32":
        from local_agent.oscompat import _win32
        from local_agent.oscompat.permissions import _win_error

        access_denied = 5  # ERROR_ACCESS_DENIED

        def fail_set_security(*args: object) -> int:
            return access_denied

        monkeypatch.setattr(_win32, "SetNamedSecurityInfoW", fail_set_security)
        reason = _win_error(access_denied).strerror
        assert reason
        return reason

    reason = os.strerror(errno.EACCES)

    def fail_chmod(path: object, mode: int, *args: object, **kwargs: object) -> None:
        raise OSError(errno.EACCES, reason)

    monkeypatch.setattr(os, "chmod", fail_chmod)
    return reason


_RESTRICT_CALLS: list[tuple[str, Callable[[str, str], None]]] = [
    ("restrict_dir", restrict_dir),
    ("restrict_file", restrict_file),
    ("restrict_storage", restrict_storage),
]


@pytest.mark.parametrize(("name", "restrict"), _RESTRICT_CALLS, ids=[c[0] for c in _RESTRICT_CALLS])
def test_failure_raises_permission_setup_error_naming_path_and_os_reason(
    name: str,
    restrict: Callable[[str, str], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "storage"
    target.mkdir()
    if name == "restrict_file":
        target = target / STORAGE_FILES[0]
        target.write_bytes(b"data")
    reason = _inject_failure(monkeypatch)

    with pytest.raises(PermissionSetupError) as caught:
        restrict(str(target), PLATFORM)

    error = caught.value
    assert error.path == str(target)
    assert error.os_reason == reason
    assert str(target) in str(error)
    assert reason in str(error)
