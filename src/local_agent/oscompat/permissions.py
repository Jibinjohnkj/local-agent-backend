"""User-only access for the Storage_Directory and its files (Requirements 16.8, 16.9, 17.36, 17.37).

On macOS and Linux the directory gets mode `0700` and each file `0600`. On Windows each path gets a
protected DACL (inherited entries removed) with exactly one allow entry that grants full control to
the SID of the user running the Backend. The directory entry is inheritable, so the `-wal` and
`-shm` files SQLite creates later receive the same restriction.

`platform` takes `sys.platform` values: `"win32"` selects the Windows implementation and every
other value selects the POSIX one. The Windows code calls the `ctypes` bindings in `_win32.py`
through the module (`_win32.SetNamedSecurityInfoW`), so a test can replace one binding to inject a
failure. Those calls sit behind `sys.platform == "win32"` checks, which is how mypy and ruff see
only the code that matches the host. Asking for `"win32"` on another host raises
`PermissionSetupError`.

Every failure becomes `PermissionSetupError(path, os_reason)`. The reason is the OS message only
(`strerror` or the `FormatMessageW` text), so it holds no file name that a caller would have to
redact.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import stat
import sys
from dataclasses import dataclass
from typing import Final

if sys.platform == "win32":
    import ctypes

    from local_agent.oscompat import _win32

__all__ = [
    "STORAGE_FILES",
    "AccessReport",
    "AceInfo",
    "PermissionSetupError",
    "current_user_sid",
    "describe_access",
    "restrict_dir",
    "restrict_file",
    "restrict_storage",
]

STORAGE_FILES: Final = (
    "local-agent.db",
    "local-agent.db-wal",
    "local-agent.db-shm",
    "local-agent.lock",
)

_DIR_MODE: Final = 0o700
_FILE_MODE: Final = 0o600

# ACE_HEADER.AceType of an access-denied ACE. Its layout matches an access-allowed ACE.
_ACCESS_DENIED_ACE_TYPE: Final = 0x01
# Characters reserved for one `FormatMessageW` result.
_MESSAGE_CHARS: Final = 512


class PermissionSetupError(Exception):
    """The Backend could not restrict `path` to the current user (16.9).

    `os_reason` is the OS message for the failure, already free of file names.
    """

    path: str
    os_reason: str

    def __init__(self, path: str, os_reason: str) -> None:
        super().__init__(path, os_reason)
        self.path = path
        self.os_reason = os_reason

    def __str__(self) -> str:
        return f"{self.path}: {self.os_reason}"


@dataclass(frozen=True, slots=True)
class AceInfo:
    """One access control entry of a Windows DACL.

    `ace_type` is `ACE_HEADER.AceType` (`0` allow, `1` deny), `flags` is `AceFlags`
    (`OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE`, ...), and `mask` is the access mask. `sid` holds
    the raw bytes of the entry's SID, compared with `current_user_sid()`. It is empty for ACE types
    that this module does not decode.
    """

    ace_type: int
    flags: int
    mask: int
    sid: bytes


@dataclass(frozen=True, slots=True)
class AccessReport:
    """What `describe_access` found on one path.

    POSIX: `mode` is `stat.S_IMODE(st_mode)`; the Windows fields keep their defaults.
    Windows: `dacl_present` is false for a NULL DACL (everyone has full access); `protected` is the
    `SE_DACL_PROTECTED` flag; `aces` lists the entries in DACL order. `mode` is `None`.
    """

    platform: str
    mode: int | None = None
    dacl_present: bool = False
    protected: bool = False
    aces: tuple[AceInfo, ...] = ()


def _join(platform: str, directory: str, name: str) -> str:
    if platform == "win32":
        return ntpath.join(directory, name)
    return posixpath.join(directory, name)


def _os_reason(exc: OSError) -> str:
    return exc.strerror or type(exc).__name__


# --- POSIX ---------------------------------------------------------------------------------------


def _chmod(path: str, mode: int) -> None:
    """Set `mode` unconditionally, whatever the path's current mode or the umask is (17.36)."""
    try:
        os.chmod(path, mode)
    except OSError as exc:
        raise PermissionSetupError(path, _os_reason(exc)) from exc


def _describe_posix(path: str, platform: str) -> AccessReport:
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError as exc:
        raise PermissionSetupError(path, _os_reason(exc)) from exc
    return AccessReport(platform=platform, mode=mode)


# --- Windows -------------------------------------------------------------------------------------

if sys.platform == "win32":

    def _win_error(code: int) -> OSError:
        """An `OSError` whose `strerror` is the system message for Windows error `code`."""
        buffer = ctypes.create_unicode_buffer(_MESSAGE_CHARS)
        length = _win32.FormatMessageW(
            _win32.FORMAT_MESSAGE_FROM_SYSTEM | _win32.FORMAT_MESSAGE_IGNORE_INSERTS,
            None,
            code,
            0,
            buffer,
            _MESSAGE_CHARS,
            None,
        )
        text = buffer.value.strip() if length else ""
        return OSError(0, text or f"Windows error {code}")

    def _sid_length(address: int) -> int:
        """Byte length of the SID at `address`: 8 header bytes plus 4 per sub-authority."""
        sub_authority_count = ctypes.string_at(address + 1, 1)[0]
        return 8 + 4 * sub_authority_count

    def _read_sid(address: int) -> bytes:
        return bytes(ctypes.string_at(address, _sid_length(address)))

    def current_user_sid() -> bytes:
        """Raw SID bytes of the user who runs this process, from its access token (17.37).

        Raises `OSError` with the Windows message when the token cannot be read.
        """
        token = _win32.HANDLE()
        if not _win32.OpenProcessToken(
            _win32.GetCurrentProcess(), _win32.TOKEN_QUERY, ctypes.byref(token)
        ):
            raise _win_error(ctypes.get_last_error())
        try:
            needed = _win32.DWORD(0)
            # The first call only reports the buffer size; it is expected to fail with
            # ERROR_INSUFFICIENT_BUFFER.
            _win32.GetTokenInformation(token, _win32.TokenUser, None, 0, ctypes.byref(needed))
            error = ctypes.get_last_error()
            if needed.value == 0 or error != _win32.ERROR_INSUFFICIENT_BUFFER:
                raise _win_error(error)
            buffer = ctypes.create_string_buffer(needed.value)
            if not _win32.GetTokenInformation(
                token, _win32.TokenUser, buffer, needed.value, ctypes.byref(needed)
            ):
                raise _win_error(ctypes.get_last_error())
            sid_address = _win32.TOKEN_USER.from_buffer(buffer).User.Sid
            if not sid_address:
                raise OSError(0, "The access token holds no user SID")
            # Copy the SID: it points into `buffer`, which dies with this call.
            return _read_sid(int(sid_address))
        finally:
            _win32.CloseHandle(token)

    def _build_acl(sid: bytes, ace_flags: int) -> ctypes.Array[ctypes.c_char]:
        """A DACL holding one allow entry: `FILE_ALL_ACCESS` for `sid` with `ace_flags`."""
        ace_head = ctypes.sizeof(_win32.ACCESS_ALLOWED_ACE) - ctypes.sizeof(_win32.DWORD)
        size = ctypes.sizeof(_win32.ACL) + ace_head + len(sid)
        size = (size + 3) & ~3  # ACLs are DWORD aligned
        acl = ctypes.create_string_buffer(size)
        if not _win32.InitializeAcl(acl, size, _win32.ACL_REVISION):
            raise _win_error(ctypes.get_last_error())
        sid_buffer = ctypes.create_string_buffer(sid, len(sid))
        if not _win32.AddAccessAllowedAceEx(
            acl, _win32.ACL_REVISION, ace_flags, _win32.FILE_ALL_ACCESS, sid_buffer
        ):
            raise _win_error(ctypes.get_last_error())
        return acl

    def _apply_dacl(path: str, *, inherit: bool) -> None:
        """Replace the DACL of `path` with one protected, user-only entry (17.37).

        `inherit` marks the entry `OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE`, for a directory.
        The owner and the group stay as they are.
        """
        ace_flags = _win32.OBJECT_INHERIT_ACE | _win32.CONTAINER_INHERIT_ACE if inherit else 0
        try:
            acl = _build_acl(current_user_sid(), ace_flags)
            code = _win32.SetNamedSecurityInfoW(
                path,
                _win32.SE_FILE_OBJECT,
                _win32.DACL_SECURITY_INFORMATION | _win32.PROTECTED_DACL_SECURITY_INFORMATION,
                None,
                None,
                acl,
                None,
            )
            if code != _win32.ERROR_SUCCESS:
                raise _win_error(int(code))
        except OSError as exc:
            raise PermissionSetupError(path, _os_reason(exc)) from exc

    def _read_aces(dacl_address: int) -> tuple[AceInfo, ...]:
        aces: list[AceInfo] = []
        count = int(_win32.ACL.from_address(dacl_address).AceCount)
        for index in range(count):
            ace_address = ctypes.c_void_p()
            if not _win32.GetAce(dacl_address, index, ctypes.byref(ace_address)):
                raise _win_error(ctypes.get_last_error())
            if ace_address.value is None:
                raise OSError(0, "GetAce returned no entry")
            address = ace_address.value
            ace = _win32.ACCESS_ALLOWED_ACE.from_address(address)
            ace_type = int(ace.Header.AceType)
            sid = b""
            if ace_type in (_win32.ACCESS_ALLOWED_ACE_TYPE, _ACCESS_DENIED_ACE_TYPE):
                # The SID starts at `SidStart`, the last DWORD of the fixed head.
                sid = _read_sid(address + ctypes.sizeof(_win32.ACCESS_ALLOWED_ACE) - 4)
            aces.append(
                AceInfo(
                    ace_type=ace_type,
                    flags=int(ace.Header.AceFlags),
                    mask=int(ace.Mask),
                    sid=sid,
                )
            )
        return tuple(aces)

    def _describe_windows(path: str, platform: str) -> AccessReport:
        dacl = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        try:
            code = _win32.GetNamedSecurityInfoW(
                path,
                _win32.SE_FILE_OBJECT,
                _win32.DACL_SECURITY_INFORMATION,
                None,
                None,
                ctypes.byref(dacl),
                None,
                ctypes.byref(descriptor),
            )
            if code != _win32.ERROR_SUCCESS:
                raise _win_error(int(code))
            try:
                control = _win32.WORD(0)
                revision = _win32.DWORD(0)
                if not _win32.GetSecurityDescriptorControl(
                    descriptor, ctypes.byref(control), ctypes.byref(revision)
                ):
                    raise _win_error(ctypes.get_last_error())
                protected = bool(control.value & _win32.SE_DACL_PROTECTED)
                if dacl.value is None:
                    return AccessReport(platform=platform, protected=protected)
                return AccessReport(
                    platform=platform,
                    dacl_present=True,
                    protected=protected,
                    aces=_read_aces(dacl.value),
                )
            finally:
                # The descriptor owns the memory behind `dacl`; free it after reading the entries.
                if descriptor.value is not None:
                    _win32.LocalFree(descriptor)
        except OSError as exc:
            raise PermissionSetupError(path, _os_reason(exc)) from exc

else:

    def _unavailable(path: str) -> PermissionSetupError:
        return PermissionSetupError(path, "Windows access control lists need a Windows host")

    def current_user_sid() -> bytes:
        """Raw SID bytes of the current user. Available on Windows only."""
        raise NotImplementedError("current_user_sid needs a Windows host")

    def _apply_dacl(path: str, *, inherit: bool) -> None:
        raise _unavailable(path)

    def _describe_windows(path: str, platform: str) -> AccessReport:
        raise _unavailable(path)


# --- Public API ----------------------------------------------------------------------------------


def restrict_dir(path: str, platform: str) -> None:
    """Make the directory `path` reachable by the current user only (16.8, 17.36, 17.37).

    POSIX: `chmod 0700`, applied even if the directory has a broader mode. Windows: a protected
    DACL with one inheritable full-control entry for the current user.
    Raises `PermissionSetupError(path, os_reason)` on failure.
    """
    if platform == "win32":
        _apply_dacl(path, inherit=True)
    else:
        _chmod(path, _DIR_MODE)


def restrict_file(path: str, platform: str) -> None:
    """Make the file `path` readable and writable by the current user only (16.8, 17.36, 17.37).

    POSIX: `chmod 0600`, applied even if the file has a broader mode. Windows: a protected DACL
    with one full-control entry for the current user and no inheritance flags.
    Raises `PermissionSetupError(path, os_reason)` on failure.
    """
    if platform == "win32":
        _apply_dacl(path, inherit=False)
    else:
        _chmod(path, _FILE_MODE)


def restrict_storage(dir: str, platform: str) -> None:
    """Restrict the Storage_Directory, then every `STORAGE_FILES` entry that exists in it (16.8).

    The directory goes first, so no other user can reach a file while it is being fixed. Entries
    that do not exist are skipped; the Backend restricts them with `restrict_file` after it creates
    them. The first failure raises `PermissionSetupError` and the remaining entries stay as they
    are, which is safe because the caller exits before it serves anything (16.9).
    """
    restrict_dir(dir, platform)
    for name in STORAGE_FILES:
        path = _join(platform, dir, name)
        if os.path.exists(path):
            restrict_file(path, platform)


def describe_access(path: str, platform: str) -> AccessReport:
    """Report the access rules now on `path`, for tests and diagnostics.

    POSIX: the permission bits. Windows: whether the DACL exists and is protected, plus its entries.
    Raises `PermissionSetupError(path, os_reason)` when the path cannot be read.
    """
    if platform == "win32":
        return _describe_windows(path, platform)
    return _describe_posix(path, platform)
