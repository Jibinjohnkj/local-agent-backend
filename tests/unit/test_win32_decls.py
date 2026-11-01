"""The `_win32` module imports everywhere and declares Windows layouts and constants (17.37)."""

import ctypes
import sys

from local_agent.oscompat import _win32

_PTR = ctypes.sizeof(ctypes.c_void_p)


def test_constants_match_windows_headers() -> None:
    assert _win32.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE == 0x2000
    assert _win32.ACL_REVISION == 2
    assert _win32.FILE_ALL_ACCESS == 0x1F01FF
    assert _win32.OBJECT_INHERIT_ACE | _win32.CONTAINER_INHERIT_ACE == 0x3
    assert _win32.SE_FILE_OBJECT == 1
    assert _win32.DACL_SECURITY_INFORMATION == 0x4
    assert _win32.PROTECTED_DACL_SECURITY_INFORMATION == 0x80000000


def test_structure_sizes_match_windows_x64() -> None:
    # Sizes are fixed-width, so they hold on any 64-bit host, not only on Windows.
    if _PTR != 8:
        return
    assert ctypes.sizeof(_win32.JOBOBJECT_BASIC_LIMIT_INFORMATION) == 64
    assert ctypes.sizeof(_win32.JOBOBJECT_EXTENDED_LIMIT_INFORMATION) == 144
    assert ctypes.sizeof(_win32.JOBOBJECT_BASIC_ACCOUNTING_INFORMATION) == 48
    assert ctypes.sizeof(_win32.TOKEN_USER) == 16
    assert ctypes.sizeof(_win32.THREADENTRY32) == 28
    assert ctypes.sizeof(_win32.ACL) == 8
    assert ctypes.sizeof(_win32.ACCESS_ALLOWED_ACE) == 12


def test_function_bindings_exist_only_on_windows() -> None:
    assert hasattr(_win32, "CreateJobObjectW") is (sys.platform == "win32")
    assert hasattr(_win32, "HANDLER_ROUTINE")
