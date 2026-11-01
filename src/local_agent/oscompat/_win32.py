"""Windows `ctypes` declarations (17.10, 17.11, 17.37).

This module imports on every OS. Constants, type aliases, structures, and `HANDLER_ROUTINE` are
defined everywhere, so unit tests can check them on macOS and Linux. The function bindings exist
only when `sys.platform == "win32"`, behind one `if` block, so type checkers on other platforms
see no Windows-only names and code that uses a binding must also sit behind a platform check.

The aliases below use fixed-width `ctypes` types instead of `ctypes.wintypes`. `wintypes.DWORD`
is `c_ulong`, which is 8 bytes on Linux and macOS, so structures built from it would have the
wrong layout off Windows. A Windows `DWORD` is always 32 bits, and so are these aliases.

Callers that need Windows error text for a failed call use `FormatMessageW`. Callers that call a
binding with `use_last_error` semantics read the code with `ctypes.get_last_error()`.

Consumers:
- process and job objects: `oscompat/process.py` (spawn, `terminate_tree`)
- console control: `oscompat/ctrl_break.py`, `oscompat/signals.py`
- machine architecture: `oscompat/platform.py` (`IsWow64Process2`)
- user-only ACLs: `oscompat/permissions.py`
"""

import ctypes
import sys
from collections.abc import Callable
from typing import Any

# --- Type aliases (fixed width, identical on every OS) -------------------------------------------

BYTE = ctypes.c_uint8
WORD = ctypes.c_uint16
USHORT = ctypes.c_uint16
DWORD = ctypes.c_uint32
LONG = ctypes.c_int32
BOOL = ctypes.c_int32
LARGE_INTEGER = ctypes.c_int64
ULONGLONG = ctypes.c_uint64
SIZE_T = ctypes.c_size_t
ULONG_PTR = ctypes.c_size_t
HANDLE = ctypes.c_void_p
PVOID = ctypes.c_void_p

# --- Constants: job objects ----------------------------------------------------------------------

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

# JOBOBJECTINFOCLASS values for SetInformationJobObject / QueryInformationJobObject.
JobObjectBasicAccountingInformation = 1
JobObjectBasicProcessIdList = 3
JobObjectExtendedLimitInformation = 9

# Process ids `JOBOBJECT_BASIC_PROCESS_ID_LIST` has room for. A job with more processes fills the
# list and `QueryInformationJobObject` fails with `ERROR_MORE_DATA`; the filled part is still valid.
JOB_PROCESS_ID_LIST_CAPACITY = 64
ERROR_MORE_DATA = 234

# --- Constants: threads and snapshots ------------------------------------------------------------

TH32CS_SNAPTHREAD = 0x00000004
THREAD_SUSPEND_RESUME = 0x0002
# `ResumeThread` returns this on failure.
RESUME_THREAD_FAILED = 0xFFFFFFFF
# `CreateToolhelp32Snapshot` returns `(HANDLE)-1` on failure. `HANDLE` is a `c_void_p`, so the
# restype yields the unsigned pointer-sized value; compare against `INVALID_HANDLE_VALUE`.
INVALID_HANDLE_VALUE = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1

# --- Constants: console control ------------------------------------------------------------------

CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6

# --- Constants: machine types for IsWow64Process2 ------------------------------------------------

IMAGE_FILE_MACHINE_UNKNOWN = 0x0000
IMAGE_FILE_MACHINE_I386 = 0x014C
IMAGE_FILE_MACHINE_ARM64 = 0xAA64
IMAGE_FILE_MACHINE_AMD64 = 0x8664

# --- Constants: security -------------------------------------------------------------------------

TOKEN_QUERY = 0x0008
# TOKEN_INFORMATION_CLASS value for GetTokenInformation.
TokenUser = 1

ACL_REVISION = 2
ACCESS_ALLOWED_ACE_TYPE = 0x00

FILE_ALL_ACCESS = 0x001F01FF

OBJECT_INHERIT_ACE = 0x01
CONTAINER_INHERIT_ACE = 0x02

# SE_OBJECT_TYPE value for Get/SetNamedSecurityInfoW.
SE_FILE_OBJECT = 1

DACL_SECURITY_INFORMATION = 0x00000004
PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000

# SECURITY_DESCRIPTOR_CONTROL flag reported by GetSecurityDescriptorControl.
SE_DACL_PROTECTED = 0x1000

# --- Constants: errors and message formatting ----------------------------------------------------

ERROR_SUCCESS = 0
ERROR_INSUFFICIENT_BUFFER = 122

FORMAT_MESSAGE_ALLOCATE_BUFFER = 0x00000100
FORMAT_MESSAGE_IGNORE_INSERTS = 0x00000200
FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000


# --- Structures ----------------------------------------------------------------------------------


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", LARGE_INTEGER),
        ("PerJobUserTimeLimit", LARGE_INTEGER),
        ("LimitFlags", DWORD),
        ("MinimumWorkingSetSize", SIZE_T),
        ("MaximumWorkingSetSize", SIZE_T),
        ("ActiveProcessLimit", DWORD),
        ("Affinity", ULONG_PTR),
        ("PriorityClass", DWORD),
        ("SchedulingClass", DWORD),
    )


class IO_COUNTERS(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ULONGLONG),
        ("WriteOperationCount", ULONGLONG),
        ("OtherOperationCount", ULONGLONG),
        ("ReadTransferCount", ULONGLONG),
        ("WriteTransferCount", ULONGLONG),
        ("OtherTransferCount", ULONGLONG),
    )


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    """Argument for `SetInformationJobObject(..., JobObjectExtendedLimitInformation, ...)`.

    Set `BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` so the kernel
    kills every process in the job when the last job handle closes, including on a Backend crash.
    """

    _fields_ = (
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", SIZE_T),
        ("JobMemoryLimit", SIZE_T),
        ("PeakProcessMemoryUsed", SIZE_T),
        ("PeakJobMemoryUsed", SIZE_T),
    )


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    """Result of `QueryInformationJobObject(..., JobObjectBasicAccountingInformation, ...)`.

    `ActiveProcesses == 0` means the Process_Tree is gone.
    """

    _fields_ = (
        ("TotalUserTime", LARGE_INTEGER),
        ("TotalKernelTime", LARGE_INTEGER),
        ("ThisPeriodTotalUserTime", LARGE_INTEGER),
        ("ThisPeriodTotalKernelTime", LARGE_INTEGER),
        ("TotalPageFaultCount", DWORD),
        ("TotalProcesses", DWORD),
        ("ActiveProcesses", DWORD),
        ("TotalTerminatedProcesses", DWORD),
    )


class JOBOBJECT_BASIC_PROCESS_ID_LIST(ctypes.Structure):
    """Result of `QueryInformationJobObject(..., JobObjectBasicProcessIdList, ...)`.

    The first `NumberOfProcessIdsInList` entries of `ProcessIdList` are the pids in the job. The
    Windows header declares a one-element array; this one holds `JOB_PROCESS_ID_LIST_CAPACITY`.
    """

    _fields_ = (
        ("NumberOfAssignedProcesses", DWORD),
        ("NumberOfProcessIdsInList", DWORD),
        ("ProcessIdList", ULONG_PTR * JOB_PROCESS_ID_LIST_CAPACITY),
    )


class THREADENTRY32(ctypes.Structure):
    """Entry for `Thread32First`/`Thread32Next`. Set `dwSize = sizeof(THREADENTRY32)` first."""

    _fields_ = (
        ("dwSize", DWORD),
        ("cntUsage", DWORD),
        ("th32ThreadID", DWORD),
        ("th32OwnerProcessID", DWORD),
        ("tpBasePri", LONG),
        ("tpDeltaPri", LONG),
        ("dwFlags", DWORD),
    )


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = (
        ("Sid", PVOID),
        ("Attributes", DWORD),
    )


class TOKEN_USER(ctypes.Structure):
    """Head of the buffer `GetTokenInformation(TokenUser)` fills; `User.Sid` points into it."""

    _fields_ = (("User", SID_AND_ATTRIBUTES),)


class ACL(ctypes.Structure):
    _fields_ = (
        ("AclRevision", BYTE),
        ("Sbz1", BYTE),
        ("AclSize", WORD),
        ("AceCount", WORD),
        ("Sbz2", WORD),
    )


class ACE_HEADER(ctypes.Structure):
    _fields_ = (
        ("AceType", BYTE),
        ("AceFlags", BYTE),
        ("AceSize", WORD),
    )


class ACCESS_ALLOWED_ACE(ctypes.Structure):
    """Fixed head of an access-allowed ACE; the SID starts at `SidStart` and runs past it."""

    _fields_ = (
        ("Header", ACE_HEADER),
        ("Mask", DWORD),
        ("SidStart", DWORD),
    )


# --- Console control handler type ----------------------------------------------------------------

# BOOL WINAPI HandlerRoutine(DWORD dwCtrlType). `WINFUNCTYPE` selects stdcall on 32-bit Windows;
# off Windows there is no stdcall, and the type only needs to exist for tests.
if sys.platform == "win32":
    HANDLER_ROUTINE = ctypes.WINFUNCTYPE(BOOL, DWORD)
else:
    HANDLER_ROUTINE = ctypes.CFUNCTYPE(BOOL, DWORD)


# --- Function bindings (Windows only) ------------------------------------------------------------

if sys.platform == "win32":
    _PHANDLE = ctypes.POINTER(HANDLE)
    _PDWORD = ctypes.POINTER(DWORD)
    _PUSHORT = ctypes.POINTER(USHORT)
    _PWORD = ctypes.POINTER(WORD)
    _LPWSTR = ctypes.c_wchar_p

    def _declare(
        dll: ctypes.WinDLL,
        name: str,
        restype: Any,
        argtypes: tuple[Any, ...],
    ) -> Callable[..., Any]:
        """Return `dll.name` with `restype` and `argtypes` set.

        Explicit prototypes matter on 64-bit Windows: without them `ctypes` passes handles and
        pointers as 32-bit `int`s and truncates them.
        """
        function = getattr(dll, name)
        function.restype = restype
        function.argtypes = argtypes
        result: Callable[..., Any] = function
        return result

    # `use_last_error=True` gives each thread a private copy of the Windows error code that
    # `ctypes.get_last_error()` reads, so another thread's call cannot overwrite it.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    # Jobs.
    CreateJobObjectW = _declare(kernel32, "CreateJobObjectW", HANDLE, (PVOID, _LPWSTR))
    SetInformationJobObject = _declare(
        kernel32, "SetInformationJobObject", BOOL, (HANDLE, ctypes.c_int, PVOID, DWORD)
    )
    AssignProcessToJobObject = _declare(
        kernel32, "AssignProcessToJobObject", BOOL, (HANDLE, HANDLE)
    )
    QueryInformationJobObject = _declare(
        kernel32,
        "QueryInformationJobObject",
        BOOL,
        (HANDLE, ctypes.c_int, PVOID, DWORD, _PDWORD),
    )
    TerminateJobObject = _declare(kernel32, "TerminateJobObject", BOOL, (HANDLE, ctypes.c_uint))

    # Threads: find and resume the suspended primary thread of a new child.
    CreateToolhelp32Snapshot = _declare(
        kernel32, "CreateToolhelp32Snapshot", HANDLE, (DWORD, DWORD)
    )
    Thread32First = _declare(
        kernel32, "Thread32First", BOOL, (HANDLE, ctypes.POINTER(THREADENTRY32))
    )
    Thread32Next = _declare(kernel32, "Thread32Next", BOOL, (HANDLE, ctypes.POINTER(THREADENTRY32)))
    OpenThread = _declare(kernel32, "OpenThread", HANDLE, (DWORD, BOOL, DWORD))
    ResumeThread = _declare(kernel32, "ResumeThread", DWORD, (HANDLE,))
    CloseHandle = _declare(kernel32, "CloseHandle", BOOL, (HANDLE,))

    # Processes and architecture.
    GetCurrentProcess = _declare(kernel32, "GetCurrentProcess", HANDLE, ())
    IsWow64Process2 = _declare(kernel32, "IsWow64Process2", BOOL, (HANDLE, _PUSHORT, _PUSHORT))

    # Console control (ctrl_break.py, signals.py). `SetConsoleCtrlHandler(None, True)` makes the
    # calling process ignore Ctrl+C; `HANDLER_ROUTINE` registers a callback.
    SetConsoleCtrlHandler = _declare(
        kernel32, "SetConsoleCtrlHandler", BOOL, (HANDLER_ROUTINE, BOOL)
    )
    FreeConsole = _declare(kernel32, "FreeConsole", BOOL, ())
    AttachConsole = _declare(kernel32, "AttachConsole", BOOL, (DWORD,))
    GenerateConsoleCtrlEvent = _declare(kernel32, "GenerateConsoleCtrlEvent", BOOL, (DWORD, DWORD))

    # Error text and memory.
    FormatMessageW = _declare(
        kernel32,
        "FormatMessageW",
        DWORD,
        (DWORD, PVOID, DWORD, DWORD, _LPWSTR, DWORD, PVOID),
    )
    LocalFree = _declare(kernel32, "LocalFree", PVOID, (PVOID,))

    # Security (permissions.py). Pointer arguments are `c_void_p`: callers pass `ctypes.byref`
    # objects, `ctypes.create_string_buffer` buffers, or addresses.
    OpenProcessToken = _declare(advapi32, "OpenProcessToken", BOOL, (HANDLE, DWORD, _PHANDLE))
    GetTokenInformation = _declare(
        advapi32,
        "GetTokenInformation",
        BOOL,
        (HANDLE, ctypes.c_int, PVOID, DWORD, _PDWORD),
    )
    InitializeAcl = _declare(advapi32, "InitializeAcl", BOOL, (PVOID, DWORD, DWORD))
    AddAccessAllowedAceEx = _declare(
        advapi32, "AddAccessAllowedAceEx", BOOL, (PVOID, DWORD, DWORD, DWORD, PVOID)
    )
    SetNamedSecurityInfoW = _declare(
        advapi32,
        "SetNamedSecurityInfoW",
        DWORD,
        (_LPWSTR, ctypes.c_int, DWORD, PVOID, PVOID, PVOID, PVOID),
    )
    GetNamedSecurityInfoW = _declare(
        advapi32,
        "GetNamedSecurityInfoW",
        DWORD,
        (_LPWSTR, ctypes.c_int, DWORD, PVOID, PVOID, PVOID, PVOID, PVOID),
    )
    GetAce = _declare(advapi32, "GetAce", BOOL, (PVOID, DWORD, PVOID))
    GetSecurityDescriptorControl = _declare(
        advapi32, "GetSecurityDescriptorControl", BOOL, (PVOID, _PWORD, _PDWORD)
    )
