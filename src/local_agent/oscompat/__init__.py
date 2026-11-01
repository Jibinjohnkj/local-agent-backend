"""All OS-specific code lives in this package (named to avoid shadowing the stdlib `platform`).

This module holds platform detection (Requirements 17.1, 17.2). `classify_platform` is a pure
function over `PlatformFacts`, so every OS rule is testable on any runner. `gather_facts` is the
only function here that touches the running system.
"""

import platform
import re
import sys
import sysconfig
from dataclasses import dataclass

__all__ = [
    "MIN_GLIBC",
    "MIN_MACOS_MAJOR",
    "MIN_WINDOWS_BUILD",
    "PlatformFacts",
    "PlatformVerdict",
    "classify_platform",
    "gather_facts",
]

MIN_WINDOWS_BUILD = 19045  # Windows 10 22H2; Windows 11 builds start at 22000
MIN_MACOS_MAJOR = 14  # macOS 14 Sonoma
MIN_GLIBC = (2, 28)  # manylinux_2_28
_WINDOWS_11_FIRST_BUILD = 22000
_SUPPORTED_PYTHON = (3, 14)

# IMAGE_FILE_MACHINE_* values reported by IsWow64Process2.
_MACHINE_NAMES = {
    0x8664: "x86_64",
    0xAA64: "arm64",
    0x014C: "x86",
    0x01C4: "arm",
}


@dataclass(frozen=True)
class PlatformFacts:
    """What the running interpreter and OS report about themselves."""

    system: str  # platform.system(): "Windows" | "Darwin" | "Linux" | other
    release: str  # Windows build number, macOS product version, or "" on Linux
    native_machine: str  # normalized: "x86_64" | "arm64" | other
    libc: tuple[str, str]  # platform.libc_ver(), ("glibc", "2.39") or ("", "")
    implementation: str  # sys.implementation.name
    version: tuple[int, int]
    gil_disabled: bool  # sysconfig.get_config_var("Py_GIL_DISABLED") == 1


@dataclass(frozen=True)
class PlatformVerdict:
    """Result of classifying a `PlatformFacts` against the Supported_Platform table."""

    supported: bool
    os_label: str  # e.g. "Windows 10 build 19045", "macOS 14.4", "Linux glibc 2.39"
    arch: str  # normalized architecture, e.g. "x86_64", "arm64"
    build: str  # Python build, e.g. "CPython 3.14 (GIL)"


def normalize_machine(machine: str) -> str:
    """Map the many spellings of a CPU architecture onto `x86_64` and `arm64`.

    Anything else is returned lower-cased and stripped, so it never matches a supported value.
    """
    value = machine.strip().lower()
    if value in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if value in ("arm64", "aarch64"):
        return "arm64"
    return value


def _leading_ints(text: str, count: int) -> tuple[int, ...] | None:
    """Parse the first `count` dot-separated integers of `text`, or None if not possible."""
    parts = re.findall(r"\d+", text)
    if len(parts) < count:
        return None
    return tuple(int(p) for p in parts[:count])


def _windows_build_number(release: str) -> int | None:
    """Read a build number from `19045` or from a full version such as `10.0.19045`."""
    numbers = re.findall(r"\d+", release)
    if len(numbers) == 1:
        return int(numbers[0])
    if len(numbers) >= 3:
        return int(numbers[2])
    return None


def _python_build_label(f: PlatformFacts) -> str:
    impl = {"cpython": "CPython"}.get(f.implementation, f.implementation or "unknown")
    gil = "free-threaded" if f.gil_disabled else "GIL"
    return f"{impl} {f.version[0]}.{f.version[1]} ({gil})"


def classify_platform(f: PlatformFacts) -> PlatformVerdict:
    """Report whether `f` is a Supported_Platform. Pure: no I/O, no global state."""
    arch = normalize_machine(f.native_machine)
    arch_ok = arch in ("x86_64", "arm64")
    os_ok = False
    os_label = f.system or "unknown"

    if f.system == "Windows":
        build = _windows_build_number(f.release)
        if build is None:
            os_label = "Windows (unknown build)"
        else:
            edition = "11" if build >= _WINDOWS_11_FIRST_BUILD else "10"
            os_label = f"Windows {edition} build {build}"
            # Windows arm64 is not supported, even through x64 emulation.
            os_ok = build >= MIN_WINDOWS_BUILD and arch == "x86_64"
    elif f.system == "Darwin":
        parsed = _leading_ints(f.release, 1)
        os_label = f"macOS {f.release}" if f.release else "macOS (unknown version)"
        os_ok = parsed is not None and parsed[0] >= MIN_MACOS_MAJOR and arch_ok
    elif f.system == "Linux":
        libc_name, libc_version = f.libc
        if libc_name == "glibc":
            os_label = f"Linux glibc {libc_version}".rstrip()
            parsed = _leading_ints(libc_version, 2)
            os_ok = parsed is not None and parsed >= MIN_GLIBC and arch_ok
        else:
            # platform.libc_ver() returns ("", "") on musl and on unknown libraries.
            os_label = f"Linux ({libc_name or 'non-glibc'})"

    python_ok = (
        f.implementation == "cpython" and f.version == _SUPPORTED_PYTHON and not f.gil_disabled
    )
    return PlatformVerdict(
        supported=os_ok and python_ok,
        os_label=os_label,
        arch=arch,
        build=_python_build_label(f),
    )


def _windows_native_machine() -> str | None:
    """Ask `IsWow64Process2` for the real CPU, which `platform.machine()` hides under emulation.

    Returns None when the call is unavailable (Windows before 10 1709) or fails.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        is_wow64_process2 = getattr(kernel32, "IsWow64Process2", None)
        if is_wow64_process2 is None:
            return None
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        is_wow64_process2.restype = wintypes.BOOL
        is_wow64_process2.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.USHORT),
            ctypes.POINTER(wintypes.USHORT),
        ]
        process_machine = wintypes.USHORT(0)
        native_machine = wintypes.USHORT(0)
        ok = is_wow64_process2(
            kernel32.GetCurrentProcess(),
            ctypes.byref(process_machine),
            ctypes.byref(native_machine),
        )
        if ok:
            return _MACHINE_NAMES.get(
                native_machine.value, f"unknown(0x{native_machine.value:04x})"
            )
    return None


def _windows_build_string() -> str:
    build = platform.version()
    if sys.platform == "win32":
        build = str(sys.getwindowsversion().build)
    return build


def gather_facts() -> PlatformFacts:
    """Read the facts about the running system. The only I/O in platform detection."""
    system = platform.system()
    libc: tuple[str, str] = ("", "")
    machine = normalize_machine(platform.machine())

    if system == "Windows":
        release = _windows_build_string()
        native = _windows_native_machine()
        if native is not None:
            machine = normalize_machine(native)
    elif system == "Darwin":
        release = platform.mac_ver()[0]
    elif system == "Linux":
        release = ""
        libc = platform.libc_ver()
    else:
        release = platform.release()

    return PlatformFacts(
        system=system,
        release=release,
        native_machine=machine,
        libc=libc,
        implementation=sys.implementation.name,
        version=(sys.version_info.major, sys.version_info.minor),
        gil_disabled=sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
    )
