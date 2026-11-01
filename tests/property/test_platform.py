"""Property test for platform classification (Requirements 17.1, 17.2).

The oracle below is written straight from the Supported_Platform table and works on the generated
semantic values (build number, major version, glibc pair). It does not parse strings or call any
helper from the implementation, so a bug in `classify_platform` cannot cancel out in the oracle.
"""

from dataclasses import dataclass

from hypothesis import event, given, settings
from hypothesis import strategies as st

from local_agent.oscompat import PlatformFacts, classify_platform

X86_64_SPELLINGS = ["x86_64", "AMD64", "amd64", "x64", " X86_64 "]
ARM64_SPELLINGS = ["arm64", "ARM64", "aarch64", "AArch64"]
OTHER_MACHINES = ["x86", "i686", "arm", "armv7l", "riscv64", "ppc64le", "s390x", ""]


@dataclass(frozen=True)
class Case:
    """A generated `PlatformFacts` plus the facts the oracle needs, in already-parsed form."""

    facts: PlatformFacts
    system: str
    arch: str | None  # "x86_64", "arm64", or None for any other CPU
    windows_build: int | None
    macos_major: int | None
    glibc: tuple[int, int] | None  # None when the libc is not glibc


@st.composite
def cases(draw: st.DrawFn) -> Case:
    # Weighted toward the three known systems so supported verdicts come up often.
    systems = ["Windows", "Darwin", "Linux"] * 3 + ["FreeBSD", "Java", ""]
    system = draw(st.sampled_from(systems))

    machine = draw(
        st.one_of(
            st.sampled_from(X86_64_SPELLINGS + ARM64_SPELLINGS),
            st.sampled_from(X86_64_SPELLINGS + ARM64_SPELLINGS + OTHER_MACHINES),
        )
    )
    if machine in X86_64_SPELLINGS:
        arch: str | None = "x86_64"
    elif machine in ARM64_SPELLINGS:
        arch = "arm64"
    else:
        arch = None

    release = ""
    libc = ("", "")
    windows_build: int | None = None
    macos_major: int | None = None
    glibc: tuple[int, int] | None = None

    if system == "Windows":
        # Dense around the 19045 cutoff and the 22000 Windows 11 start, plus a wide range.
        windows_build = draw(
            st.one_of(
                st.integers(min_value=19040, max_value=19050),
                st.integers(min_value=21995, max_value=22005),
                st.integers(min_value=0, max_value=30000),
            )
        )
        release = draw(st.sampled_from(["{b}", "10.0.{b}"])).format(b=windows_build)
    elif system == "Darwin":
        macos_major = draw(st.one_of(st.integers(min_value=12, max_value=16), st.integers(0, 40)))
        minor = draw(st.integers(min_value=0, max_value=9))
        release = f"{macos_major}.{minor}"
        if draw(st.booleans()):
            release += f".{draw(st.integers(min_value=0, max_value=9))}"
    elif system == "Linux":
        if draw(st.sampled_from([True, True, True, False])):
            glibc = draw(
                st.one_of(
                    st.tuples(st.just(2), st.integers(min_value=26, max_value=30)),
                    st.tuples(st.integers(min_value=1, max_value=3), st.integers(0, 40)),
                )
            )
            libc = ("glibc", f"{glibc[0]}.{glibc[1]}")
        else:
            libc = draw(st.sampled_from([("", ""), ("musl", "1.2"), ("libc", "6.0")]))

    # Most draws use the supported Python build, so the OS rules decide the verdict. The rest
    # vary each Python field independently.
    if draw(st.sampled_from([True, True, True, False])):
        implementation, version, gil_disabled = "cpython", (3, 14), False
    else:
        implementation = draw(st.sampled_from(["cpython", "cpython", "pypy", "graalpy"]))
        version = draw(
            st.one_of(
                st.just((3, 14)),
                st.tuples(st.just(3), st.integers(min_value=9, max_value=16)),
            )
        )
        gil_disabled = draw(st.booleans())

    facts = PlatformFacts(
        system=system,
        release=release,
        native_machine=machine,
        libc=libc,
        implementation=implementation,
        version=version,
        gil_disabled=gil_disabled,
    )
    return Case(facts, system, arch, windows_build, macos_major, glibc)


def expected_supported(case: Case) -> bool:
    """The Supported_Platform table from Requirement 17.1."""
    f = case.facts
    python_ok = f.implementation == "cpython" and f.version == (3, 14) and not f.gil_disabled
    if case.system == "Windows":
        assert case.windows_build is not None
        os_ok = case.windows_build >= 19045 and case.arch == "x86_64"
    elif case.system == "Darwin":
        assert case.macos_major is not None
        os_ok = case.macos_major >= 14 and case.arch is not None
    elif case.system == "Linux":
        os_ok = case.glibc is not None and case.glibc >= (2, 28) and case.arch is not None
    else:
        os_ok = False
    return python_ok and os_ok


# Feature: local-agent-backend, Property 33: Platform classification
# **Validates: Requirements 17.1, 17.2**
@settings(max_examples=200)
@given(case=cases())
def test_platform_classification(case: Case) -> None:
    verdict = classify_platform(case.facts)

    event(f"{case.system or 'empty'} supported={verdict.supported}")
    assert verdict.supported == expected_supported(case)

    # The verdict always names the architecture and Python build, supported or not, so the
    # startup warning can report them.
    if case.arch is not None:
        assert verdict.arch == case.arch
    assert f"{case.facts.version[0]}.{case.facts.version[1]}" in verdict.build
    assert verdict.os_label
