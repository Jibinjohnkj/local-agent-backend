"""Property test for environment-variable name matching (Requirements 17.21, 13.2).

`lookup_env` matches a name exactly on macOS and Linux, and by upper-cased name on Windows.
It returns `None` when no entry matches. The test checks the function against a small model
of that rule, over mappings whose names are unique after upper-casing.
"""

from collections.abc import Callable

from hypothesis import given
from hypothesis import strategies as st

from local_agent.oscompat.env import lookup_env

# Letters, digits and punctuation seen in real names, plus characters whose case mapping is not
# one-to-one (sharp s upper-cases to "SS", dotless i to "I", long s to "S"), so that the ordinal
# `str.upper()` rule is exercised rather than an ASCII-only approximation.
_NAME_ALPHABET = st.sampled_from("abcXYZ019_()\u00df\u0131\u017f\u00e9\u00c9")
_names = st.text(_NAME_ALPHABET, min_size=1, max_size=8)
_values = st.text(max_size=12)
_platforms = st.sampled_from(["win32", "darwin", "linux"])
_case_transforms: list[Callable[[str], str]] = [
    lambda s: s,
    str.upper,
    str.lower,
    str.swapcase,
]


@st.composite
def _lookup_cases(draw: st.DrawFn) -> tuple[dict[str, str], str, str]:
    """An environment, a `NAME` to look up, and a platform.

    Names in the environment are unique after upper-casing. `NAME` is either an existing
    name, a re-cased copy of one, or an unrelated name that may or may not be present.
    """
    names = draw(st.lists(_names, max_size=6, unique_by=str.upper))
    env = {name: draw(_values) for name in names}
    reuse_existing: bool = draw(st.booleans())
    if names and reuse_existing:
        base = draw(st.sampled_from(names))
        wanted = draw(st.sampled_from(_case_transforms))(base)
    else:
        wanted = draw(_names)
    return env, wanted, draw(_platforms)


# Feature: local-agent-backend, Property 37: Environment-variable name matching
# Validates: Requirements 17.21, 13.2
@given(_lookup_cases())
def test_lookup_env_matches_names_by_platform_rule(
    case: tuple[dict[str, str], str, str],
) -> None:
    env, wanted, platform = case

    if platform == "win32":
        matches = [v for k, v in env.items() if k.upper() == wanted.upper()]
    else:
        matches = [v for k, v in env.items() if k == wanted]
    # Names are unique after upper-casing, so at most one entry can match on any platform.
    assert len(matches) <= 1
    expected = matches[0] if matches else None

    assert lookup_env(env, wanted, platform) == expected
