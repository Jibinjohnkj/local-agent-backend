"""Property test for the subprocess environment merge (Requirements 17.22)."""

from collections.abc import Mapping

from hypothesis import given
from hypothesis import strategies as st

from local_agent.oscompat.env import merge_env

# A small alphabet makes names that differ only by case (``Path`` / ``PATH`` / ``path``) common.
# ``ß`` upper-cases to ``SS``, which covers names whose upper-cased form changes length.
_NAME_ALPHABET = "pPaAtThH_ßsS1"
_names = st.text(alphabet=_NAME_ALPHABET, min_size=1, max_size=4)
_values = st.text(alphabet="xyzXYZ01 ;:/\\", max_size=6)
_platforms = st.sampled_from(["win32", "darwin", "linux"])


def _unique_after_upper(env: dict[str, str]) -> dict[str, str]:
    """Keep the first entry for each upper-cased name, as a real Windows environment holds."""
    seen: set[str] = set()
    unique: dict[str, str] = {}
    for name, value in env.items():
        if name.upper() not in seen:
            seen.add(name.upper())
            unique[name] = value
    return unique


@st.composite
def _merge_inputs(
    draw: st.DrawFn,
) -> tuple[dict[str, str], dict[str, str], str]:
    platform = draw(_platforms)
    inherited = draw(st.dictionaries(_names, _values, max_size=6))
    configured = draw(st.dictionaries(_names, _values, max_size=6))
    if platform == "win32":
        # Windows environments never hold two spellings of one name, so each input mapping is
        # unique after upper-casing. Collisions between the two mappings are still generated.
        inherited = _unique_after_upper(inherited)
        configured = _unique_after_upper(configured)
    return inherited, configured, platform


def _key(name: str, platform: str) -> str:
    return name.upper() if platform == "win32" else name


def _check_merge(
    inherited: Mapping[str, str], configured: Mapping[str, str], platform: str
) -> None:
    result = merge_env(inherited, configured, platform)
    configured_keys = {_key(n, platform) for n in configured}

    # Every configured name is present with its configured value.
    for name, value in configured.items():
        assert result[name] == value

    # Nothing appears that was in neither input.
    assert set(result) <= set(inherited) | set(configured)

    # Every inherited entry that no configured name replaces is unchanged.
    for name, value in inherited.items():
        if _key(name, platform) not in configured_keys:
            assert result[name] == value

    if platform == "win32":
        # Exactly one entry per upper-cased name.
        assert len(result) == len({name.upper() for name in result})
        # No inherited entry survives under a name a configured entry replaces.
        for name in result:
            if name.upper() in configured_keys:
                assert name in configured
    else:
        # Exact-key union, configured values winning.
        assert result == {**inherited, **configured}


# Feature: local-agent-backend, Property 38: Subprocess environment merge
# Validates: Requirements 17.22
@given(_merge_inputs())
def test_merge_env_overlays_configured_over_inherited(
    inputs: tuple[dict[str, str], dict[str, str], str],
) -> None:
    inherited, configured, platform = inputs
    _check_merge(inherited, configured, platform)
