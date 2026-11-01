"""Property test for `${NAME}` substitution in the config tree (Requirements 13.2, 17.21).

`substitute_env` is checked against a reference built next to the input: each generated node is
a pair of (what the YAML parser would hand over, what substitution must return). Strings are
assembled from literal text, `${NAME}` references, and `$${` escapes, and the expected text is
assembled from the matching pieces, so the reference never re-parses the input.
"""

import copy
from collections.abc import Callable
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from local_agent.config.loader import substitute_env

_NAME_START = "abcXYZ_"
_NAME_REST = "abcXYZ_019"
# Literal text has no "$": a "$" next to a reference or an escape would change how the pieces
# tokenize (for example "$" + "${A}" reads as the escape "$${" followed by "A}").
_LITERAL = st.text(alphabet="ab {}X_-.:/\\\u00e9", max_size=6)
_platforms = st.sampled_from(["win32", "darwin", "linux"])
# Case rewrites for a reference's spelling; only Windows is meant to see through them.
_case_transforms: list[Callable[[str], str]] = [
    lambda s: s,
    str.upper,
    str.lower,
    str.swapcase,
]
_recase = st.sampled_from(_case_transforms)
_names = st.builds(
    lambda head, rest: head + rest,
    st.sampled_from(_NAME_START),
    st.text(alphabet=_NAME_REST, max_size=5),
)
# Values may themselves contain "${" and "$${"; substitution is a single pass, so they stay as is.
_values = st.one_of(st.text(max_size=12), st.text(alphabet="$_{}AB ", max_size=8))
# Keys are never substituted, so they may look like references.
_keys = st.text(alphabet="ab${}X_ ", max_size=6)

_Pair = st.SearchStrategy[tuple[Any, Any]]


@st.composite
def _strings(draw: st.DrawFn, env: dict[str, str], platform: str) -> tuple[str, str]:
    """A string made of literals, references to names in `env`, and escapes, with its result."""
    source: list[str] = []
    expected: list[str] = []
    kinds = ["literal", "escape", "reference"] if env else ["literal", "escape"]
    for kind in draw(st.lists(st.sampled_from(kinds), max_size=6)):
        if kind == "literal":
            text = draw(_LITERAL)
            source.append(text)
            expected.append(text)
        elif kind == "escape":
            source.append("$${")
            expected.append("${")
        else:
            defined = draw(st.sampled_from(sorted(env)))
            spelled = draw(_recase)(defined) if platform == "win32" else defined
            source.append("${" + spelled + "}")
            expected.append(env[defined])
    return "".join(source), "".join(expected)


@st.composite
def _cases(draw: st.DrawFn) -> tuple[Any, Any, dict[str, str], str]:
    """A tree, the tree substitution must return, an environment defining every name, a platform.

    Names in the environment are unique after the platform's comparison rule, so a reference
    always has exactly one entry to find.
    """
    platform = draw(_platforms)
    unique_key = str.upper if platform == "win32" else str
    names = draw(st.lists(_names, max_size=5, unique_by=unique_key))
    env = {name: draw(_values) for name in names}

    leaves: _Pair = st.one_of(
        _strings(env, platform),
        st.integers().map(lambda n: (n, n)),
        st.floats(allow_nan=False).map(lambda x: (x, x)),
        st.booleans().map(lambda b: (b, b)),
        st.none().map(lambda n: (n, n)),
    )

    def extend(children: _Pair) -> _Pair:
        lists = st.lists(children, max_size=4).map(
            lambda items: ([a for a, _ in items], [b for _, b in items])
        )
        mappings = st.dictionaries(_keys, children, max_size=4).map(
            lambda items: (
                {k: a for k, (a, _) in items.items()},
                {k: b for k, (_, b) in items.items()},
            )
        )
        return st.one_of(lists, mappings)

    tree, expected = draw(st.recursive(leaves, extend, max_leaves=12))
    return tree, expected, env, platform


# Feature: local-agent-backend, Property 24: Environment substitution
# Validates: Requirements 13.2
@given(_cases())
def test_substitute_env_matches_reference(case: tuple[Any, Any, dict[str, str], str]) -> None:
    tree, expected, env, platform = case
    original = copy.deepcopy(tree)

    result = substitute_env(tree, env, platform=platform)

    assert result == expected
    # `==` treats True as 1 and 1.0 as 1; the repr also pins the types and the key order.
    assert repr(result) == repr(expected)
    # Pure: the input tree is not modified.
    assert tree == original
