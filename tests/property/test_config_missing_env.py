"""Property test for missing environment variable errors (Requirement 13.8).

A generated YAML-compatible tree holds `${NAME}` references, `$${` escapes, and plain text. Some
of the referenced names are undefined. Substitution, and a full load of the same tree rendered
as YAML, must report every missing variable with the dotted path of the string that holds it,
and no error text may contain the value of a variable that was defined.

Trees are generated as segment tuples, and the expected issues are computed from the segments,
so the check does not reuse the loader's reference pattern.
"""

import re
import string
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.config.loader import (
    ConfigError,
    ConfigIssue,
    load_config_from_text,
    substitute_env,
)

type Segment = tuple[str, str]  # ("lit", text) | ("ref", NAME) | ("esc", "")
type Node = tuple[Segment, ...] | int | list[Node] | dict[str, Node]

_NAME = st.from_regex(r"[A-Z_][A-Z0-9_]{0,5}", fullmatch=True)
# Keys start with "k" so no generated key can equal a Backend_Config section name.
_KEY = st.text(string.ascii_lowercase, max_size=4).map(lambda s: "k" + s)
_LITERAL = st.text(string.ascii_lowercase + " ", max_size=6)
# Defined values carry "@@" markers and digits, which appear in no path, name, or message text.
_VALUE = st.text(string.digits, min_size=5, max_size=10).map(lambda s: f"@@{s}@@")
_MESSAGE = re.compile(r"environment variable ([A-Z_][A-Z0-9_]*) is not set")


@dataclass(frozen=True)
class _Scenario:
    tree: dict[str, Node]
    env: dict[str, str]
    missing: frozenset[str]
    platform: str


def _render(node: Node) -> Any:
    """The raw YAML data for `node`: segment tuples become strings."""
    if isinstance(node, tuple):
        parts = {"lit": "{}", "ref": "${{{}}}", "esc": "$${{"}
        return "".join(parts[kind].format(text) for kind, text in node)
    if isinstance(node, list):
        return [_render(item) for item in node]
    if isinstance(node, dict):
        return {key: _render(item) for key, item in node.items()}
    return node


def _references(node: Node, path: str) -> Iterator[tuple[str, str]]:
    """Every `(dotted path, NAME)` pair for a `${NAME}` reference in `node`."""
    if isinstance(node, tuple):
        for kind, text in node:
            if kind == "ref":
                yield path, text
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _references(item, f"{path}.{index}" if path else str(index))
    elif isinstance(node, dict):
        for key, item in node.items():
            yield from _references(item, f"{path}.{key}" if path else key)


@st.composite
def _scenarios(draw: st.DrawFn) -> _Scenario:
    names = draw(st.lists(_NAME, min_size=1, max_size=6, unique=True))
    split = draw(st.integers(min_value=1, max_value=len(names)))
    missing, defined = names[:split], names[split:]
    env = {name: draw(_VALUE) for name in defined}

    segment: st.SearchStrategy[Segment] = st.one_of(
        st.tuples(st.just("lit"), _LITERAL),
        st.tuples(st.just("ref"), st.sampled_from(names)),
        st.tuples(st.just("esc"), st.just("")),
    )
    leaf: st.SearchStrategy[Node] = st.one_of(
        st.lists(segment, max_size=5).map(tuple), st.integers(-1000, 1000)
    )
    node = st.recursive(
        leaf,
        lambda children: st.one_of(
            st.lists(children, max_size=3), st.dictionaries(_KEY, children, max_size=3)
        ),
        max_leaves=12,
    )
    tree: dict[str, Node] = draw(st.dictionaries(_KEY, node, max_size=4))
    # One string that is certain to reference a missing name, so the tree always has an error.
    tree["_missing"] = (("lit", "x"), ("ref", draw(st.sampled_from(missing))))
    return _Scenario(
        tree, env, frozenset(missing), draw(st.sampled_from(["win32", "darwin", "linux"]))
    )


def _missing_pairs(issues: list[ConfigIssue]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for issue in issues:
        found = _MESSAGE.fullmatch(issue.message)
        if found:
            pairs.add((issue.path, found.group(1)))
    return pairs


# Feature: local-agent-backend, Property 25: Missing environment variable error
# Validates: Requirements 13.8
@settings(max_examples=100)
@given(_scenarios())
def test_missing_env_variables_are_reported_with_paths_and_no_values(
    scenario: _Scenario,
) -> None:
    expected = {
        (path, name) for path, name in _references(scenario.tree, "") if name in scenario.missing
    }
    assert expected
    leaked = {
        scenario.env[name] for _, name in _references(scenario.tree, "") if name in scenario.env
    }
    raw = _render(scenario.tree)

    errors: list[ConfigError] = []
    try:
        substitute_env(raw, scenario.env, platform=scenario.platform)
    except ConfigError as exc:
        errors.append(exc)
    try:
        load_config_from_text(
            yaml.safe_dump(raw), scenario.env, "config.yaml", platform=scenario.platform
        )
    except ConfigError as exc:
        errors.append(exc)

    # Both entry points must fail, and each must name every missing variable at its path.
    assert len(errors) == 2
    for error in errors:
        assert expected <= _missing_pairs(error.issues)
        text = str(error) + "".join(f"{i.path}\n{i.message}\n" for i in error.issues)
        for value in leaked:
            assert value not in text
    # Substitution reports nothing but the missing variables.
    assert _missing_pairs(errors[0].issues) == expected
    assert all(_MESSAGE.fullmatch(issue.message) for issue in errors[0].issues)
