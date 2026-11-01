"""Property test for line-ending and BOM invariance (Requirements 17.23, 17.25, 13.7).

A config file may use LF or CRLF line endings, per line, and may start with a UTF-8 BOM. Loading
must give the same `BackendConfig` as the LF, BOM-less text. For malformed YAML the reported
line and column must not depend on either.

Every variant is built in memory. The valid texts use several scalar styles that span lines
(literal and folded blocks, quoted and plain multi-line scalars), because those are where a stray
carriage return would change a value. Texts go through `load_config` (bytes on disk, so
`decode_config_bytes` is in the path) and through `load_config_from_text`.
"""

import string
import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from local_agent.config.loader import (
    ConfigError,
    ConfigIssue,
    decode_config_bytes,
    load_config,
    load_config_from_text,
)
from local_agent.config.models import BackendConfig

_BOM = b"\xef\xbb\xbf"
_SOURCE = "<line-endings>"

# Safe content: no quotes, backslashes, colons, hashes, or characters that start YAML syntax.
# Every line starts with a letter and ends without a space, so each style keeps it verbatim.
_LINE_BODY = string.ascii_letters + string.digits + "é日🙂 ,._"
_line = st.builds(
    lambda head, tail: ("z" + head + tail).rstrip(),
    st.sampled_from(["", "a", "É", "日"]),
    st.text(_LINE_BODY, max_size=12),
)
# Interior blank lines are allowed in literal and folded blocks, so lines may be empty there.
_block_lines = st.lists(_line, min_size=1, max_size=4).flatmap(
    lambda lines: st.lists(st.booleans(), min_size=len(lines), max_size=len(lines)).map(
        lambda blanks: _with_blank_lines(lines, blanks)
    )
)
_plain_lines = st.lists(_line, min_size=1, max_size=4)
_STYLES = ["literal", "folded", "double", "single", "plain", "single_line"]
_comment = st.one_of(
    st.just(""),
    st.text(_LINE_BODY, max_size=10).map(lambda t: "# " + t),
)


def _with_blank_lines(lines: list[str], blanks: list[bool]) -> list[str]:
    """Insert a blank line before each line flagged in `blanks` (never first or last)."""
    result: list[str] = []
    for index, (line, blank) in enumerate(zip(lines, blanks, strict=True)):
        if blank and index > 0:
            result.append("")
        result.append(line)
    return result


def _render(style: str, lines: list[str], indent: int) -> str:
    """The text that follows `key: ` or `- `, continuation lines indented past `indent`."""
    pad = " " * (indent + 2)
    if style == "literal":
        return "|-" + "".join("\n" + (pad + line if line else "") for line in lines)
    if style == "folded":
        return ">-" + "".join("\n" + (pad + line if line else "") for line in lines)
    content = [line for line in lines if line]
    if style == "double":
        return '"' + ("\n" + pad).join(content) + '"'
    if style == "single":
        return "'" + ("\n" + pad).join(content) + "'"
    if style == "plain":
        return ("\n" + pad).join(content)
    return " ".join(content)  # single_line: plain on one line


@st.composite
def _scalar(draw: st.DrawFn, indent: int) -> str:
    style = draw(st.sampled_from(_STYLES))
    lines = draw(_block_lines if style in ("literal", "folded") else _plain_lines)
    return _render(style, lines, indent)


@st.composite
def _valid_units(draw: st.DrawFn) -> list[str]:
    """Top-level chunks of a valid config; joined with newlines they form the document."""
    units = ["llm:", "  model: " + draw(_scalar(2)), "agent:"]
    units.append("  name: " + draw(_scalar(2)))
    units.append("  description: " + draw(_scalar(2)))
    units.append("  version: " + draw(_scalar(2)))
    if draw(st.booleans()):
        units.append("  base_instructions: " + draw(_scalar(2)))
    args = draw(st.lists(_scalar(6), max_size=3))
    units.extend(["mcp_servers:", "  - name: srv", "    transport: stdio", "    command: tool"])
    if args:
        units.append("    args:")
        units.extend("      - " + arg for arg in args)
    else:
        units.append("    args: []")
    units.extend(["storage:", "  backend: memory"])
    return units


def _interleave(units: list[str], fillers: list[str]) -> str:
    """Join `units` with newlines, adding a blank or comment line after some of them."""
    out: list[str] = []
    for index, unit in enumerate(units):
        out.append(unit)
        if index < len(fillers):
            out.append(fillers[index])
    return "\n".join(out)


@st.composite
def _valid_text(draw: st.DrawFn) -> str:
    units = draw(_valid_units())
    fillers = draw(st.lists(_comment, min_size=len(units), max_size=len(units)))
    text = _interleave(units, fillers)
    if draw(st.booleans()):
        text = "---\n" + text
    if draw(st.booleans()):
        text += "\n"
    return text


# One bad line at column 0. Each makes `yaml.safe_load` fail wherever it sits after valid lines.
_BAD_LINES = [
    "foo: [1, 2",
    "bar: {a: 1",
    "\tx: 1",
    "a: b: c",
    "key: 'unterminated",
    '"unterminated',
    "key: *undefined_alias",
    "{unclosed",
    "ctl: \x07",
]


@st.composite
def _malformed_text(draw: st.DrawFn) -> str:
    units = draw(_valid_units())
    keep = draw(st.integers(0, len(units)))
    fillers = draw(st.lists(_comment, min_size=keep, max_size=keep))
    prefix = _interleave(units[:keep], fillers)
    bad = draw(st.sampled_from(_BAD_LINES))
    return prefix + "\n" + bad if keep else bad


def _with_endings(text: str, crlf: list[bool]) -> str:
    """Replace the i-th LF of `text` with CRLF when `crlf[i]` is true."""
    parts = text.split("\n")
    out = [part + ("\r\n" if crlf[i] else "\n") for i, part in enumerate(parts[:-1])]
    return "".join(out) + parts[-1]


@st.composite
def _variants(draw: st.DrawFn, text: str) -> list[tuple[bool, str]]:
    """(has_bom, text) variants of the LF text: mixed and all-CRLF endings, with and without BOM."""
    count = text.count("\n")
    mixed = draw(st.lists(st.booleans(), min_size=count, max_size=count))
    return [
        (draw(st.booleans()), _with_endings(text, mixed)),
        (draw(st.booleans()), _with_endings(text, [True] * count)),
        (True, text),
        (False, _with_endings(text, [True] * count)),
    ]


def _as_bytes(has_bom: bool, text: str) -> bytes:
    return (_BOM if has_bom else b"") + text.encode("utf-8")


def _issues_of_load(path: Path) -> list[ConfigIssue]:
    try:
        load_config(path, {})
    except ConfigError as exc:
        return exc.issues
    raise AssertionError("expected a ConfigError")


# Feature: local-agent-backend, Property 35: Line-ending and BOM invariance
# **Validates: Requirements 17.23, 17.25, 13.7**
@given(data=st.data(), malformed=st.booleans())
def test_line_endings_and_bom_do_not_change_the_result(
    data: st.DataObject, malformed: bool
) -> None:
    text = data.draw(_malformed_text() if malformed else _valid_text())
    variants = data.draw(_variants(text))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_bytes(_as_bytes(False, text))

        if malformed:
            expected = _issues_of_load(path)
            assert len(expected) == 1
            assert expected[0].message.startswith("malformed YAML")
            expected_text = _issues_of_load_text(text)
            for has_bom, variant in variants:
                path.write_bytes(_as_bytes(has_bom, variant))
                assert _issues_of_load(path) == expected
                assert _issues_of_load_text(variant) == expected_text
                bom_text = ("\ufeff" if has_bom else "") + variant
                assert _issues_of_load_text(bom_text) == expected_text
            return

        baseline = load_config(path, {})
        assert baseline == load_config_from_text(text, {}, _SOURCE)
        for has_bom, variant in variants:
            data_bytes = _as_bytes(has_bom, variant)
            path.write_bytes(data_bytes)
            assert load_config(path, {}) == baseline
            decoded = decode_config_bytes(data_bytes, _SOURCE)
            assert decoded == variant
            assert load_config_from_text(decoded, {}, _SOURCE) == baseline
            assert load_config_from_text(("\ufeff" if has_bom else "") + variant, {}, _SOURCE) == (
                baseline
            )
        assert isinstance(baseline, BackendConfig)


def _issues_of_load_text(text: str) -> list[ConfigIssue]:
    try:
        load_config_from_text(text, {}, _SOURCE)
    except ConfigError as exc:
        return exc.issues
    raise AssertionError("expected a ConfigError")
