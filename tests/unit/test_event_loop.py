"""Loop selection and absence of deprecated asyncio APIs (17.28, 17.29)."""

import ast
import asyncio
from asyncio import AbstractEventLoop
from pathlib import Path

import pytest

from local_agent.oscompat.event_loop import select_loop_factory

_SRC = Path(__file__).resolve().parents[2] / "src"

_DEPRECATED_POLICY_NAMES = frozenset(
    {
        "set_event_loop_policy",
        "get_event_loop_policy",
        "WindowsSelectorEventLoopPolicy",
        "WindowsProactorEventLoopPolicy",
        "DefaultEventLoopPolicy",
        "AbstractEventLoopPolicy",
    }
)


def _fake_uvloop_factory() -> AbstractEventLoop:
    raise AssertionError("select_loop_factory must return this factory, not call it")


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_posix_with_uvloop_uses_uvloop(platform: str) -> None:
    assert select_loop_factory(platform, _fake_uvloop_factory) is _fake_uvloop_factory


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_posix_without_uvloop_uses_asyncio_event_loop(platform: str) -> None:
    assert select_loop_factory(platform, None) is asyncio.EventLoop


def test_windows_with_uvloop_ignores_uvloop() -> None:
    assert select_loop_factory("win32", _fake_uvloop_factory) is asyncio.EventLoop


def test_windows_without_uvloop_uses_asyncio_event_loop() -> None:
    assert select_loop_factory("win32", None) is asyncio.EventLoop


def _deprecated_uses(tree: ast.AST) -> list[tuple[int, str]]:
    """Find deprecated names used as code (not in comments, docstrings or other strings)."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _DEPRECATED_POLICY_NAMES:
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            if node.attr in _DEPRECATED_POLICY_NAMES:
                found.append((node.lineno, node.attr))
            elif (
                node.attr == "iscoroutinefunction"
                and isinstance(node.value, ast.Name)
                and node.value.id == "asyncio"
            ):
                found.append((node.lineno, "asyncio.iscoroutinefunction"))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _DEPRECATED_POLICY_NAMES:
                    found.append((node.lineno, alias.name))
                elif node.module == "asyncio" and alias.name == "iscoroutinefunction":
                    found.append((node.lineno, "asyncio.iscoroutinefunction"))
    return found


def test_src_does_not_use_deprecated_asyncio_apis() -> None:
    files = sorted(_SRC.rglob("*.py"))
    assert files, f"no Python files found under {_SRC}"
    offenders = [
        f"{path.relative_to(_SRC)}:{lineno}: {name}"
        for path in files
        for lineno, name in _deprecated_uses(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert offenders == []


def test_scanner_detects_deprecated_usage() -> None:
    sample = (
        "import asyncio\n"
        "from asyncio import DefaultEventLoopPolicy, iscoroutinefunction\n"
        "asyncio.set_event_loop_policy(None)\n"
        "asyncio.iscoroutinefunction(print)\n"
        "# get_event_loop_policy in a comment is ignored\n"
        "'''WindowsProactorEventLoopPolicy in a string is ignored'''\n"
    )
    names = [name for _, name in _deprecated_uses(ast.parse(sample))]
    assert sorted(names) == [
        "DefaultEventLoopPolicy",
        "asyncio.iscoroutinefunction",
        "asyncio.iscoroutinefunction",
        "set_event_loop_policy",
    ]
