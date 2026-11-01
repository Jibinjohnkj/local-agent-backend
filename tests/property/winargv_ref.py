"""Reference parsers for Windows command lines, shared by the Property 40 and 41 tests.

Two independent models live here. Neither imports anything from `local_agent`, so a bug in the
implementation cannot also hide in the oracle that checks it.

1. The Microsoft C runtime / `CommandLineToArgvW` splitter (`parse_command_line`,
   `parse_arguments`). It is a line-by-line port of the UCRT's `parse_command_line` state machine
   (post-2008 rules). Both entry points exist because the first token is special:

   - `parse_command_line(line)` splits a whole `CreateProcess` command line. The first token is
     the program name. It follows the program-name rule: a `"` toggles quoting and nothing is
     escaped, so backslashes are always literal, and the token ends at the first space or tab
     outside quotes. A program name therefore cannot hold a `"`, which matches Windows, where no
     file name may contain one.
   - `parse_arguments(line)` splits text that holds only arguments, so every token follows the
     argument rules. Task 8.5 uses it to parse the inner batch line after the batch path.

   Argument rules, applied by both:

   - `2n` backslashes followed by `"` give `n` backslashes, and the `"` toggles quoting.
   - `2n+1` backslashes followed by `"` give `n` backslashes and a literal `"`.
   - Backslashes not followed by `"` are literal.
   - Inside a quoted region, `""` gives one literal `"` and quoting continues.
   - Only space and tab separate arguments, and only outside quotes.
   - A pair of quotes with nothing between them is an empty argument.

2. The `cmd.exe` quote-state scanner (`cmd_quote_states`, `cmd_unquoted_metachars`). `cmd.exe`
   flips its quote state on every `"` and reads `& | < > ^` as operators only outside quotes. A
   `^` outside quotes escapes the next character, so an escaped `"` does not flip the state.
   Inside quotes `^` is an ordinary character.
"""

from typing import Final

__all__ = [
    "CMD_METACHARS",
    "cmd_quote_states",
    "cmd_unquoted_metachars",
    "parse_arguments",
    "parse_command_line",
]

_BLANKS: Final = " \t"

# The characters `cmd.exe` treats as operators outside quotes. Task 8.5 adds the space, which
# separates arguments outside quotes and so must stay inside quotes when it belongs to an argument.
CMD_METACHARS: Final = "&|<>^"


def _skip_blanks(line: str, pos: int) -> int:
    while pos < len(line) and line[pos] in _BLANKS:
        pos += 1
    return pos


def _program_name(line: str, pos: int) -> tuple[str, int]:
    """Read the program name starting at `pos`; return it and the position after it.

    Quotes toggle and are dropped. Nothing is escaped. The token ends at a space or tab outside
    quotes, or at the end of the line.
    """
    chars: list[str] = []
    in_quotes = False
    while pos < len(line):
        char = line[pos]
        if char == '"':
            in_quotes = not in_quotes
        elif not in_quotes and char in _BLANKS:
            break
        else:
            chars.append(char)
        pos += 1
    return "".join(chars), pos


def _argument(line: str, pos: int) -> tuple[str, int]:
    """Read one argument starting at `pos` (not on a blank); return it and the next position."""
    chars: list[str] = []
    in_quotes = False
    while True:
        backslashes = 0
        while pos < len(line) and line[pos] == "\\":
            backslashes += 1
            pos += 1
        copy_char = True
        if pos < len(line) and line[pos] == '"':
            if backslashes % 2 == 0:
                if in_quotes and line[pos + 1 : pos + 2] == '"':
                    # `""` inside quotes: skip the first quote and copy the second.
                    pos += 1
                else:
                    copy_char = False
                    in_quotes = not in_quotes
            backslashes //= 2
        chars.append("\\" * backslashes)
        if pos >= len(line) or (not in_quotes and line[pos] in _BLANKS):
            return "".join(chars), pos
        if copy_char:
            chars.append(line[pos])
        pos += 1


def _split_arguments(line: str, pos: int) -> list[str]:
    arguments: list[str] = []
    while True:
        pos = _skip_blanks(line, pos)
        if pos >= len(line):
            return arguments
        argument, pos = _argument(line, pos)
        arguments.append(argument)


def parse_command_line(line: str) -> list[str]:
    """Split a whole command line the way the C runtime fills `argv`; the first item is `argv[0]`.

    An empty or blank line has no program name and gives an empty list.
    """
    pos = _skip_blanks(line, 0)
    if pos >= len(line):
        return []
    program, pos = _program_name(line, pos)
    return [program, *_split_arguments(line, pos)]


def parse_arguments(line: str) -> list[str]:
    """Split text that holds only arguments, applying the argument rules to every token."""
    return _split_arguments(line, 0)


def cmd_quote_states(line: str) -> tuple[bool, ...]:
    """For each character of `line`, say whether `cmd.exe` reads it inside a quoted region.

    The delimiting `"` characters are outside the region (`False`). A region left open at the end
    of the line runs to the end. A character escaped by `^` outside quotes is a plain character,
    so it is `False` and, when it is a `"`, it does not open a region.
    """
    states: list[bool] = []
    in_quotes = False
    escaped = False
    for char in line:
        if escaped:
            escaped = False
            states.append(False)
        elif char == '"':
            in_quotes = not in_quotes
            states.append(False)
        elif char == "^" and not in_quotes:
            escaped = True
            states.append(False)
        else:
            states.append(in_quotes)
    return tuple(states)


def cmd_unquoted_metachars(line: str) -> list[tuple[int, str]]:
    """The `(index, char)` pairs of `& | < > ^` that `cmd.exe` would read as operators.

    These are the characters outside every quoted region. The `^` that escapes the next character
    is reported too, because it is itself the operator.
    """
    states = cmd_quote_states(line)
    return [
        (index, char)
        for index, char in enumerate(line)
        if char in CMD_METACHARS and not states[index]
    ]
