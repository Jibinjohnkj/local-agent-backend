"""The Backend's own `stdio` transport for MCP_Servers (6.7, 6.11, 17.11, 17.12, 17.27).

`managed_stdio_client` yields the same `(read_stream, write_stream)` pair of `SessionMessage`
streams as the SDK's `stdio_client`, so it feeds a normal `ClientSession`. Two things differ from
the SDK transport:

- The process is started and stopped by `oscompat.process` (`spawn`, `terminate_tree`), which owns
  the process group or job object, so no process sits in two Backend-managed jobs.
- Framing is bytes-level. Output is split on `b"\\n"` before any decoding, one trailing `b"\\r"`
  is dropped, and every line is decoded as strict UTF-8 whatever the locale is (17.27).

The SDK's message framing is kept: one JSON-RPC message per line, validated by the SDK's JSON-RPC
adapter and written with `model_dump_json(by_alias=True, exclude_unset=True)`.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import anyio
import anyio.abc
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from local_agent.logging_setup import get_logger
from local_agent.mcp_client.mcp_compat import SessionMessage, jsonrpc_message_adapter
from local_agent.oscompat.process import SpawnSpec, TreeHandle, spawn, terminate_tree

__all__ = ["decode_line", "encode_line", "managed_stdio_client"]

# Seconds the writer gets to flush accepted messages before the server's stdin closes.
_WRITER_FLUSH_TIMEOUT_S = 0.5

_NEWLINE = b"\n"
_CARRIAGE_RETURN = b"\r"

_log = get_logger("mcp_client.stdio_transport")

# A pipe that was closed under a read or a write, by the peer or by our own shutdown.
_PIPE_ERRORS = (anyio.ClosedResourceError, anyio.BrokenResourceError, OSError, ValueError)


def encode_line(msg: SessionMessage) -> bytes:
    """The wire form of one message: compact JSON, UTF-8, ended by a single `\\n` (17.27).

    Pure. `by_alias` and `exclude_unset` match the SDK transport, so a field the caller never set
    stays off the wire.
    """
    return (msg.message.model_dump_json(by_alias=True, exclude_unset=True) + "\n").encode("utf-8")


def decode_line(line: bytes) -> SessionMessage | Exception:
    """Parse one line (without its terminator) into a message, or return the failure (17.27).

    Pure. The bytes are decoded as strict UTF-8, so an invalid sequence is a failure instead of a
    replacement character. A failure is returned as a value, not raised: `ClientSession` surfaces
    an `Exception` item from the read stream to its message handler and carries on, as it does
    with the SDK transport.
    """
    try:
        text = line.decode("utf-8")
        message = jsonrpc_message_adapter.validate_json(text, by_name=False)
    except ValueError as exc:
        # `UnicodeDecodeError` and pydantic's `ValidationError` are both `ValueError`.
        return exc
    return SessionMessage(message)


def _split_lines(buffer: bytearray) -> list[bytes]:
    """Remove every complete line from `buffer` and return them, framing characters dropped.

    Lines end at `b"\\n"`. One trailing `b"\\r"` is dropped for servers that write CRLF. Bytes
    after the last newline stay in `buffer` until more output arrives.
    """
    lines: list[bytes] = []
    while (index := buffer.find(_NEWLINE)) >= 0:
        end = index - 1 if index > 0 and buffer[index - 1 : index] == _CARRIAGE_RETURN else index
        lines.append(bytes(buffer[:end]))
        del buffer[: index + 1]
    return lines


async def _drain(stream: anyio.abc.ByteReceiveStream) -> None:
    """Read and discard the server's remaining stdout until it closes.

    A server that is still flushing output cannot read its stdin or exit while its stdout pipe is
    full, so this keeps the graceful step of `terminate_tree` from waiting on a blocked child.
    """
    try:
        async for _ in stream:
            pass
    except _PIPE_ERRORS:
        pass


@asynccontextmanager
async def managed_stdio_client(
    spec: SpawnSpec, *, grace: Callable[[], float], server: str | None = None
) -> AsyncIterator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]
]:
    """Start the server described by `spec` and yield `(read_stream, write_stream)` (6.7).

    `grace` returns the grace period in seconds for `terminate_tree`. `terminate_tree` calls it
    while it waits, not when the context starts, so a shutdown that shortens the grace period
    (console close, or a second Shutdown_Signal) applies to servers that are already running and
    to a wait that is already under way. `server` names the server in log records.

    Leaving the context always stops the whole Process_Tree with `terminate_tree`, whether the
    body finished, raised, or was cancelled (6.11, 17.11, 17.12). The stop runs under a shielded
    scope, and every wait in it is bounded.

    The read stream ends (end of stream) when the server closes its stdout or the stdin writer
    fails, which `ClientSession` treats as a closed connection. A line that is not valid UTF-8 or
    not a JSON-RPC message arrives as an `Exception` item.

    Raises whatever `spawn` raises (`OSError` when the child cannot start, `ValueError` for a
    `SpawnSpec` built for another operating system).
    """
    handle = await spawn(spec, server=server)
    terminated = False

    async def stop() -> None:
        nonlocal terminated
        terminated = True
        await terminate_tree(handle, grace)

    try:
        async with _bridge(handle, stop) as streams:
            yield streams
    finally:
        # Reached with the tree still running only when entering the bridge failed or was
        # cancelled before its own shutdown could run.
        if not terminated:
            with anyio.CancelScope(shield=True):
                await stop()


@asynccontextmanager
async def _bridge(
    handle: TreeHandle, stop: Callable[[], Awaitable[None]]
) -> AsyncIterator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]
]:
    """Run the stdout reader and stdin writer tasks around `handle`'s pipes.

    `stop` must stop the tree. It runs on exit before the tasks are cancelled, because the reader
    only ends once the child's stdout reaches end of file.
    """
    read_send, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    write_stream, write_receive = anyio.create_memory_object_stream[SessionMessage](0)
    writer_done = anyio.Event()

    async def stdout_reader() -> None:
        buffer = bytearray()
        try:
            async with read_send:
                try:
                    async for chunk in handle.stdout:
                        buffer.extend(chunk)
                        for line in _split_lines(buffer):
                            item = decode_line(line)
                            if isinstance(item, Exception):
                                _log.debug(
                                    "mcp_server_bad_line",
                                    server=handle.server,
                                    size=len(line),
                                    reason=type(item).__name__,
                                )
                            try:
                                await read_send.send(item)
                            except anyio.ClosedResourceError, anyio.BrokenResourceError:
                                # The session is gone; only the drain below remains.
                                return
                finally:
                    await _drain(handle.stdout)
        except _PIPE_ERRORS:
            # Our own shutdown closed the pipe under the read, or the pipe broke. Either way the
            # session sees the read stream end.
            pass

    async def stdin_writer() -> None:
        try:
            async with write_receive:
                async for message in write_receive:
                    await handle.stdin.send(encode_line(message))
        except _PIPE_ERRORS:
            # The server may still be alive; end the read stream so a pending request fails
            # instead of waiting forever.
            await read_send.aclose()
        finally:
            writer_done.set()

    async def shutdown() -> None:
        # Unblock the reader into its drain, and give the writer a bounded window to flush the
        # messages it already accepted.
        read_stream.close()
        write_stream.close()
        with anyio.move_on_after(_WRITER_FLUSH_TIMEOUT_S):
            await writer_done.wait()
        await stop()
        for stream in (read_stream, write_stream, read_send, write_receive):
            await stream.aclose()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        try:
            yield read_stream, write_stream
        finally:
            with anyio.CancelScope(shield=True):
                await shutdown()
            # Release a task that a surviving descendant's open pipe could still hold.
            tg.cancel_scope.cancel()
