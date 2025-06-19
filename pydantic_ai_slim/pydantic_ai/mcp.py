from __future__ import annotations

import base64
import functools
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, cast

import anyio
import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
from mcp.shared.exceptions import McpError
from mcp.shared.message import SessionMessage
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ClientRequest,
    Content,
    EmbeddedResource,
    ImageContent,
    LoggingLevel,
    RequestParams,
    TextContent,
    TextResourceContents,
    Tool as MCPTool,
)
from typing_extensions import Self, assert_never, deprecated

from ._run_context import RunContext
from .exceptions import ModelRetry, UserError
from .messages import BinaryContent
from .toolset import BaseToolset, MCPToolset, PrefixedToolset, ProcessedToolset, ToolsProcessFunc

try:
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use the MCP server, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error

__all__ = 'MCPServer', 'MCPServerStdio', 'MCPServerHTTP', 'MCPServerSSE', 'MCPServerStreamableHTTP'


class MCPServer(ABC):
    """Base class for attaching agents to MCP servers.

    See <https://modelcontextprotocol.io> for more information.
    """

    is_running: bool = False
    tool_prefix: str | None = None
    """A prefix to add to all tools that are registered with the server.

    If not empty, will include a trailing underscore(`_`).

    e.g. if `tool_prefix='foo'`, then a tool named `bar` will be registered as `foo_bar`
    """

    process_tool_call: ProcessToolCallback | None = None
    """Hook to customize tool calling and optionally pass extra metadata."""

    _client: ClientSession
    _read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    _write_stream: MemoryObjectSendStream[SessionMessage]
    _exit_stack: AsyncExitStack

    @abstractmethod
    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        """Create the streams for the MCP server."""
        raise NotImplementedError('MCP Server subclasses must implement this method.')
        yield

    @abstractmethod
    def _get_log_level(self) -> LoggingLevel | None:
        """Get the log level for the MCP server."""
        raise NotImplementedError('MCP Server subclasses must implement this method.')

    def _get_client_initialize_timeout(self) -> float:
        return 5  # pragma: no cover

    async def list_tools(self) -> list[MCPTool]:
        """Retrieve tools that are currently active on the server.

        Note:
        - We don't cache tools as they might change.
        - We also don't subscribe to the server to avoid complexity.
        """
        if not self.is_running:  # pragma: no cover
            raise UserError(f'MCP server is not running: {self}')
        result = await self._client.list_tools()
        return result.tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Call a tool on the server.

        Args:
            tool_name: The name of the tool to call.
            arguments: The arguments to pass to the tool.
            metadata: Request-level metadata (optional)

        Returns:
            The result of the tool call.

        Raises:
            ModelRetry: If the tool call fails.
        """
        if not self.is_running:  # pragma: no cover
            raise UserError(f'MCP server is not running: {self}')
        try:
            # meta param is not provided by session yet, so build and can send_request directly.
            result = await self._client.send_request(
                ClientRequest(
                    CallToolRequest(
                        method='tools/call',
                        params=CallToolRequestParams(
                            name=tool_name,
                            arguments=arguments,
                            _meta=RequestParams.Meta(**metadata) if metadata else None,
                        ),
                    )
                ),
                CallToolResult,
            )
        except McpError as e:
            raise ModelRetry(e.error.message)

        content = [self._map_tool_result_part(part) for part in result.content]

        if result.isError:
            text = '\n'.join(str(part) for part in content)
            raise ModelRetry(text)

        if len(content) == 1:
            return content[0]
        return content

    def as_toolset(self, max_retries: int = 1) -> BaseToolset:
        toolset = MCPToolset(self, max_retries=max_retries)
        if self.process_tool_call:
            toolset = ProcessedToolset(
                toolset, cast(ToolsProcessFunc, self.process_tool_call)
            )  # TODO: What to do with the extra metadata arg?
        if self.tool_prefix:
            toolset = PrefixedToolset(toolset, self.tool_prefix)
        return toolset

    async def __aenter__(self) -> Self:
        self._exit_stack = AsyncExitStack()

        self._read_stream, self._write_stream = await self._exit_stack.enter_async_context(self.client_streams())
        client = ClientSession(read_stream=self._read_stream, write_stream=self._write_stream)
        self._client = await self._exit_stack.enter_async_context(client)

        with anyio.fail_after(self._get_client_initialize_timeout()):
            await self._client.initialize()

        if log_level := self._get_log_level():
            await self._client.set_logging_level(log_level)
        self.is_running = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        await self._exit_stack.aclose()
        self.is_running = False

    def _map_tool_result_part(self, part: Content) -> str | BinaryContent | dict[str, Any] | list[Any]:
        # See https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#return-values

        if isinstance(part, TextContent):
            text = part.text
            if text.startswith(('[', '{')):
                try:
                    return json.loads(text)
                except ValueError:
                    pass
            return text
        elif isinstance(part, ImageContent):
            return BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)
        elif isinstance(part, AudioContent):
            # NOTE: The FastMCP server doesn't support audio content.
            # See <https://github.com/modelcontextprotocol/python-sdk/issues/952> for more details.
            return BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)  # pragma: no cover
        elif isinstance(part, EmbeddedResource):
            resource = part.resource
            if isinstance(resource, TextResourceContents):
                return resource.text
            elif isinstance(resource, BlobResourceContents):
                return BinaryContent(
                    data=base64.b64decode(resource.blob),
                    media_type=resource.mimeType or 'application/octet-stream',
                )
            else:
                assert_never(resource)
        else:
            assert_never(part)


@dataclass
class MCPServerStdio(MCPServer):
    """Runs an MCP server in a subprocess and communicates with it over stdin/stdout.

    This class implements the stdio transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio> for more information.

    !!! note
        Using this class as an async context manager will start the server as a subprocess when entering the context,
        and stop it when exiting the context.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio

    server = MCPServerStdio(  # (1)!
        'deno',
        args=[
            'run',
            '-N',
            '-R=node_modules',
            '-W=node_modules',
            '--node-modules-dir=auto',
            'jsr:@pydantic/mcp-run-python',
            'stdio',
        ]
    )
    agent = Agent('openai:gpt-4o', mcp_servers=[server])

    async def main():
        async with agent.run_toolsets():  # (2)!
            ...
    ```

    1. See [MCP Run Python](../mcp/run-python.md) for more information.
    2. This will start the server as a subprocess and connect to it.
    """

    command: str
    """The command to run."""

    args: Sequence[str]
    """The arguments to pass to the command."""

    env: dict[str, str] | None = None
    """The environment variables the CLI server will have access to.

    By default the subprocess will not inherit any environment variables from the parent process.
    If you want to inherit the environment variables from the parent process, use `env=os.environ`.
    """
    log_level: LoggingLevel | None = None
    """The log level to set when connecting to the server, if any.

    See <https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/logging#logging> for more details.

    If `None`, no log level will be set.
    """

    cwd: str | Path | None = None
    """The working directory to use when spawning the process."""

    tool_prefix: str | None = None
    """A prefix to add to all tools that are registered with the server.

    If not empty, will include a trailing underscore(`_`).

    e.g. if `tool_prefix='foo'`, then a tool named `bar` will be registered as `foo_bar`
    """

    process_tool_call: ProcessToolCallback | None = None
    """Hook to customize tool calling and optionally pass extra metadata."""

    timeout: float = 5
    """ The timeout in seconds to wait for the client to initialize."""

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        server = StdioServerParameters(command=self.command, args=list(self.args), env=self.env, cwd=self.cwd)
        async with stdio_client(server=server) as (read_stream, write_stream):
            yield read_stream, write_stream

    def _get_log_level(self) -> LoggingLevel | None:
        return self.log_level

    def __repr__(self) -> str:
        return f'MCPServerStdio(command={self.command!r}, args={self.args!r}, tool_prefix={self.tool_prefix!r})'

    def _get_client_initialize_timeout(self) -> float:
        return self.timeout


@dataclass
class _MCPServerHTTP(MCPServer):
    url: str
    """The URL of the endpoint on the MCP server."""

    headers: dict[str, Any] | None = None
    """Optional HTTP headers to be sent with each request to the endpoint.

    These headers will be passed directly to the underlying `httpx.AsyncClient`.
    Useful for authentication, custom headers, or other HTTP-specific configurations.

    !!! note
        You can either pass `headers` or `http_client`, but not both.

        See [`MCPServerHTTP.http_client`][pydantic_ai.mcp.MCPServerHTTP.http_client] for more information.
    """

    http_client: httpx.AsyncClient | None = None
    """An `httpx.AsyncClient` to use with the endpoint.

    This client may be configured to use customized connection parameters like self-signed certificates.

    !!! note
        You can either pass `headers` or `http_client`, but not both.

        If you want to use both, you can pass the headers to the `http_client` instead.

        ```python {py="3.10" test="skip"}
        import httpx

        from pydantic_ai.mcp import MCPServerSSE

        http_client = httpx.AsyncClient(headers={'Authorization': 'Bearer ...'})
        server = MCPServerSSE('http://localhost:3001/sse', http_client=http_client)
        ```
    """

    timeout: float = 5
    """Initial connection timeout in seconds for establishing the connection.

    This timeout applies to the initial connection setup and handshake.
    If the connection cannot be established within this time, the operation will fail.
    """

    sse_read_timeout: float = 5 * 60
    """Maximum time in seconds to wait for new SSE messages before timing out.

    This timeout applies to the long-lived SSE connection after it's established.
    If no new messages are received within this time, the connection will be considered stale
    and may be closed. Defaults to 5 minutes (300 seconds).
    """

    log_level: LoggingLevel | None = None
    """The log level to set when connecting to the server, if any.

    See <https://modelcontextprotocol.io/introduction#logging> for more details.

    If `None`, no log level will be set.
    """

    tool_prefix: str | None = None
    """A prefix to add to all tools that are registered with the server.

    If not empty, will include a trailing underscore (`_`).

    For example, if `tool_prefix='foo'`, then a tool named `bar` will be registered as `foo_bar`
    """

    process_tool_call: ProcessToolCallback | None = None
    """Hook to customize tool calling and optionally pass extra metadata."""

    @property
    @abstractmethod
    def _transport_client(
        self,
    ) -> Callable[
        ...,
        AbstractAsyncContextManager[
            tuple[
                MemoryObjectReceiveStream[SessionMessage | Exception],
                MemoryObjectSendStream[SessionMessage],
                GetSessionIdCallback,
            ],
        ]
        | AbstractAsyncContextManager[
            tuple[
                MemoryObjectReceiveStream[SessionMessage | Exception],
                MemoryObjectSendStream[SessionMessage],
            ]
        ],
    ]: ...

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[MemoryObjectReceiveStream[SessionMessage | Exception], MemoryObjectSendStream[SessionMessage]]
    ]:  # pragma: no cover
        if self.http_client and self.headers:
            raise ValueError('`http_client` is mutually exclusive with `headers`.')

        transport_client_partial = functools.partial(
            self._transport_client,
            url=self.url,
            timeout=self.timeout,
            sse_read_timeout=self.sse_read_timeout,
        )

        if self.http_client is not None:

            def httpx_client_factory(
                headers: dict[str, str] | None = None,
                timeout: httpx.Timeout | None = None,
                auth: httpx.Auth | None = None,
            ) -> httpx.AsyncClient:
                assert self.http_client is not None
                return self.http_client

            async with transport_client_partial(httpx_client_factory=httpx_client_factory) as (
                read_stream,
                write_stream,
                *_,
            ):
                yield read_stream, write_stream
        else:
            async with transport_client_partial(headers=self.headers) as (read_stream, write_stream, *_):
                yield read_stream, write_stream

    def _get_log_level(self) -> LoggingLevel | None:
        return self.log_level

    def __repr__(self) -> str:  # pragma: no cover
        return f'{self.__class__.__name__}(url={self.url!r}, tool_prefix={self.tool_prefix!r})'

    def _get_client_initialize_timeout(self) -> float:  # pragma: no cover
        return self.timeout


@dataclass
class MCPServerSSE(_MCPServerHTTP):
    """An MCP server that connects over streamable HTTP connections.

    This class implements the SSE transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerSSE

    server = MCPServerSSE('http://localhost:3001/sse')  # (1)!
    agent = Agent('openai:gpt-4o', mcp_servers=[server])

    async def main():
        async with agent.run_toolsets():  # (2)!
            ...
    ```

    1. E.g. you might be connecting to a server run with [`mcp-run-python`](../mcp/run-python.md).
    2. This will connect to a server running on `localhost:3001`.
    """

    @property
    def _transport_client(self):
        return sse_client  # pragma: no cover


@deprecated('The `MCPServerHTTP` class is deprecated, use `MCPServerSSE` instead.')
@dataclass
class MCPServerHTTP(MCPServerSSE):
    """An MCP server that connects over HTTP using the old SSE transport.

    This class implements the SSE transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10" test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerHTTP

    server = MCPServerHTTP('http://localhost:3001/sse')  # (1)!
    agent = Agent('openai:gpt-4o', mcp_servers=[server])

    async def main():
        async with agent.run_toolsets():  # (2)!
            ...
    ```

    1. E.g. you might be connecting to a server run with [`mcp-run-python`](../mcp/run-python.md).
    2. This will connect to a server running on `localhost:3001`.
    """


@dataclass
class MCPServerStreamableHTTP(_MCPServerHTTP):
    """An MCP server that connects over HTTP using the Streamable HTTP transport.

    This class implements the Streamable HTTP transport from the MCP specification.
    See <https://modelcontextprotocol.io/introduction#streamable-http> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStreamableHTTP

    server = MCPServerStreamableHTTP('http://localhost:8000/mcp')  # (1)!
    agent = Agent('openai:gpt-4o', mcp_servers=[server])

    async def main():
        async with agent.run_toolsets():  # (2)!
            ...
    ```
    """

    @property
    def _transport_client(self):
        return streamablehttp_client  # pragma: no cover


ToolResult = (
    str | BinaryContent | dict[str, Any] | list[Any] | Sequence[str | BinaryContent | dict[str, Any] | list[Any]]
)
"""The result type of a tool call."""

CallToolFunc = Callable[[str, dict[str, Any], dict[str, Any] | None], Awaitable[ToolResult]]
"""A function type that represents a tool call."""

ProcessToolCallback = Callable[
    [
        RunContext[Any],
        CallToolFunc,
        str,
        dict[str, Any],
    ],
    Awaitable[ToolResult],
]
"""A process tool callback.

It accepts a run context, the original tool call function, a tool name, and arguments.

Allows wrapping an MCP server tool call to customize it, including adding extra request
metadata.
"""
