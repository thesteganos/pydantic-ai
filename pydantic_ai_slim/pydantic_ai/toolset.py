from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Generic, Self

from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import SchemaValidator, core_schema

from ._output import BaseOutputSchema
from ._run_context import AgentDepsT, RunContext
from .tools import (
    DocstringFormat,
    GenerateToolJsonSchema,
    Tool,
    ToolDefinition,
    ToolFuncEither,
    ToolParams,
    ToolPrepareFunc,
    ToolsPrepareFunc,
)

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer


class BaseToolset(ABC, Generic[AgentDepsT]):
    """A toolset is a collection of tools that can be used by an agent.

    It is responsible for:
    - Listing the tools it contains
    - Validating the arguments of the tools
    - Calling the tools
    """

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return None

    @abstractmethod
    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        raise NotImplementedError()

    async def list_tool_names(self, ctx: RunContext[AgentDepsT]) -> set[str]:
        return {tool_def.name for tool_def in await self.list_tool_defs(ctx)}

    @abstractmethod
    async def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        raise NotImplementedError()

    async def validate_tool_args(
        self, ctx: RunContext[AgentDepsT], name: str, args: str | dict[str, Any] | None
    ) -> dict[str, Any]:
        validator = await self.get_tool_args_validator(ctx, name)
        if isinstance(args, str):
            return validator.validate_json(args or '{}')
        else:
            return validator.validate_python(args or {})

    @abstractmethod
    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, args: dict[str, Any]) -> Any:
        raise NotImplementedError()

    async def call_tools(self, ctx: RunContext[AgentDepsT], calls: list[tuple[str, dict[str, Any]]]) -> list[Any]:
        return await asyncio.gather(*[self.call_tool(ctx, name, args) for name, args in calls])


@dataclass(init=False)
class FunctionToolset(BaseToolset[AgentDepsT]):
    """A toolset that functions can be registered to as tools."""

    max_retries: int = field(default=1)
    _tools: dict[str, Tool[Any]] = field(default_factory=dict)

    def __init__(self, tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]], max_retries: int = 1):
        self.max_retries = max_retries
        self._tools = {}
        for tool in tools:
            if isinstance(tool, Tool):
                self.register_tool(tool)
            else:
                self.register_function(tool)

    def register_function(
        self,
        func: ToolFuncEither[AgentDepsT, ToolParams],
        takes_ctx: bool | None = None,
        name: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
    ) -> None:
        """Register a function as a tool."""
        retries_ = retries if retries is not None else self.max_retries
        tool = Tool[AgentDepsT](
            func,
            takes_ctx=takes_ctx,
            name=name,
            max_retries=retries_,
            prepare=prepare,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            schema_generator=schema_generator,
            strict=strict,
        )
        self.register_tool(tool)

    def register_tool(self, tool: Tool[AgentDepsT]) -> None:
        if tool.name in self._tools:
            raise ValueError(f'Tool name conflicts with existing tool: {tool.name!r}')
        if tool.max_retries is None:
            tool = replace(tool, max_retries=self.max_retries)
        self._tools[tool.name] = tool

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return [tool_def for tool in self._tools.values() if (tool_def := await tool.prepare_tool_def(ctx))]

    async def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self._tools[name].function_schema.validator

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, args: dict[str, Any]) -> Any:
        # TODO: This won't work if prepare_tool_def renamed the tool -- I guess PreparedToolset should maintain the map. Just like prefixed?
        return await self._tools[name].function_schema.call(args, ctx)


@dataclass
class OutputToolset(BaseToolset[AgentDepsT]):
    """A toolset that contains output tools."""

    output_schema: BaseOutputSchema[Any]
    max_retries: int = field(default=1)

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return [tool.tool_def for tool in self.output_schema.tools.values()]

    async def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        # TODO: Should never be called for an output tool?
        return self.output_schema.tools[name].processor._validator  # pyright: ignore[reportPrivateUsage]

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, args: dict[str, Any]) -> Any:
        # TODO: Should never be called for an output tool?
        return await self.output_schema.tools[name].processor.process(args, ctx)


@dataclass
class MCPToolset(BaseToolset[AgentDepsT]):
    """A toolset that contains MCP tools and handles running the server."""

    server: MCPServer
    max_retries: int = field(default=1)

    async def __aenter__(self) -> Self:
        await self.server.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return await self.server.__aexit__(exc_type, exc_value, traceback)

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        tools = await self.server.list_tools()
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description or '',
                parameters_json_schema=tool.inputSchema,
            )
            for tool in tools
        ]

    async def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return SchemaValidator(schema=core_schema.dict_schema(core_schema.str_schema(), core_schema.any_schema()))

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, args: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> Any:
        return await self.server.call_tool(name, args, metadata)


@dataclass
class WrapperToolset(BaseToolset[AgentDepsT]):
    """A toolset that wraps another toolset and delegates to it."""

    wrapped: BaseToolset[AgentDepsT]

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return await self.wrapped.__aexit__(exc_type, exc_value, traceback)

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return await self.wrapped.list_tool_defs(ctx)

    async def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return await self.wrapped.get_tool_args_validator(ctx, name)

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, args: dict[str, Any]) -> Any:
        return await self.wrapped.call_tool(ctx, name, args)

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)  # pragma: no cover


@dataclass(init=False)
class CombinedToolset(BaseToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets."""

    toolsets: list[BaseToolset[AgentDepsT]]
    _exit_stack: AsyncExitStack

    def __init__(self, toolsets: list[BaseToolset[AgentDepsT]]):
        self.toolsets = toolsets

    async def __aenter__(self) -> Self:
        self._exit_stack = AsyncExitStack()
        for toolset in self.toolsets:
            await self._exit_stack.enter_async_context(toolset)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        await self._exit_stack.aclose()

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return [tool_def for toolset in self.toolsets for tool_def in await toolset.list_tool_defs(ctx)]

    async def list_tool_names(self, ctx: RunContext[AgentDepsT]) -> set[str]:
        names: set[str] = set()
        for toolset in self.toolsets:
            new_names = await toolset.list_tool_names(ctx)
            if duplicates := new_names & names:
                raise ValueError(f'Toolset {toolset} has conflicting tool names: {duplicates}')
            names.update(new_names)
        return names

    async def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        for toolset in self.toolsets:
            if name in await toolset.list_tool_names(ctx):
                return await toolset.get_tool_args_validator(ctx, name)
        raise ValueError(f'Tool {name} not found in any toolset')

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, args: dict[str, Any]) -> Any:
        for toolset in self.toolsets:
            if name in await toolset.list_tool_names(ctx):
                return await toolset.call_tool(ctx, name, args)
        raise ValueError(f'Tool {name} not found in any toolset')

    async def call_tools(self, ctx: RunContext[AgentDepsT], calls: list[tuple[str, dict[str, Any]]]) -> list[Any]:
        toolset_per_tool_name = await self._toolset_per_tool_name(ctx)
        calls_per_toolset: defaultdict[BaseToolset[AgentDepsT], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for name, args in calls:
            try:
                toolset = toolset_per_tool_name[name]
            except KeyError as e:
                raise ValueError(f'Tool {name} not found in any toolset') from e

            calls_per_toolset[toolset].append((name, args))
        results_per_toolset = await asyncio.gather(
            *[toolset.call_tools(ctx, calls) for toolset, calls in calls_per_toolset.items()]
        )
        return [result for results in results_per_toolset for result in results]

    async def _toolset_per_tool_name(self, ctx: RunContext[AgentDepsT]) -> dict[str, BaseToolset[AgentDepsT]]:
        # TODO: Cache this somehow, even though names may change...
        return {name: toolset for toolset in self.toolsets for name in await toolset.list_tool_names(ctx)}


@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains."""

    prefix: str

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return [
            replace(tool_def, name=self._prefixed_tool_name(tool_def.name))
            for tool_def in await super().list_tool_defs(ctx)
        ]

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, args: dict[str, Any]) -> Any:
        return await super().call_tool(ctx, self._unprefixed_tool_name(name), args)

    def _prefixed_tool_name(self, tool_name: str) -> str:
        return f'{self.prefix}_{tool_name}'

    def _unprefixed_tool_name(self, tool_name: str) -> str:
        full_prefix = f'{self.prefix}_'
        if not tool_name.startswith(full_prefix):
            raise ValueError(f"Tool name {tool_name} does not start with prefix '{full_prefix}'")
        return tool_name[len(full_prefix) :]


@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a prepare function."""

    prepare: ToolsPrepareFunc[AgentDepsT]

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return await self.prepare(ctx, await super().list_tool_defs(ctx)) or []


@dataclass(init=False)
class PrepareSingleToolset(PreparedToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a per-tool prepare function."""

    def __init__(self, toolset: BaseToolset[AgentDepsT], prepare: ToolPrepareFunc[AgentDepsT]):
        async def prepare_tool_defs(
            ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:
            return [new_tool_def for tool_def in tool_defs if (new_tool_def := await prepare(ctx, tool_def))]

        super().__init__(toolset, prepare_tool_defs)


@dataclass(init=False)
class FilteredToolset(PrepareSingleToolset[AgentDepsT]):
    """A toolset that filters the tools it contains using a filter function."""

    def __init__(
        self, toolset: BaseToolset[AgentDepsT], filter: Callable[[RunContext[AgentDepsT], ToolDefinition], bool]
    ):
        async def filter_tool_def(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
            return tool_def if filter(ctx, tool_def) else None

        super().__init__(toolset, filter_tool_def)


CallToolFunc = Callable[
    [str, dict[str, Any]], Awaitable[Any]
]  # TODO: What if it takes more args, like metadata in case of MCP?
"""A function type that represents a tool call."""

ToolsProcessFunc = Callable[
    [
        RunContext[AgentDepsT],
        CallToolFunc,
        str,
        dict[str, Any],
    ],
    Awaitable[Any],
]


@dataclass
class ProcessedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that lets the tool call arguments and return value be customized using a process function."""

    process: ToolsProcessFunc[AgentDepsT]

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, args: dict[str, Any]) -> Any:
        return await self.process(ctx, partial(self.wrapped.call_tool, ctx), name, args)
