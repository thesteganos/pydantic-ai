from __future__ import annotations as _annotations

import inspect
import json
import re
from collections.abc import Awaitable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Callable, Generic, Literal, Union, cast

from pydantic import TypeAdapter, ValidationError
from pydantic_core import SchemaValidator
from typing_extensions import TypeAliasType, TypedDict, TypeVar, get_args, get_origin
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

from pydantic_ai.profiles import ModelProfile

from . import _function_schema, _utils, messages as _messages
from .exceptions import ModelRetry, UserError
from .tools import AgentDepsT, GenerateToolJsonSchema, ObjectJsonSchema, RunContext, ToolDefinition

T = TypeVar('T')
"""An invariant TypeVar."""
OutputDataT_inv = TypeVar('OutputDataT_inv', default=str)
"""
An invariant type variable for the result data of a model.

We need to use an invariant typevar for `OutputValidator` and `OutputValidatorFunc` because the output data type is used
in both the input and output of a `OutputValidatorFunc`. This can theoretically lead to some issues assuming that types
possessing OutputValidator's are covariant in the result data type, but in practice this is rarely an issue, and
changing it would have negative consequences for the ergonomics of the library.

At some point, it may make sense to change the input to OutputValidatorFunc to be `Any` or `object` as doing that would
resolve these potential variance issues.
"""
OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
"""Covariant type variable for the result data type of a run."""

OutputValidatorFunc = Union[
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], OutputDataT_inv],
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], Awaitable[OutputDataT_inv]],
    Callable[[OutputDataT_inv], OutputDataT_inv],
    Callable[[OutputDataT_inv], Awaitable[OutputDataT_inv]],
]
"""
A function that always takes and returns the same type of data (which is the result type of an agent run), and:

* may or may not take [`RunContext`][pydantic_ai.tools.RunContext] as a first argument
* may or may not be async

Usage `OutputValidatorFunc[AgentDepsT, T]`.
"""


DEFAULT_OUTPUT_TOOL_NAME = 'final_result'
DEFAULT_OUTPUT_TOOL_DESCRIPTION = 'The final response which ends this conversation'
DEFAULT_PROMPTED_JSON_PROMPT = dedent(
    """
    Always respond with a JSON object that's compatible with this schema:

    {schema}

    Don't include any text or Markdown fencing before or after.
    """
)


@dataclass
class OutputValidator(Generic[AgentDepsT, OutputDataT_inv]):
    function: OutputValidatorFunc[AgentDepsT, OutputDataT_inv]
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 1
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def validate(
        self,
        result: T,
        tool_call: _messages.ToolCallPart | None,
        run_context: RunContext[AgentDepsT],
    ) -> T:
        """Validate a result but calling the function.

        Args:
            result: The result data after Pydantic validation the message content.
            tool_call: The original tool call message, `None` if there was no tool call.
            run_context: The current run context.

        Returns:
            Result of either the validated result data (ok) or a retry message (Err).
        """
        if self._takes_ctx:
            ctx = run_context.replace_with(tool_name=tool_call.tool_name if tool_call else None)
            args = ctx, result
        else:
            args = (result,)

        try:
            if self._is_async:
                function = cast(Callable[[Any], Awaitable[T]], self.function)
                result_data = await function(*args)
            else:
                function = cast(Callable[[Any], T], self.function)
                result_data = await _utils.run_in_executor(function, *args)
        except ModelRetry as r:
            m = _messages.RetryPromptPart(content=r.message)
            if tool_call is not None:
                m.tool_name = tool_call.tool_name
                m.tool_call_id = tool_call.tool_call_id
            raise ToolRetryError(m) from r
        else:
            return result_data


class ToolRetryError(Exception):
    """Internal exception used to signal a `ToolRetry` message should be returned to the LLM."""

    def __init__(self, tool_retry: _messages.RetryPromptPart):
        self.tool_retry = tool_retry
        super().__init__()


@dataclass(init=False)
class ToolOutput(Generic[OutputDataT]):
    """Marker class to use tools for outputs, and customize the tool."""

    output_type: OutputTypeOrFunction[OutputDataT]
    name: str | None
    description: str | None
    max_retries: int | None
    strict: bool | None

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        max_retries: int | None = None,
        strict: bool | None = None,
    ):
        self.output_type = type_
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.strict = strict


@dataclass
class TextOutput(Generic[OutputDataT]):
    """Marker class to use text output with an output function."""

    output_function: TextOutputFunction[OutputDataT]


@dataclass(init=False)
class JsonSchemaOutput(Generic[OutputDataT]):
    """Marker class to use JSON schema output for outputs."""

    outputs: Sequence[OutputTypeOrFunction[OutputDataT]]
    name: str | None
    description: str | None
    strict: bool | None

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = True,
    ):
        self.outputs = flatten_output_spec(type_)
        self.name = name
        self.description = description
        self.strict = strict


class PromptedJsonOutput(Generic[OutputDataT]):
    """Marker class to use prompted JSON mode for outputs."""

    outputs: Sequence[OutputTypeOrFunction[OutputDataT]]
    name: str | None
    description: str | None

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ):
        self.outputs = flatten_output_spec(type_)
        self.name = name
        self.description = description


T_co = TypeVar('T_co', covariant=True)

OutputTypeOrFunction = TypeAliasType(
    'OutputTypeOrFunction', Union[type[T_co], Callable[..., Union[Awaitable[T_co], T_co]]], type_params=(T_co,)
)
OutputSpec = TypeAliasType(
    'OutputSpec',
    Union[
        OutputTypeOrFunction[T_co],
        ToolOutput[T_co],
        TextOutput[T_co],
        Sequence[Union[OutputTypeOrFunction[T_co], ToolOutput[T_co], TextOutput[T_co]]],
        JsonSchemaOutput[T_co],
        PromptedJsonOutput[T_co],
    ],
    type_params=(T_co,),
)

TextOutputFunction = TypeAliasType(
    'TextOutputFunction',
    Union[
        Callable[[RunContext, str], Union[Awaitable[T_co], T_co]],
        Callable[[str], Union[Awaitable[T_co], T_co]],
    ],
    type_params=(T_co,),
)

OutputMode = Literal['text', 'tool', 'tool_or_text', 'json_schema', 'prompted_json']


@dataclass(init=False)
class OutputSchema(Generic[OutputDataT]):
    """Model the final output from an agent run."""

    mode: OutputMode | None = None
    text_output_schema: (
        OutputObjectSchema[OutputDataT] | OutputUnionSchema[OutputDataT] | OutputFunctionSchema[OutputDataT] | None
    ) = None
    tools: dict[str, OutputTool[OutputDataT]] = field(default_factory=dict)

    def __init__(
        self,
        output_spec: OutputSpec[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        """Build an OutputSchema dataclass from an output type."""
        self.mode = None
        self.text_output_schema = None
        self.tools = {}

        if output_spec is str:
            self.mode = 'text'
            return

        if isinstance(output_spec, JsonSchemaOutput):
            self.mode = 'json_schema'
            self.text_output_schema = self._build_text_output_schema(
                output_spec.outputs,
                name=output_spec.name,
                description=output_spec.description,
                strict=output_spec.strict,
            )
            return

        if isinstance(output_spec, PromptedJsonOutput):
            self.mode = 'prompted_json'
            self.text_output_schema = self._build_text_output_schema(
                output_spec.outputs, name=output_spec.name, description=output_spec.description
            )
            return

        text_outputs: Sequence[type[str] | TextOutput[OutputDataT]] = []
        tool_outputs: Sequence[ToolOutput[OutputDataT]] = []
        other_outputs: Sequence[OutputTypeOrFunction[OutputDataT]] = []
        for output in flatten_output_spec(output_spec):
            if output is str:
                text_outputs.append(cast(type[str], output))
            elif isinstance(output, TextOutput):
                text_outputs.append(output)
            elif isinstance(output, ToolOutput):
                tool_outputs.append(output)
            else:
                other_outputs.append(output)

        self.tools = self._build_tools(tool_outputs + other_outputs, name=name, description=description, strict=strict)

        if len(text_outputs) > 0:
            if len(text_outputs) > 1:
                raise UserError('Only one text output is allowed.')
            text_output = text_outputs[0]

            self.mode = 'text'
            if len(self.tools) > 0:
                self.mode = 'tool_or_text'

            if isinstance(text_output, TextOutput):
                self.text_output_schema = OutputFunctionSchema(text_output.output_function)
        elif len(tool_outputs) > 0:
            self.mode = 'tool'
        elif len(other_outputs) > 0:
            self.text_output_schema = self._build_text_output_schema(
                other_outputs, name=name, description=description, strict=strict
            )
        else:
            raise UserError('No output type provided.')  # pragma: no cover

    @staticmethod
    def _build_tools(
        outputs: list[OutputTypeOrFunction[OutputDataT] | ToolOutput[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> dict[str, OutputTool[OutputDataT]]:
        tools: dict[str, OutputTool[OutputDataT]] = {}

        default_name = name or DEFAULT_OUTPUT_TOOL_NAME
        default_description = description
        default_strict = strict

        multiple = len(outputs) > 1
        for output in outputs:
            name = None
            description = None
            strict = None
            if isinstance(output, ToolOutput):
                output_type = output.output_type
                # do we need to error on conflicts here? (DavidM): If this is internal maybe doesn't matter, if public, use overloads
                name = output.name
                description = output.description
                strict = output.strict
            else:
                output_type = output

            if name is None:
                name = default_name
                if multiple:
                    name += f'_{output_type.__name__}'

            i = 1
            original_name = name
            while name in tools:
                i += 1
                name = f'{original_name}_{i}'

            description = description or default_description
            if strict is None:
                strict = default_strict

            parameters_schema = OutputObjectSchema(output=output_type, description=description, strict=strict)
            tools[name] = OutputTool(name=name, parameters_schema=parameters_schema, multiple=multiple)

        return tools

    @staticmethod
    def _build_text_output_schema(
        outputs: Sequence[OutputTypeOrFunction[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> OutputObjectSchema[OutputDataT] | OutputUnionSchema[OutputDataT] | None:
        if len(outputs) == 0:
            return None  # pragma: no cover

        outputs = flatten_output_spec(outputs)
        if len(outputs) == 1:
            return OutputObjectSchema(output=outputs[0], name=name, description=description, strict=strict)

        return OutputUnionSchema(outputs=outputs, strict=strict, name=name, description=description)

    @property
    def allow_text_output(self) -> Literal['plain', 'json', False]:
        """Whether the model allows text output."""
        if self.mode == 'tool':
            return False
        if self.mode in ('text', 'tool_or_text'):
            return 'plain'
        return 'json'

    def is_mode_supported(self, profile: ModelProfile) -> bool:
        """Whether the model supports the output mode."""
        mode = self.mode
        if mode in ('text', 'prompted_json'):
            return True
        if self.mode == 'tool_or_text':
            mode = 'tool'
        return mode in profile.output_modes

    def find_named_tool(
        self, parts: Iterable[_messages.ModelResponsePart], tool_name: str
    ) -> tuple[_messages.ToolCallPart, OutputTool[OutputDataT]] | None:
        """Find a tool that matches one of the calls, with a specific name."""
        for part in parts:  # pragma: no branch
            if isinstance(part, _messages.ToolCallPart):  # pragma: no branch
                if part.tool_name == tool_name:
                    return part, self.tools[tool_name]

    def find_tool(
        self,
        parts: Iterable[_messages.ModelResponsePart],
    ) -> Iterator[tuple[_messages.ToolCallPart, OutputTool[OutputDataT]]]:
        """Find a tool that matches one of the calls."""
        for part in parts:
            if isinstance(part, _messages.ToolCallPart):  # pragma: no branch
                if result := self.tools.get(part.tool_name):
                    yield part, result

    def tool_names(self) -> list[str]:
        """Return the names of the tools."""
        return list(self.tools.keys())

    def tool_defs(self) -> list[ToolDefinition]:
        """Get tool definitions to register with the model."""
        if self.mode not in ('tool', 'tool_or_text'):
            return []
        return [t.tool_def for t in self.tools.values()]

    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            text: The output text to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        assert self.allow_text_output is not False

        if self.text_output_schema is None:
            return cast(OutputDataT, text)

        def strip_markdown_fences(text: str) -> str:
            if text.startswith('{'):
                return text

            regex = r'```(?:\w+)?\n(\{.*\})\n```'
            match = re.search(regex, text, re.DOTALL)
            if match:
                return match.group(1)

            return text

        text = strip_markdown_fences(text)

        return await self.text_output_schema.process(
            text, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass
class OutputObjectDefinition:
    json_schema: ObjectJsonSchema
    name: str | None = None
    description: str | None = None
    strict: bool | None = None

    @property
    def instructions(self) -> str:
        """Get instructions for model to output manual JSON matching the schema."""
        schema = self.json_schema.copy()
        if self.name:
            schema['title'] = self.name
        if self.description:
            schema['description'] = self.description

        # Eventually move DEFAULT_PROMPTED_JSON_PROMPT to ModelProfile so it can be tweaked on a per model basis
        return DEFAULT_PROMPTED_JSON_PROMPT.format(schema=json.dumps(schema))


@dataclass(init=False)
class OutputUnionDataEntry:
    kind: str
    data: dict[str, Any]


@dataclass(init=False)
class OutputUnionData:
    result: OutputUnionDataEntry


# TODO: Better class naming
@dataclass(init=False)
class OutputUnionSchema(Generic[OutputDataT]):
    object_def: OutputObjectDefinition
    _root_object_schema: OutputObjectSchema[OutputUnionData]
    _object_schemas: dict[str, OutputObjectSchema[OutputDataT]]

    def __init__(
        self,
        outputs: Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        json_schemas: list[ObjectJsonSchema] = []
        self._object_schemas = {}
        for output in outputs:
            object_schema = OutputObjectSchema(output=output, strict=strict)
            object_def = object_schema.object_def

            object_key = object_def.name or output.__name__
            i = 1
            original_key = object_key
            while object_key in self._object_schemas:
                i += 1
                object_key = f'{original_key}_{i}'

            self._object_schemas[object_key] = object_schema

            json_schema = object_def.json_schema
            if object_name := object_def.name:
                json_schema['title'] = object_name
            if object_description := object_def.description:
                json_schema['description'] = object_description

            json_schemas.append(json_schema)

        json_schemas, all_defs = _utils.merge_json_schema_defs(json_schemas)

        discriminated_json_schemas: list[ObjectJsonSchema] = []
        for object_key, json_schema in zip(self._object_schemas.keys(), json_schemas):
            title = json_schema.pop('title', None)
            description = json_schema.pop('description', None)

            discriminated_json_schema = {
                'type': 'object',
                'properties': {
                    'kind': {
                        'type': 'string',
                        'const': object_key,
                    },
                    'data': json_schema,
                },
                'required': ['kind', 'data'],
                'additionalProperties': False,
            }
            if title:
                discriminated_json_schema['title'] = title
            if description:
                discriminated_json_schema['description'] = description

            discriminated_json_schemas.append(discriminated_json_schema)

        self._root_object_schema = OutputObjectSchema(output=OutputUnionData)

        json_schema = {
            'type': 'object',
            'properties': {
                'result': {
                    'anyOf': discriminated_json_schemas,
                }
            },
            'required': ['result'],
            'additionalProperties': False,
        }
        if all_defs:
            json_schema['$defs'] = all_defs

        self.object_def = OutputObjectDefinition(
            json_schema=json_schema,
            strict=strict,
            name=name,
            description=description,
        )

    async def process(
        self,
        data: str | dict[str, Any] | None,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        result = await self._root_object_schema.process(
            data, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )

        result = result.result
        kind = result.kind
        data = result.data
        try:
            object_schema = self._object_schemas[kind]
        except KeyError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(content=f'Invalid kind: {kind}')
                raise ToolRetryError(m) from e
            else:
                raise  # pragma: lax no cover

        return await object_schema.process(
            data, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass(init=False)
class OutputObjectSchema(Generic[OutputDataT]):
    object_def: OutputObjectDefinition
    outer_typed_dict_key: str | None = None
    _validator: SchemaValidator
    _function_schema: _function_schema.FunctionSchema | None = None

    def __init__(
        self,
        output: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        if inspect.isfunction(output) or inspect.ismethod(output):
            self._function_schema = _function_schema.function_schema(output, GenerateToolJsonSchema)
            self._validator = self._function_schema.validator
            json_schema = self._function_schema.json_schema
            json_schema['description'] = self._function_schema.description
        else:
            type_adapter: TypeAdapter[Any]
            if _utils.is_model_like(output):
                type_adapter = TypeAdapter(output)
            else:
                self.outer_typed_dict_key = 'response'
                response_data_typed_dict = TypedDict(  # noqa: UP013
                    'response_data_typed_dict',
                    {'response': cast(type[OutputDataT], output)},  # pyright: ignore[reportInvalidTypeForm]
                )
                type_adapter = TypeAdapter(response_data_typed_dict)

            # Really a PluggableSchemaValidator, but it's API-compatible
            self._validator = cast(SchemaValidator, type_adapter.validator)
            json_schema = _utils.check_object_json_schema(
                type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
            )

            if self.outer_typed_dict_key:
                # including `response_data_typed_dict` as a title here doesn't add anything and could confuse the LLM
                json_schema.pop('title')

        if json_schema_description := json_schema.pop('description', None):
            if description is None:
                description = json_schema_description
            else:
                description = f'{description}. {json_schema_description}'

        self.object_def = OutputObjectDefinition(
            name=name or getattr(output, '__name__', None),
            description=description,
            json_schema=json_schema,
            strict=strict,
        )

    async def process(
        self,
        data: str | dict[str, Any] | None,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message, performing validation and (if necessary) calling the output function.

        Args:
            data: The output data to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
            if isinstance(data, str):
                output = self._validator.validate_json(data or '{}', allow_partial=pyd_allow_partial)
            else:
                output = self._validator.validate_python(data or {}, allow_partial=pyd_allow_partial)
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=e.errors(include_url=False),
                )
                raise ToolRetryError(m) from e
            else:
                raise  # pragma: lax no cover

        if k := self.outer_typed_dict_key:
            output = output[k]

        if self._function_schema:
            try:
                output = await self._function_schema.call(output, run_context)
            except ModelRetry as r:
                if wrap_validation_errors:
                    m = _messages.RetryPromptPart(
                        content=r.message,
                    )
                    raise ToolRetryError(m) from r
                else:
                    raise  # pragma: lax no cover

        return output


@dataclass(init=False)
class OutputFunctionSchema(Generic[OutputDataT]):
    _function_schema: _function_schema.FunctionSchema
    _str_argument_name: str

    def __init__(
        self,
        output_function: TextOutputFunction[OutputDataT],
    ):
        if inspect.isfunction(output_function) or inspect.ismethod(output_function):
            self._function_schema = _function_schema.function_schema(output_function, GenerateToolJsonSchema)

            arguments_schema = self._function_schema.json_schema.get('properties', {})
            argument_name = next(iter(arguments_schema.keys()), None)
            if argument_name and arguments_schema.get(argument_name, {}).get('type') == 'string':
                self._str_argument_name = argument_name
                return

        raise UserError('TextOutput must take a function taking a `str`')

    @property
    def object_def(self) -> None:
        return None

    async def process(
        self,
        data: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        args = {self._str_argument_name: data}

        try:
            output = await self._function_schema.call(args, run_context)
        except ModelRetry as r:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=r.message,
                )
                raise ToolRetryError(m) from r
            else:
                raise  # pragma: lax no cover

        return cast(OutputDataT, output)


@dataclass(init=False)
class OutputTool(Generic[OutputDataT]):
    parameters_schema: OutputObjectSchema[OutputDataT]
    tool_def: ToolDefinition

    def __init__(self, *, name: str, parameters_schema: OutputObjectSchema[OutputDataT], multiple: bool):
        self.parameters_schema = parameters_schema
        definition = parameters_schema.object_def

        description = definition.description
        if not description:
            description = DEFAULT_OUTPUT_TOOL_DESCRIPTION
            if multiple:
                description = f'{definition.name}: {description}'

        self.tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters_json_schema=definition.json_schema,
            strict=definition.strict,
            outer_typed_dict_key=parameters_schema.outer_typed_dict_key,
        )

    async def process(
        self,
        tool_call: _messages.ToolCallPart,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message.

        Args:
            tool_call: The tool call from the LLM to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            output = await self.parameters_schema.process(
                tool_call.args, run_context, allow_partial=allow_partial, wrap_validation_errors=False
            )
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    tool_name=tool_call.tool_name,
                    content=e.errors(include_url=False, include_context=False),
                    tool_call_id=tool_call.tool_call_id,
                )
                raise ToolRetryError(m) from e
            else:
                raise  # pragma: lax no cover
        except ModelRetry as r:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    tool_name=tool_call.tool_name,
                    content=r.message,
                    tool_call_id=tool_call.tool_call_id,
                )
                raise ToolRetryError(m) from r
            else:
                raise  # pragma: lax no cover
        else:
            return output


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `output_type` is a union, otherwise return an empty tuple."""
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    origin = get_origin(tp)
    if is_union_origin(origin):
        return get_args(tp)
    else:
        return ()


def flatten_output_spec(output_spec: T | Sequence[T]) -> list[T]:
    outputs: Sequence[T]
    if isinstance(output_spec, Sequence):
        outputs = output_spec
    else:
        outputs = (output_spec,)

    outputs_flat: list[T] = []
    for output in outputs:
        if union_types := get_union_args(output):
            outputs_flat.extend(union_types)
        else:
            outputs_flat.append(output)
    return outputs_flat
