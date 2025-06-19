"""Tests for the MCP (Model Context Protocol) server implementation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final
from unittest.mock import AsyncMock, patch

import pytest
from inline_snapshot import snapshot

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import Usage

from .conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from mcp import ErrorData, McpError

    from pydantic_ai.mcp import CallToolFunc, MCPServerSSE, MCPServerStdio, ToolResult
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

TOOL_COUNT: Final[int] = 12

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mcp and openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture
def mcp_server() -> MCPServerStdio:
    return MCPServerStdio('python', ['-m', 'tests.mcp_server'])


@pytest.fixture
def model(openai_api_key: str) -> Model:
    return OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))


@pytest.fixture
def agent(model: Model, mcp_server: MCPServerStdio) -> Agent:
    return Agent(model, mcp_servers=[mcp_server])


@pytest.fixture
def run_context(model: Model) -> RunContext[None]:
    return RunContext(deps=None, model=model, usage=Usage())


async def test_stdio_server(run_context: RunContext[None]):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    async with server:
        tools = await server.as_toolset().list_tool_defs(run_context)
        assert len(tools) == TOOL_COUNT
        assert tools[0].name == 'celsius_to_fahrenheit'
        assert tools[0].description.startswith('Convert Celsius to Fahrenheit.')

        # Test calling the temperature conversion tool
        result = await server.call_tool('celsius_to_fahrenheit', {'celsius': 0})
        assert result == snapshot('32.0')


async def test_stdio_server_with_tool_prefix(run_context: RunContext[None]):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], tool_prefix='foo')
    async with server:
        tools = await server.as_toolset().list_tool_defs(run_context)
        assert all(tool.name.startswith('foo_') for tool in tools)


async def test_stdio_server_with_cwd(run_context: RunContext[None]):
    test_dir = Path(__file__).parent
    server = MCPServerStdio('python', ['mcp_server.py'], cwd=test_dir)
    async with server:
        tools = await server.as_toolset().list_tool_defs(run_context)
        assert len(tools) == TOOL_COUNT


async def test_process_tool_call(run_context: RunContext[None]) -> None:
    called: bool = False

    async def process_tool_call(
        ctx: RunContext[None],
        call_tool: CallToolFunc,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolResult:
        """A process_tool_call that sets a flag and sends deps as metadata."""
        nonlocal called
        called = True
        return await call_tool(tool_name, args, {'deps': ctx.deps})

    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], process_tool_call=process_tool_call)
    async with server:
        agent = Agent(deps_type=int, model=TestModel(call_tools=['echo_deps']), mcp_servers=[server])
        result = await agent.run('Echo with deps set to 42', deps=42)
        assert result.output == snapshot('{"echo_deps":{"echo":"This is an echo message","deps":42}}')
        assert called, 'process_tool_call should have been called'


def test_sse_server():
    sse_server = MCPServerSSE(url='http://localhost:8000/sse')
    assert sse_server.url == 'http://localhost:8000/sse'
    assert sse_server._get_log_level() is None  # pyright: ignore[reportPrivateUsage]


def test_sse_server_with_header_and_timeout():
    sse_server = MCPServerSSE(
        url='http://localhost:8000/sse',
        headers={'my-custom-header': 'my-header-value'},
        timeout=10,
        sse_read_timeout=100,
        log_level='info',
    )
    assert sse_server.url == 'http://localhost:8000/sse'
    assert sse_server.headers is not None and sse_server.headers['my-custom-header'] == 'my-header-value'
    assert sse_server.timeout == 10
    assert sse_server.sse_read_timeout == 100
    assert sse_server._get_log_level() == 'info'  # pyright: ignore[reportPrivateUsage]


@pytest.mark.vcr()
async def test_agent_with_stdio_server(allow_model_requests: None, agent: Agent):
    async with agent.run_toolsets():
        result = await agent.run('What is 0 degrees Celsius in Fahrenheit?')
        assert result.output == snapshot('0 degrees Celsius is equal to 32 degrees Fahrenheit.')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is 0 degrees Celsius in Fahrenheit?',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='celsius_to_fahrenheit',
                            args='{"celsius":0}',
                            tool_call_id='call_QssdxTGkPblTYHmyVES1tKBj',
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=195,
                        response_tokens=19,
                        total_tokens=214,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRlnvvqIPFofAtKqtQKMWZkgXhzlT',
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='celsius_to_fahrenheit',
                            content='32.0',
                            tool_call_id='call_QssdxTGkPblTYHmyVES1tKBj',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[TextPart(content='0 degrees Celsius is equal to 32 degrees Fahrenheit.')],
                    usage=Usage(
                        requests=1,
                        request_tokens=227,
                        response_tokens=13,
                        total_tokens=240,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRlnyjUo5wlyqvdNdM5I8vIWjo1qF',
                ),
            ]
        )


async def test_agent_with_conflict_tool_name(agent: Agent):
    @agent.tool_plain
    def get_none() -> None:  # pragma: no cover
        """Return nothing"""
        return None

    async with agent.run_toolsets():
        with pytest.raises(
            UserError,
            match=re.escape(
                "MCP Server 'MCPServerStdio(command='python', args=['-m', 'tests.mcp_server'], tool_prefix=None)' defines a tool whose name conflicts with existing tool: 'get_none'. Consider using `tool_prefix` to avoid name conflicts."
            ),
        ):
            await agent.run('Get me a conflict')


async def test_agent_with_prefix_tool_name(openai_api_key: str):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], tool_prefix='foo')
    model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        model,
        mcp_servers=[server],
    )

    @agent.tool_plain
    def get_none() -> None:  # pragma: no cover
        """Return nothing"""
        return None

    async with agent.run_toolsets():
        # This means that we passed the _prepare_request_parameters check and there is no conflict in the tool name
        with pytest.raises(RuntimeError, match='Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False'):
            await agent.run('No conflict')


async def test_agent_with_server_not_running(openai_api_key: str):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, mcp_servers=[server])
    with pytest.raises(UserError, match='MCP server is not running'):
        await agent.run('What is 0 degrees Celsius in Fahrenheit?')


async def test_log_level_unset(run_context: RunContext[None]):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    assert server._get_log_level() is None  # pyright: ignore[reportPrivateUsage]
    async with server:
        tools = await server.as_toolset().list_tool_defs(run_context)
        assert len(tools) == TOOL_COUNT
        assert tools[10].name == 'get_log_level'

        result = await server.call_tool('get_log_level', {})
        assert result == snapshot('unset')


async def test_log_level_set():
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], log_level='info')
    assert server._get_log_level() == 'info'  # pyright: ignore[reportPrivateUsage]
    async with server:
        result = await server.call_tool('get_log_level', {})
        assert result == snapshot('info')


@pytest.mark.vcr()
async def test_tool_returning_str(allow_model_requests: None, agent: Agent):
    async with agent.run_toolsets():
        result = await agent.run('What is the weather in Mexico City?')
        assert result.output == snapshot(
            'The weather in Mexico City is currently sunny with a temperature of 26 degrees Celsius.'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the weather in Mexico City?',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather_forecast',
                            args='{"location":"Mexico City"}',
                            tool_call_id='call_m9goNwaHBbU926w47V7RtWPt',
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=194,
                        response_tokens=18,
                        total_tokens=212,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRlo3e1Ud2lnvkddMilmwC7LAemiy',
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather_forecast',
                            content='The weather in Mexico City is sunny and 26 degrees Celsius.',
                            tool_call_id='call_m9goNwaHBbU926w47V7RtWPt',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The weather in Mexico City is currently sunny with a temperature of 26 degrees Celsius.'
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=234,
                        response_tokens=19,
                        total_tokens=253,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRlo41LxqBYgGKWgGrQn67fQacOLp',
                ),
            ]
        )


@pytest.mark.vcr()
async def test_tool_returning_text_resource(allow_model_requests: None, agent: Agent):
    async with agent.run_toolsets():
        result = await agent.run('Get me the product name')
        assert result.output == snapshot('The product name is "PydanticAI".')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me the product name',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_product_name',
                            args='{}',
                            tool_call_id='call_LaiWltzI39sdquflqeuF0EyE',
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=200,
                        response_tokens=12,
                        total_tokens=212,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRmhyweJVYonarb7s9ckIMSHf2vHo',
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_product_name',
                            content='PydanticAI',
                            tool_call_id='call_LaiWltzI39sdquflqeuF0EyE',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[TextPart(content='The product name is "PydanticAI".')],
                    usage=Usage(
                        requests=1,
                        request_tokens=224,
                        response_tokens=12,
                        total_tokens=236,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRmhzqXFObpYwSzREMpJvX9kbDikR',
                ),
            ]
        )


@pytest.mark.vcr()
async def test_tool_returning_image_resource(allow_model_requests: None, agent: Agent, image_content: BinaryContent):
    async with agent.run_toolsets():
        result = await agent.run('Get me the image resource')
        assert result.output == snapshot(
            'This is an image of a sliced kiwi with a vibrant green interior and black seeds.'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me the image resource',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_image_resource',
                            args='{}',
                            tool_call_id='call_nFsDHYDZigO0rOHqmChZ3pmt',
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=191,
                        response_tokens=12,
                        total_tokens=203,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRlo7KYJVXuNZ5lLLdYcKZDsX2CHb',
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_image_resource',
                            content='See file 1c8566',
                            tool_call_id='call_nFsDHYDZigO0rOHqmChZ3pmt',
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(content=['This is file 1c8566:', image_content], timestamp=IsDatetime()),
                    ]
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='This is an image of a sliced kiwi with a vibrant green interior and black seeds.'
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=1332,
                        response_tokens=19,
                        total_tokens=1351,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloBGHh27w3fQKwxq4fX2cPuZJa9',
                ),
            ]
        )


@pytest.mark.vcr()
async def test_tool_returning_audio_resource(
    allow_model_requests: None, agent: Agent, audio_content: BinaryContent, gemini_api_key: str
):
    model = GoogleModel('gemini-2.5-pro-preview-03-25', provider=GoogleProvider(api_key=gemini_api_key))
    async with agent.run_toolsets():
        result = await agent.run("What's the content of the audio resource?", model=model)
        assert result.output == snapshot('The audio resource contains a voice saying "Hello, my name is Marcelo."')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the content of the audio resource?", timestamp=IsDatetime())]
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_audio_resource', args={}, tool_call_id=IsStr())],
                    usage=Usage(
                        requests=1,
                        request_tokens=383,
                        response_tokens=12,
                        total_tokens=520,
                        details={'thoughts_tokens': 125, 'text_prompt_tokens': 383},
                    ),
                    model_name='models/gemini-2.5-pro-preview-05-06',
                    timestamp=IsDatetime(),
                    vendor_details={'finish_reason': 'STOP'},
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_audio_resource',
                            content='See file 2d36ae',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(content=['This is file 2d36ae:', audio_content], timestamp=IsDatetime()),
                    ]
                ),
                ModelResponse(
                    parts=[TextPart(content='The audio resource contains a voice saying "Hello, my name is Marcelo."')],
                    usage=Usage(
                        requests=1,
                        request_tokens=575,
                        response_tokens=15,
                        total_tokens=590,
                        details={'text_prompt_tokens': 431, 'audio_prompt_tokens': 144},
                    ),
                    model_name='models/gemini-2.5-pro-preview-05-06',
                    timestamp=IsDatetime(),
                    vendor_details={'finish_reason': 'STOP'},
                ),
            ]
        )


@pytest.mark.vcr()
async def test_tool_returning_image(allow_model_requests: None, agent: Agent, image_content: BinaryContent):
    async with agent.run_toolsets():
        result = await agent.run('Get me an image')
        assert result.output == snapshot('Here is an image of a sliced kiwi on a white background.')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me an image',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_image',
                            args='{}',
                            tool_call_id='call_Q7xG8CCG0dyevVfUS0ubsDdN',
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=190,
                        response_tokens=11,
                        total_tokens=201,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloGQJWIX0Qk7gtNzF4s2Fez0O29',
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_image',
                            content='See file 1c8566',
                            tool_call_id='call_Q7xG8CCG0dyevVfUS0ubsDdN',
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(
                            content=[
                                'This is file 1c8566:',
                                image_content,
                            ],
                            timestamp=IsDatetime(),
                        ),
                    ]
                ),
                ModelResponse(
                    parts=[TextPart(content='Here is an image of a sliced kiwi on a white background.')],
                    usage=Usage(
                        requests=1,
                        request_tokens=1329,
                        response_tokens=15,
                        total_tokens=1344,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloJHR654fSD0fcvLWZxtKtn0pag',
                ),
            ]
        )


@pytest.mark.vcr()
async def test_tool_returning_dict(allow_model_requests: None, agent: Agent):
    async with agent.run_toolsets():
        result = await agent.run('Get me a dict, respond on one line')
        assert result.output == snapshot('{"foo":"bar","baz":123}')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me a dict, respond on one line',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_dict', args='{}', tool_call_id='call_oqKviITBj8PwpQjGyUu4Zu5x')],
                    usage=Usage(
                        requests=1,
                        request_tokens=195,
                        response_tokens=11,
                        total_tokens=206,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloOs7Bb2tq8wJyy9Rv7SQ7L65a7',
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_dict',
                            content={'foo': 'bar', 'baz': 123},
                            tool_call_id='call_oqKviITBj8PwpQjGyUu4Zu5x',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[TextPart(content='{"foo":"bar","baz":123}')],
                    usage=Usage(
                        requests=1,
                        request_tokens=222,
                        response_tokens=11,
                        total_tokens=233,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloPczU1HSCWnreyo21DdNtdOM7L',
                ),
            ]
        )


@pytest.mark.vcr()
async def test_tool_returning_error(allow_model_requests: None, agent: Agent):
    async with agent.run_toolsets():
        result = await agent.run('Get me an error, pass False as a value, unless the tool tells you otherwise')
        assert result.output == snapshot(
            'I called the tool with the correct parameter, and it returned: "This is not an error."'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me an error, pass False as a value, unless the tool tells you otherwise',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_error',
                            args='{"value":false}',
                            tool_call_id='call_rETXZWddAGZSHyVHAxptPGgc',
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=203,
                        response_tokens=15,
                        total_tokens=218,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloSNg7aGSp1rXDkhInjMIUHKd7A',
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Error executing tool get_error: This is an error. Call the tool with True instead',
                            tool_name='get_error',
                            tool_call_id='call_rETXZWddAGZSHyVHAxptPGgc',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_error',
                            args='{"value":true}',
                            tool_call_id='call_4xGyvdghYKHN8x19KWkRtA5N',
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=250,
                        response_tokens=15,
                        total_tokens=265,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloTvSkFeX4DZKQLqfH9KbQkWlpt',
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_error',
                            content='This is not an error',
                            tool_call_id='call_4xGyvdghYKHN8x19KWkRtA5N',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='I called the tool with the correct parameter, and it returned: "This is not an error."'
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=277,
                        response_tokens=22,
                        total_tokens=299,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloU3MhnqNEqujs28a3ofRbs7VPF',
                ),
            ]
        )


@pytest.mark.vcr()
async def test_tool_returning_none(allow_model_requests: None, agent: Agent):
    async with agent.run_toolsets():
        result = await agent.run('Call the none tool and say Hello')
        assert result.output == snapshot('Hello! How can I assist you today?')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Call the none tool and say Hello',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_none', args='{}', tool_call_id='call_mJTuQ2Cl5SaHPTJbIILEUhJC')],
                    usage=Usage(
                        requests=1,
                        request_tokens=193,
                        response_tokens=11,
                        total_tokens=204,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloX2RokWc9j9PAXAuNXGR73WNqY',
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_none',
                            content=[],
                            tool_call_id='call_mJTuQ2Cl5SaHPTJbIILEUhJC',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[TextPart(content='Hello! How can I assist you today?')],
                    usage=Usage(
                        requests=1,
                        request_tokens=212,
                        response_tokens=11,
                        total_tokens=223,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloYWGujk8yE94gfVSsM1T1Ol2Ej',
                ),
            ]
        )


@pytest.mark.vcr()
async def test_tool_returning_multiple_items(allow_model_requests: None, agent: Agent, image_content: BinaryContent):
    async with agent.run_toolsets():
        result = await agent.run('Get me multiple items and summarize in one sentence')
        assert result.output == snapshot(
            'The data includes two strings, a dictionary with a key-value pair, and an image of a sliced kiwi.'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me multiple items and summarize in one sentence',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_multiple_items',
                            args='{}',
                            tool_call_id='call_kL0TvjEVQBDGZrn1Zv7iNYOW',
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=195,
                        response_tokens=12,
                        total_tokens=207,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRlobKLgm6vf79c9O8sloZaYx3coC',
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_multiple_items',
                            content=[
                                'This is a string',
                                'Another string',
                                {'foo': 'bar', 'baz': 123},
                                'See file 1c8566',
                            ],
                            tool_call_id='call_kL0TvjEVQBDGZrn1Zv7iNYOW',
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(
                            content=[
                                'This is file 1c8566:',
                                image_content,
                            ],
                            timestamp=IsDatetime(),
                        ),
                    ]
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The data includes two strings, a dictionary with a key-value pair, and an image of a sliced kiwi.'
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=1355,
                        response_tokens=24,
                        total_tokens=1379,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                            'cached_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    vendor_id='chatcmpl-BRloepWR5NJpTgSqFBGTSPeM1SWm8',
                ),
            ]
        )


async def test_mcp_server_raises_mcp_error(
    allow_model_requests: None, mcp_server: MCPServerStdio, agent: Agent
) -> None:
    mcp_error = McpError(error=ErrorData(code=400, message='Test MCP error conversion'))

    async with agent.run_toolsets():
        with patch.object(
            mcp_server._client,  # pyright: ignore[reportPrivateUsage]
            'send_request',
            new=AsyncMock(side_effect=mcp_error),
        ):
            with pytest.raises(ModelRetry, match='Test MCP error conversion'):
                await mcp_server.call_tool('test_tool', {})
