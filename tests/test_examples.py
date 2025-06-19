from __future__ import annotations as _annotations

import json
import os
import re
import sys
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from inspect import FrameInfo
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import httpx
import pytest
from _pytest.mark import ParameterSet
from devtools import debug
from pytest_examples import CodeExample, EvalExample, find_examples
from pytest_mock import MockerFixture
from rich.console import Console

from pydantic_ai import ModelHTTPError
from pydantic_ai._utils import group_by_temporal
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.mcp import MCPServer
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolset import BaseToolset, MCPToolset

from .conftest import ClientWithHandler, TestEnv, try_import

try:
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider
except ImportError:  # pragma: lax no cover
    GoogleVertexProvider = None
    GoogleProvider = None


try:
    import logfire
except ImportError:  # pragma: lax no cover
    logfire = None


with try_import() as imports_successful:
    from pydantic_evals.reporting import EvaluationReport


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='extras not installed'),
    pytest.mark.skipif(
        GoogleVertexProvider is None or logfire is None or GoogleProvider is None,
        reason='google-auth or logfire or google-provider not installed',
    ),
]


def find_filter_examples() -> Iterable[ParameterSet]:
    # Ensure this is run from the package root regardless of where/how the tests are run
    os.chdir(Path(__file__).parent.parent)

    for ex in find_examples('docs', 'pydantic_ai_slim', 'pydantic_graph', 'pydantic_evals'):
        if ex.path.name != '_utils.py':
            try:
                path = ex.path.relative_to(Path.cwd())
            except ValueError:
                path = ex.path
            test_id = f'{path}:{ex.start_line}'
            prefix_settings = ex.prefix_settings()
            if opt_title := prefix_settings.get('title'):
                test_id += f':{opt_title}'
            yield pytest.param(ex, id=test_id)


@pytest.fixture
def reset_cwd():
    original_cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original_cwd)


@pytest.mark.xdist_group(name='doc_tests')
@pytest.mark.parametrize('example', find_filter_examples())
def test_docs_examples(  # noqa: C901
    example: CodeExample,
    eval_example: EvalExample,
    mocker: MockerFixture,
    client_with_handler: ClientWithHandler,
    allow_model_requests: None,
    env: TestEnv,
    tmp_path: Path,
    reset_cwd: None,
):
    mocker.patch('pydantic_ai.agent.models.infer_model', side_effect=mock_infer_model)
    mocker.patch('pydantic_ai._utils.group_by_temporal', side_effect=mock_group_by_temporal)
    mocker.patch('pydantic_evals.reporting.render_numbers._render_duration', side_effect=mock_render_duration)

    mocker.patch('httpx.Client.get', side_effect=http_request)
    mocker.patch('httpx.Client.post', side_effect=http_request)
    mocker.patch('httpx.AsyncClient.get', side_effect=async_http_request)
    mocker.patch('httpx.AsyncClient.post', side_effect=async_http_request)
    mocker.patch('random.randint', return_value=4)
    mocker.patch('rich.prompt.Prompt.ask', side_effect=rich_prompt_ask)

    class CustomEvaluationReport(EvaluationReport):
        def print(self, *args: Any, **kwargs: Any) -> None:
            if 'width' in kwargs:  # pragma: lax no cover
                raise ValueError('width should not be passed to CustomEvaluationReport')
            table = self.console_table(*args, **kwargs)
            io_file = StringIO()
            Console(file=io_file, width=150).print(table)
            print(io_file.getvalue())

    mocker.patch('pydantic_evals.dataset.EvaluationReport', side_effect=CustomEvaluationReport)

    if sys.version_info >= (3, 10):  # pragma: lax no cover
        mocker.patch('pydantic_ai.mcp.MCPServerSSE', return_value=MockMCPServer())
        mocker.patch('pydantic_ai.mcp.MCPServerStreamableHTTP', return_value=MockMCPServer())
        mocker.patch('mcp.server.fastmcp.FastMCP')

    env.set('OPENAI_API_KEY', 'testing')
    env.set('GEMINI_API_KEY', 'testing')
    env.set('GOOGLE_API_KEY', 'testing')
    env.set('GROQ_API_KEY', 'testing')
    env.set('CO_API_KEY', 'testing')
    env.set('MISTRAL_API_KEY', 'testing')
    env.set('ANTHROPIC_API_KEY', 'testing')
    env.set('AWS_ACCESS_KEY_ID', 'testing')
    env.set('AWS_SECRET_ACCESS_KEY', 'testing')
    env.set('AWS_DEFAULT_REGION', 'us-east-1')

    sys.path.append('tests/example_modules')

    prefix_settings = example.prefix_settings()
    opt_title = prefix_settings.get('title')
    opt_test = prefix_settings.get('test', '')
    opt_lint = prefix_settings.get('lint', '')
    noqa = prefix_settings.get('noqa', '')
    python_version = prefix_settings.get('py', None)

    if python_version:
        python_version_info = tuple(int(v) for v in python_version.split('.'))
        if sys.version_info < python_version_info:
            pytest.skip(f'Python version {python_version} required')  # pragma: lax no cover

    if opt_test.startswith('skip') and opt_lint.startswith('skip'):
        pytest.skip('both running code and lint skipped')

    if opt_title in {
        'ai_q_and_a_run.py',
        'count_down_from_persistence.py',
        'generate_dataset_example.py',
        'generate_dataset_example_json.py',
        'save_load_dataset_example.py',
    }:
        os.chdir(tmp_path)

    ruff_ignore: list[str] = ['D', 'Q001']
    # `from bank_database import DatabaseConn` wrongly sorted in imports
    # waiting for https://github.com/pydantic/pytest-examples/issues/43
    # and https://github.com/pydantic/pytest-examples/issues/46
    if 'import DatabaseConn' in example.source:
        ruff_ignore.append('I001')

    if noqa:
        ruff_ignore.extend(noqa.upper().split())

    line_length = int(prefix_settings.get('line_length', '88'))

    eval_example.set_config(ruff_ignore=ruff_ignore, target_version='py39', line_length=line_length)
    eval_example.print_callback = print_callback
    eval_example.include_print = custom_include_print

    call_name = prefix_settings.get('call_name', 'main')

    if not opt_lint.startswith('skip'):
        # ruff and seem to black disagree here, not sure if that's easily fixable
        if eval_example.update_examples:  # pragma: lax no cover
            eval_example.format_ruff(example)
        else:
            eval_example.lint_ruff(example)

    if opt_test.startswith('skip'):
        print(opt_test[4:].lstrip(' -') or 'running code skipped')
    else:
        test_globals: dict[str, str] = {}
        if opt_title == 'mcp_client.py':
            test_globals['__name__'] = '__test__'
        if eval_example.update_examples:  # pragma: lax no cover
            module_dict = eval_example.run_print_update(example, call=call_name, module_globals=test_globals)
        else:
            module_dict = eval_example.run_print_check(example, call=call_name, module_globals=test_globals)

        if title := opt_title:
            if title.endswith('.py'):
                module_name = title[:-3]
                sys.modules[module_name] = module = ModuleType(module_name)
                module.__dict__.update(module_dict)


def print_callback(s: str) -> str:
    s = re.sub(r'datetime\.datetime\(.+?\)', 'datetime.datetime(...)', s, flags=re.DOTALL)
    s = re.sub(r'\d\.\d{4,}e-0\d', '0.0...', s)
    return re.sub(r'datetime.date\(', 'date(', s)


def mock_render_duration(seconds: float, force_signed: bool) -> str:
    return '10ms'


def custom_include_print(path: Path, frame: FrameInfo, args: Sequence[Any]) -> bool:
    return path.samefile(frame.filename) or frame.filename.endswith('test_examples.py')


def http_request(url: str, **kwargs: Any) -> httpx.Response:
    # sys.stdout.write(f'GET {args=} {kwargs=}\n')
    request = httpx.Request('GET', url, **kwargs)
    return httpx.Response(status_code=202, content='', request=request)


async def async_http_request(url: str, **kwargs: Any) -> httpx.Response:
    return http_request(url, **kwargs)


def rich_prompt_ask(prompt: str, *_args: Any, **_kwargs: Any) -> str:
    if prompt == 'Where would you like to fly from and to?':
        return 'SFO to ANC'
    elif prompt == 'What seat would you like?':
        return 'window seat with leg room'
    if prompt == 'Insert coins':
        return '1'
    elif prompt == 'Select product':
        return 'crisps'
    elif prompt == 'What is the capital of France?':  # pragma: no cover
        return 'Vichy'
    elif prompt == 'what is 1 + 1?':  # pragma: no cover
        return '2'
    else:  # pragma: no cover
        raise ValueError(f'Unexpected prompt: {prompt}')


class MockMCPServer:
    is_running = True

    async def __aenter__(self) -> MockMCPServer:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def list_tools(self) -> list[None]:
        return []

    async def call_tool(self, name: str, args: dict[str, Any], metadata: dict[str, Any] | None = None) -> Any:
        return None

    def as_toolset(self, max_retries: int = 1) -> BaseToolset:
        return MCPToolset(cast(MCPServer, self), max_retries=max_retries)


text_responses: dict[str, str | ToolCallPart] = {
    'How many days between 2000-01-01 and 2025-03-18?': 'There are 9,208 days between January 1, 2000, and March 18, 2025.',
    'What is the weather like in West London and in Wiltshire?': (
        'The weather in West London is raining, while in Wiltshire it is sunny.'
    ),
    'What will the weather be like in Paris on Tuesday?': ToolCallPart(
        tool_name='weather_forecast', args={'location': 'Paris', 'forecast_date': '2030-01-01'}, tool_call_id='0001'
    ),
    'Tell me a joke.': 'Did you hear about the toothpaste scandal? They called it Colgate.',
    'Tell me a different joke.': 'No.',
    'Explain?': 'This is an excellent joke invented by Samuel Colvin, it needs no explanation.',
    'What is the weather in Tokyo?': 'As of 7:48 AM on Wednesday, April 2, 2025, in Tokyo, Japan, the weather is cloudy with a temperature of 53°F (12°C).',
    'What is the capital of France?': 'Paris',
    'What is the capital of Italy?': 'Rome',
    'What is the capital of the UK?': 'London',
    'Who was Albert Einstein?': 'Albert Einstein was a German-born theoretical physicist.',
    'What was his most famous equation?': "Albert Einstein's most famous equation is (E = mc^2).",
    'What is the date?': 'Hello Frank, the date today is 2032-01-02.',
    'Put my money on square eighteen': ToolCallPart(
        tool_name='roulette_wheel', args={'square': 18}, tool_call_id='pyd_ai_tool_call_id'
    ),
    'I bet five is the winner': ToolCallPart(
        tool_name='roulette_wheel', args={'square': 5}, tool_call_id='pyd_ai_tool_call_id'
    ),
    'My guess is 6': ToolCallPart(tool_name='roll_dice', args={}, tool_call_id='pyd_ai_tool_call_id'),
    'My guess is 4': ToolCallPart(tool_name='roll_dice', args={}, tool_call_id='pyd_ai_tool_call_id'),
    'Send a message to John Doe asking for coffee next week': ToolCallPart(
        tool_name='get_user_by_name', args={'name': 'John'}
    ),
    'Please get me the volume of a box with size 6.': ToolCallPart(
        tool_name='calc_volume', args={'size': 6}, tool_call_id='pyd_ai_tool_call_id'
    ),
    'Where does "hello world" come from?': (
        'The first known use of "hello, world" was in a 1974 textbook about the C programming language.'
    ),
    'What is my balance?': ToolCallPart(tool_name='customer_balance', args={'include_pending': True}),
    'I just lost my card!': ToolCallPart(
        tool_name='final_result',
        args={
            'support_advice': (
                "I'm sorry to hear that, John. "
                'We are temporarily blocking your card to prevent unauthorized transactions.'
            ),
            'block_card': True,
            'risk': 8,
        },
    ),
    'Where were the olympics held in 2012?': ToolCallPart(
        tool_name='final_result',
        args={'city': 'London', 'country': 'United Kingdom'},
    ),
    'The box is 10x20x30': 'Please provide the units for the dimensions (e.g., cm, in, m).',
    'The box is 10x20x30 cm': ToolCallPart(
        tool_name='final_result',
        args={'width': 10, 'height': 20, 'depth': 30, 'units': 'cm'},
    ),
    'red square, blue circle, green triangle': ToolCallPart(
        tool_name='final_result_list',
        args={'response': ['red', 'blue', 'green']},
    ),
    'square size 10, circle size 20, triangle size 30': ToolCallPart(
        tool_name='final_result_list_2',
        args={'response': [10, 20, 30]},
    ),
    'get me users who were last active yesterday.': ToolCallPart(
        tool_name='final_result_Success',
        args={'sql_query': 'SELECT * FROM users WHERE last_active::date = today() - interval 1 day'},
    ),
    'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.': ToolCallPart(
        tool_name='final_result',
        args={
            'name': 'Ben',
            'dob': '1990-01-28',
            'bio': 'Likes the chain the dog and the pyramid',
        },
        tool_call_id='pyd_ai_tool_call_id',
    ),
    'What is the capital of Italy? Answer with just the city.': 'Rome',
    'What is the capital of Italy? Answer with a paragraph.': (
        'The capital of Italy is Rome (Roma, in Italian), which has been a cultural and political center for centuries.'
        'Rome is known for its rich history, stunning architecture, and delicious cuisine.'
    ),
    'Begin infinite retry loop!': ToolCallPart(
        tool_name='infinite_retry_tool', args={}, tool_call_id='pyd_ai_tool_call_id'
    ),
    'Please generate 5 jokes.': ToolCallPart(
        tool_name='final_result',
        args={'response': []},
        tool_call_id='pyd_ai_tool_call_id',
    ),
    'SFO to ANC': ToolCallPart(
        tool_name='flight_search',
        args={'origin': 'SFO', 'destination': 'ANC'},
        tool_call_id='pyd_ai_tool_call_id',
    ),
    'window seat with leg room': ToolCallPart(
        tool_name='final_result_SeatPreference',
        args={'row': 1, 'seat': 'A'},
        tool_call_id='pyd_ai_tool_call_id',
    ),
    'Ask a simple question with a single correct answer.': 'What is the capital of France?',
    '<examples>\n  <question>What is the capital of France?</question>\n  <answer>Vichy</answer>\n</examples>': ToolCallPart(
        tool_name='final_result',
        args={'correct': False, 'comment': 'Vichy is no longer the capital of France.'},
        tool_call_id='pyd_ai_tool_call_id',
    ),
    '<examples>\n  <question>what is 1 + 1?</question>\n  <answer>2</answer>\n</examples>': ToolCallPart(
        tool_name='final_result',
        args={'correct': True, 'comment': 'Well done, 1 + 1 = 2'},
        tool_call_id='pyd_ai_tool_call_id',
    ),
    (
        '<examples>\n'
        '  <dish_name>Spaghetti Bolognese</dish_name>\n'
        '  <dietary_restriction>vegetarian</dietary_restriction>\n'
        '</examples>'
    ): ToolCallPart(
        tool_name='final_result',
        args={
            'ingredients': ['spaghetti', 'tomato sauce', 'vegetarian mince', 'onions', 'garlic'],
            'steps': ['Cook the spaghetti in boiling water', '...'],
        },
    ),
    (
        '<examples>\n'
        '  <dish_name>Chocolate Cake</dish_name>\n'
        '  <dietary_restriction>gluten-free</dietary_restriction>\n'
        '</examples>'
    ): ToolCallPart(
        tool_name='final_result',
        args={
            'ingredients': ['gluten-free flour', 'cocoa powder', 'sugar', 'eggs'],
            'steps': ['Mix the ingredients', 'Bake at 350°F for 30 minutes'],
        },
    ),
    'What is 123 / 456?': ToolCallPart(
        tool_name='divide',
        args={'numerator': '123', 'denominator': '456'},
        tool_call_id='pyd_ai_2e0e396768a14fe482df90a29a78dc7b',
    ),
    'Select the names and countries of all capitals': ToolCallPart(
        tool_name='final_result_hand_off_to_sql_agent',
        args={'query': 'SELECT name, country FROM capitals;'},
    ),
    'SELECT name, country FROM capitals;': ToolCallPart(
        tool_name='final_result_run_sql_query',
        args={'query': 'SELECT name, country FROM capitals;'},
    ),
    'SELECT * FROM capital_cities;': ToolCallPart(
        tool_name='final_result_run_sql_query',
        args={'query': 'SELECT * FROM capital_cities;'},
    ),
    'Select all pets': ToolCallPart(
        tool_name='final_result_hand_off_to_sql_agent',
        args={'query': 'SELECT * FROM pets;'},
    ),
    'SELECT * FROM pets;': ToolCallPart(
        tool_name='final_result_run_sql_query',
        args={'query': 'SELECT * FROM pets;'},
    ),
    'How do I fly from Amsterdam to Mexico City?': ToolCallPart(
        tool_name='final_result_RouterFailure',
        args={
            'explanation': 'I am not equipped to provide travel information, such as flights from Amsterdam to Mexico City.'
        },
    ),
    'What is a banana?': ToolCallPart(tool_name='return_fruit', args={'name': 'banana', 'color': 'yellow'}),
    'What is a Ford Explorer?': '{"result": {"kind": "Vehicle", "data": {"name": "Ford Explorer", "wheels": 4}}}',
    'What is a MacBook?': '{"result": {"kind": "Device", "data": {"name": "MacBook", "kind": "laptop"}}}',
}

tool_responses: dict[tuple[str, str], str] = {
    (
        'weather_forecast',
        'The forecast in Paris on 2030-01-01 is 24°C and sunny.',
    ): 'It will be warm and sunny in Paris on Tuesday.',
}


async def model_logic(  # noqa: C901
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:  # pragma: lax no cover
    m = messages[-1].parts[-1]
    if isinstance(m, UserPromptPart):
        if isinstance(m.content, list) and m.content[0] == 'This is file d9a13f:':
            return ModelResponse(parts=[TextPart('The company name in the logo is "Pydantic."')])
        elif isinstance(m.content, list) and m.content[0] == 'This is file c6720d:':
            return ModelResponse(parts=[TextPart('The document contains just the text "Dummy PDF file."')])

        assert isinstance(m.content, str)
        if m.content == 'Tell me a joke.' and any(t.name == 'joke_factory' for t in info.function_tools):
            return ModelResponse(
                parts=[ToolCallPart(tool_name='joke_factory', args={'count': 5}, tool_call_id='pyd_ai_tool_call_id')]
            )
        elif m.content == 'Please generate 5 jokes.' and any(t.name == 'get_jokes' for t in info.function_tools):
            return ModelResponse(
                parts=[ToolCallPart(tool_name='get_jokes', args={'count': 5}, tool_call_id='pyd_ai_tool_call_id')]
            )
        elif re.fullmatch(r'sql prompt \d+', m.content):
            return ModelResponse(parts=[TextPart('SELECT 1')])
        elif m.content.startswith('Write a welcome email for the user:'):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={
                            'subject': 'Welcome to our tech blog!',
                            'body': 'Hello John, Welcome to our tech blog! ...',
                        },
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ]
            )
        elif m.content.startswith('Write a list of 5 very rude things that I might say'):
            raise UnexpectedModelBehavior('Safety settings triggered', body='<safety settings details>')
        elif m.content.startswith('<examples>\n  <user>'):
            return ModelResponse(
                parts=[ToolCallPart(tool_name='final_result_EmailOk', args={}, tool_call_id='pyd_ai_tool_call_id')]
            )
        elif m.content == 'Ask a simple question with a single correct answer.' and len(messages) > 2:
            return ModelResponse(parts=[TextPart('what is 1 + 1?')])
        elif '<Rubric>\n' in m.content:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'reason': '-', 'pass': True, 'score': 1.0})]
            )
        elif m.content == 'What time is it?':
            return ModelResponse(
                parts=[ToolCallPart(tool_name='get_current_time', args={}, tool_call_id='pyd_ai_tool_call_id')]
            )
        elif m.content == 'What is the user name?':
            return ModelResponse(
                parts=[ToolCallPart(tool_name='get_user', args={}, tool_call_id='pyd_ai_tool_call_id')]
            )
        elif m.content == 'What is the company name in the logo?':
            return ModelResponse(
                parts=[ToolCallPart(tool_name='get_company_logo', args={}, tool_call_id='pyd_ai_tool_call_id')]
            )
        elif m.content == 'What is the main content of the document?':
            return ModelResponse(
                parts=[ToolCallPart(tool_name='get_document', args={}, tool_call_id='pyd_ai_tool_call_id')]
            )
        elif 'Generate question-answer pairs about world capitals and landmarks.' in m.content:
            return ModelResponse(
                parts=[
                    TextPart(
                        content=json.dumps(
                            {
                                'cases': [
                                    {
                                        'name': 'Easy Capital Question',
                                        'inputs': {'question': 'What is the capital of France?'},
                                        'metadata': {'difficulty': 'easy', 'category': 'Geography'},
                                        'expected_output': {'answer': 'Paris', 'confidence': 0.95},
                                        'evaluators': ['EqualsExpected'],
                                    },
                                    {
                                        'name': 'Challenging Landmark Question',
                                        'inputs': {
                                            'question': 'Which world-famous landmark is located on the banks of the Seine River?',
                                        },
                                        'metadata': {'difficulty': 'hard', 'category': 'Landmarks'},
                                        'expected_output': {'answer': 'Eiffel Tower', 'confidence': 0.9},
                                        'evaluators': ['EqualsExpected'],
                                    },
                                ],
                                'evaluators': [],
                            }
                        )
                    )
                ]
            )
        elif response := text_responses.get(m.content):
            if isinstance(response, str):
                return ModelResponse(parts=[TextPart(response)])
            else:
                return ModelResponse(parts=[response])

    elif isinstance(m, ToolReturnPart) and m.tool_name == 'roulette_wheel':
        win = m.content == 'winner'
        return ModelResponse(
            parts=[ToolCallPart(tool_name='final_result', args={'response': win}, tool_call_id='pyd_ai_tool_call_id')],
        )
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'roll_dice':
        return ModelResponse(
            parts=[ToolCallPart(tool_name='get_player_name', args={}, tool_call_id='pyd_ai_tool_call_id')]
        )
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_player_name':
        if 'Anne' in m.content:
            return ModelResponse(parts=[TextPart("Congratulations Anne, you guessed correctly! You're a winner!")])
        elif 'Yashar' in m.content:
            return ModelResponse(parts=[TextPart('Tough luck, Yashar, you rolled a 4. Better luck next time.')])
    if (
        isinstance(m, RetryPromptPart)
        and isinstance(m.content, str)
        and m.content.startswith("No user found with name 'Joh")
    ):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='get_user_by_name', args={'name': 'John Doe'}, tool_call_id='pyd_ai_tool_call_id'
                )
            ]
        )
    elif isinstance(m, RetryPromptPart) and m.tool_name == 'infinite_retry_tool':
        return ModelResponse(
            parts=[ToolCallPart(tool_name='infinite_retry_tool', args={}, tool_call_id='pyd_ai_tool_call_id')]
        )
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_user_by_name':
        args: dict[str, Any] = {
            'message': 'Hello John, would you be free for coffee sometime next week? Let me know what works for you!',
            'user_id': 123,
        }
        return ModelResponse(
            parts=[ToolCallPart(tool_name='final_result', args=args, tool_call_id='pyd_ai_tool_call_id')]
        )
    elif isinstance(m, RetryPromptPart) and m.tool_name == 'calc_volume':
        return ModelResponse(
            parts=[ToolCallPart(tool_name='calc_volume', args={'size': 6}, tool_call_id='pyd_ai_tool_call_id')]
        )
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'customer_balance':
        args = {
            'support_advice': 'Hello John, your current account balance, including pending transactions, is $123.45.',
            'block_card': False,
            'risk': 1,
        }
        return ModelResponse(
            parts=[ToolCallPart(tool_name='final_result', args=args, tool_call_id='pyd_ai_tool_call_id')]
        )
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'joke_factory':
        return ModelResponse(parts=[TextPart('Did you hear about the toothpaste scandal? They called it Colgate.')])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_jokes':
        args = {'response': []}
        return ModelResponse(
            parts=[ToolCallPart(tool_name='final_result', args=args, tool_call_id='pyd_ai_tool_call_id')]
        )
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'flight_search':
        args = {'flight_number': m.content.flight_number}  # type: ignore
        return ModelResponse(
            parts=[ToolCallPart(tool_name='final_result_FlightDetails', args=args, tool_call_id='pyd_ai_tool_call_id')]
        )
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_current_time':
        return ModelResponse(parts=[TextPart('The current time is 10:45 PM on April 17, 2025.')])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_user':
        return ModelResponse(parts=[TextPart("The user's name is John.")])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_company_logo':
        return ModelResponse(parts=[TextPart('The company name in the logo is "Pydantic."')])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_document':
        return ModelResponse(
            parts=[ToolCallPart(tool_name='get_document', args={}, tool_call_id='pyd_ai_tool_call_id')]
        )
    elif (
        isinstance(m, RetryPromptPart)
        and m.tool_name == 'final_result_run_sql_query'
        and m.content == "Only 'SELECT *' is supported, you'll have to do column filtering manually."
    ):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='final_result_run_sql_query',
                    args={'query': 'SELECT * FROM capitals;'},
                    tool_call_id='pyd_ai_tool_call_id',
                )
            ]
        )
    elif (
        isinstance(m, RetryPromptPart)
        and m.tool_name == 'final_result_hand_off_to_sql_agent'
        and m.content
        == "SQL agent failed: Unknown table 'capitals' in query 'SELECT * FROM capitals;'. Available tables: capital_cities."
    ):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='final_result_hand_off_to_sql_agent',
                    args={'query': 'SELECT * FROM capital_cities;'},
                    tool_call_id='pyd_ai_tool_call_id',
                )
            ]
        )
    elif (
        isinstance(m, RetryPromptPart)
        and m.tool_name == 'final_result_run_sql_query'
        and m.content == "Unknown table 'pets' in query 'SELECT * FROM pets;'. Available tables: capital_cities."
    ):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='final_result_SQLFailure',
                    args={
                        'explanation': "The table 'pets' does not exist in the database. Only the table 'capital_cities' is available."
                    },
                    tool_call_id='pyd_ai_tool_call_id',
                )
            ]
        )
    # SQL agent failed: The table 'pets' does not exist in the database. Only the table 'capital_cities' is available.
    elif (
        isinstance(m, RetryPromptPart)
        and m.tool_name == 'final_result_hand_off_to_sql_agent'
        and m.content
        == "SQL agent failed: The table 'pets' does not exist in the database. Only the table 'capital_cities' is available."
    ):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='final_result_RouterFailure',
                    args={
                        'explanation': "The requested table 'pets' does not exist in the database. The only available table is 'capital_cities', which does not contain data about pets."
                    },
                    tool_call_id='pyd_ai_tool_call_id',
                )
            ]
        )
    else:
        sys.stdout.write(str(debug.format(messages, info)))
        raise RuntimeError(f'Unexpected message: {m}')


async def stream_model_logic(  # noqa C901
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:  # pragma: lax no cover
    async def stream_text_response(r: str) -> AsyncIterator[str]:
        if isinstance(r, str):
            words = r.split(' ')
            chunk: list[str] = []
            for word in words:
                chunk.append(word)
                if len(chunk) == 3:
                    yield ' '.join(chunk) + ' '
                    chunk.clear()
            if chunk:
                yield ' '.join(chunk)

    async def stream_tool_call_response(r: ToolCallPart) -> AsyncIterator[DeltaToolCalls]:
        json_text = r.args_as_json_str()

        yield {1: DeltaToolCall(name=r.tool_name, tool_call_id=r.tool_call_id)}
        for chunk_index in range(0, len(json_text), 15):
            text_chunk = json_text[chunk_index : chunk_index + 15]
            yield {1: DeltaToolCall(json_args=text_chunk)}

    async def stream_part_response(r: str | ToolCallPart) -> AsyncIterator[str | DeltaToolCalls]:
        if isinstance(r, str):
            async for chunk in stream_text_response(r):
                yield chunk
        else:
            async for chunk in stream_tool_call_response(r):
                yield chunk

    last_part = messages[-1].parts[-1]
    if isinstance(last_part, UserPromptPart):
        assert isinstance(last_part.content, str)
        if response := text_responses.get(last_part.content):
            async for chunk in stream_part_response(response):
                yield chunk
            return
    elif isinstance(last_part, ToolReturnPart):
        assert isinstance(last_part.content, str)
        if response := tool_responses.get((last_part.tool_name, last_part.content)):
            async for chunk in stream_part_response(response):
                yield chunk
            return

    sys.stdout.write(str(debug.format(messages, info)))
    raise RuntimeError(f'Unexpected message: {last_part}')


def mock_infer_model(model: Model | KnownModelName) -> Model:
    if model == 'test':
        return TestModel()

    if isinstance(model, str):
        # Use the non-mocked model inference to ensure we get the same model name the user would
        model = infer_model(model)

    if isinstance(model, FallbackModel):
        # When a fallback model is encountered, replace any OpenAIModel with a model that will raise a ModelHTTPError.
        # Otherwise, do the usual inference.
        def raise_http_error(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ModelHTTPError(401, 'Invalid API Key')

        mock_fallback_models: list[Model] = []
        for m in model.models:
            try:
                from pydantic_ai.models.openai import OpenAIModel
            except ImportError:  # pragma: lax no cover
                OpenAIModel = type(None)

            if isinstance(m, OpenAIModel):
                # Raise an HTTP error for OpenAIModel
                mock_fallback_models.append(FunctionModel(raise_http_error, model_name=m.model_name))
            else:
                mock_fallback_models.append(mock_infer_model(m))
        return FallbackModel(*mock_fallback_models)
    if isinstance(model, (FunctionModel, TestModel)):
        return model
    else:
        model_name = model if isinstance(model, str) else model.model_name
        return FunctionModel(
            model_logic,
            stream_function=stream_model_logic,
            model_name=model_name,
            profile=model.profile if isinstance(model, Model) else None,
        )


def mock_group_by_temporal(aiter: Any, soft_max_interval: float | None) -> Any:
    """Mock group_by_temporal to avoid debouncing, since the iterators above have no delay."""
    return group_by_temporal(aiter, None)


@dataclass
class MockCredentials:
    project_id = 'foobar'
