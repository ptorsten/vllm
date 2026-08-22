# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tool_choice enforcement must survive the shared-engine shortcut.

``ParserManager.get_parser`` returns the parser engine class directly when the
reasoning and tool parsers resolve to the same engine (e.g. ``qwen3`` +
``qwen3_xml``).  The structural tag that enforces ``tool_choice="required"``,
named tool choice, and strict tools used to be applied only by
``Parser.adjust_request`` through a composed tool parser, so the shortcut
silently disabled enforcement: a request with ``tool_choice="required"``
could answer in plain text.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.parser_manager import ParserManager

_VOCAB = {
    "<think>": 50,
    "</think>": 51,
    "<tool_call>": 60,
    "</tool_call>": 61,
}

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]


def _make_request(tool_choice) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "What is 7 times 8?"}],
        tools=_TOOLS,
        tool_choice=tool_choice,
    )


@pytest.fixture()
def shared_engine_parser():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="qwen3_xml",
        reasoning_parser_name="qwen3",
        enable_auto_tools=True,
    )
    return parser_cls(make_mock_tokenizer(_VOCAB))


def test_shared_engine_carries_structural_tag_model(shared_engine_parser):
    # The tool adapter declares qwen_3_coder; the shared engine must too.
    assert shared_engine_parser.structural_tag_model == "qwen_3_coder"


def test_required_tool_choice_sets_structured_outputs(shared_engine_parser):
    request = _make_request("required")
    request = shared_engine_parser.adjust_request(request)

    assert request.structured_outputs is not None
    tag = json.loads(request.structured_outputs.structural_tag)
    # required maps to a grammar that cannot terminate without a call.
    assert tag["format"]["type"] == "tags_with_separator"
    assert tag["format"]["at_least_one"] is True


def test_named_tool_choice_sets_structured_outputs(shared_engine_parser):
    request = _make_request({"type": "function", "function": {"name": "calculator"}})
    request = shared_engine_parser.adjust_request(request)
    assert request.structured_outputs is not None


def test_auto_without_strict_tools_is_unconstrained(shared_engine_parser):
    # auto only builds a grammar when a tool is strict.
    request = _make_request("auto")
    request = shared_engine_parser.adjust_request(request)
    assert request.structured_outputs is None


def test_none_tool_choice_is_unconstrained(shared_engine_parser):
    request = _make_request("none")
    request = shared_engine_parser.adjust_request(request)
    assert request.structured_outputs is None


def test_enforcement_respects_env_kill_switch(shared_engine_parser, monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_ENFORCE_STRICT_TOOL_CALLING", False)
    request = _make_request("required")
    request = shared_engine_parser.adjust_request(request)
    assert request.structured_outputs is None
