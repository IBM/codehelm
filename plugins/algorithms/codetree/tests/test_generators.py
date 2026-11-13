# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codetree.generators (generator_utils and py_generate)."""

import pytest
from codetree.generators.generator_utils import (
    generic_evaluate,
    generic_gen_strategy,
    generic_generate_func_impl,
    generic_generate_internal_tests,
    print_generated_func_body,
    sample_n_random,
)
from codetree.generators.parse import add_code_block, parse_code_block
from codetree.generators.py_generate import PyGenerator, py_is_syntax_valid


# ---------------------------------------------------------------------------
# sample_n_random
# ---------------------------------------------------------------------------


def test_sample_n_random_empty_or_zero():
    assert sample_n_random([], 0) == []
    assert sample_n_random(["a", "b"], 0) == []


def test_sample_n_random_subset():
    items = ["a", "b", "c", "d"]
    sampled = sample_n_random(items, 2)
    assert len(sampled) == 2
    assert all(item in items for item in sampled)


def test_sample_n_random_full_or_larger_n():
    items = ["a", "b"]
    assert sample_n_random(items, 2) == items
    assert sample_n_random(items, 5) == items


def test_sample_n_random_negative_raises():
    with pytest.raises(AssertionError):
        sample_n_random(["a"], -1)


# ---------------------------------------------------------------------------
# py_is_syntax_valid
# ---------------------------------------------------------------------------


def test_py_is_syntax_valid():
    assert py_is_syntax_valid("def f(): return 1") is True
    assert py_is_syntax_valid("assert x == 2") is True
    assert py_is_syntax_valid("class A:\n    pass") is True
    assert py_is_syntax_valid("def bad_syntax(:") is False
    assert py_is_syntax_valid("import") is False


# ---------------------------------------------------------------------------
# generic_generate_func_impl & PyGenerator.func_impl
# ---------------------------------------------------------------------------


class MockModel:
    def __init__(self, response="def f(): return 1"):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt, **kwargs):
        self.last_prompt = prompt
        return self.response


def test_generic_generate_func_impl_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid strategy"):
        generic_generate_func_impl(
            func_sig="def f():",
            model=MockModel(),
            strategy="invalid",
            prev_func_impl=None,
            feedback=None,
            self_reflection=None,
            temperature=0.0,
            reflection_chat_instruction="",
            simple_chat_instruction="",
            code_block_instruction="",
            parse_code_block=lambda x: x,
            add_code_block=lambda x: x,
        )


def test_generic_generate_func_impl_reflexion_missing_arguments():
    with pytest.raises(ValueError, match="Invalid arguments: given `strategy=reflexion`"):
        generic_generate_func_impl(
            func_sig="def f():",
            model=MockModel(),
            strategy="reflexion",
            prev_func_impl=None,
            feedback=None,
            self_reflection=None,
            temperature=0.0,
            reflection_chat_instruction="",
            simple_chat_instruction="",
            code_block_instruction="",
            parse_code_block=lambda x: x,
            add_code_block=lambda x: x,
        )


def test_generic_generate_func_impl_simple():
    model = MockModel("```python\ndef foo():\n    return 42\n```")
    gen = PyGenerator()
    code = gen.func_impl("def foo():", model, strategy="simple", temperature=0.0)
    assert code == "def foo():\n    return 42"
    assert len(model.last_prompt) == 2
    assert model.last_prompt[0].role == "system"
    assert model.last_prompt[1].role == "user"


def test_generic_generate_func_impl_reflexion():
    model = MockModel("```python\ndef foo():\n    return 100\n```")
    gen = PyGenerator()
    code = gen.func_impl(
        func_sig="def foo():",
        model=model,
        strategy="reflexion",
        prev_func_impl="def foo(): return 0",
        feedback="AssertionError: 0 != 100",
        self_reflection="Need to return 100",
        temperature=0.0,
    )
    assert code == "def foo():\n    return 100"
    assert len(model.last_prompt) == 6


def test_generic_generate_func_impl_fallback_to_raw_on_empty_parse():
    model = MockModel("raw unparseable string")
    result = generic_generate_func_impl(
        func_sig="def f():",
        model=model,
        strategy="simple",
        prev_func_impl=None,
        feedback=None,
        self_reflection=None,
        temperature=0.0,
        reflection_chat_instruction="",
        simple_chat_instruction="",
        code_block_instruction="",
        parse_code_block=lambda x: "",  # parse returns empty
        add_code_block=lambda x: x,
    )
    assert result == "raw unparseable string"


# ---------------------------------------------------------------------------
# internal_tests
# ---------------------------------------------------------------------------


def test_internal_tests_filters_syntax_and_asserts():
    model = MockModel(
        "assert foo(1) == 2\n"
        "not an assert statement\n"
        "assert foo(2) == 4\n"
        "assert foo(( invalid syntax\n"
    )
    gen = PyGenerator()
    tests = gen.internal_tests("def foo(x):", model, max_num_tests=10)
    assert tests == ["assert foo(1) == 2", "assert foo(2) == 4"]


# ---------------------------------------------------------------------------
# generic_evaluate & agent_eval
# ---------------------------------------------------------------------------


def test_generic_evaluate_invalid_task():
    with pytest.raises(ValueError, match="task not in one of tests, stop"):
        generic_evaluate("def f():", MockModel(), lambda r: r, task="invalid")


def test_py_generator_agent_eval_stop_true():
    model = MockModel("The code satisfies all criteria.\nTrue")
    gen = PyGenerator()
    judge, analysis = gen.agent_eval("def f():", model, task="stop")
    assert judge is True
    assert "satisfies all criteria" in analysis


def test_py_generator_agent_eval_stop_false():
    model = MockModel("The code has boundary bugs.\nFalse")
    gen = PyGenerator()
    judge, analysis = gen.agent_eval("def f():", model, task="stop")
    assert judge is False
    assert "boundary bugs" in analysis


def test_py_generator_agent_eval_stop_empty():
    model = MockModel("")
    gen = PyGenerator()
    judge, analysis = gen.agent_eval("def f():", model, task="stop")
    assert judge is False
    assert analysis == ""


def test_py_generator_agent_eval_tests_scoring():
    model = MockModel("Analysis of output...\nScore: 4")
    gen = PyGenerator()
    score, analysis = gen.agent_eval("def f():", model, task="tests")
    assert score == 4
    assert "Analysis of output" in analysis


# ---------------------------------------------------------------------------
# generic_gen_strategy & PyGenerator.strategy
# ---------------------------------------------------------------------------


def test_generic_gen_strategy_invalid_task():
    with pytest.raises(ValueError, match="Must be in one of strategy/reflection"):
        generic_gen_strategy("def f():", MockModel(), lambda r: [r], task="unknown")


def test_py_generator_strategy_parsing():
    model = MockModel(
        "1. First strategy using dynamic programming\n"
        "2. Second strategy using recursion with memoization\n"
        "<3>Third strategy using greedy approach\n"
        "short\n"
        "\n"
    )
    gen = PyGenerator()
    strategies = gen.strategy("def f():", model, task="strategy", num_strategy=3)
    assert len(strategies) == 3
    assert "First strategy" in strategies[0]
    assert "Second strategy" in strategies[1]
    assert "Third strategy" in strategies[2]


def test_py_generator_reflection_task():
    model = MockModel("1. Fix the off-by-one index error in the loop")
    gen = PyGenerator()
    reflections = gen.strategy(
        func_sig="def f():",
        model=model,
        task="reflection",
        prev_func_impl="def f(): return 0",
        feedback="Failed on [1, 2]",
        given_strategy="iterative",
        num_strategy=1,
    )
    assert len(reflections) == 1
    assert "Fix the off-by-one" in reflections[0]


def test_print_generated_func_body(capsys):
    print_generated_func_body("def foo(): pass")
    captured = capsys.readouterr().out
    assert "def foo(): pass" in captured
    assert "GENERATED FUNC BODY" in captured

    print_generated_func_body("", raw="raw model output")
    captured = capsys.readouterr().out
    assert "WARNING: parse_code_block returned empty" in captured
    assert "raw model output" in captured
