# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codetree.executors.py_executor."""

import pytest
from codetree.executors.py_executor import (
    ExecuteResult,
    PyExecutor,
    check_syntax_error_source,
    get_call_str,
)


def test_execute_result_namedtuple():
    res = ExecuteResult(is_passing=True, feedback="Passed", state=(True, True))
    assert res.is_passing is True
    assert res.feedback == "Passed"
    assert res.state == (True, True)


def test_check_syntax_error_source_valid():
    error, syn = check_syntax_error_source("def f(x):\n    return x + 1")
    assert error is False
    assert syn is None


def test_check_syntax_error_source_invalid():
    error, syn = check_syntax_error_source("def f(x):\n    return x +")
    assert error is True
    assert isinstance(syn, SyntaxError)


def test_get_call_str_standard_comparison():
    assert get_call_str("assert add(1, 2) == 3") == "add(1, 2)"


def test_get_call_str_unary_or_function_call():
    assert get_call_str("assert is_valid(data)") == "is_valid(data)"


def test_get_call_str_multiline_with_setup():
    test = "a = 5\nb = 10\nassert multiply(a, b) == 50"
    assert get_call_str(test) == "multiply(a, b)"


def test_get_call_str_no_assert():
    test = "def check(x): pass"
    assert get_call_str(test) == "def check(x): pass"


def test_executor_syntax_error_in_function():
    exe = PyExecutor()
    res = exe.execute(
        func="def broken_func(:\n    pass",
        tests=["assert broken_func() == 1", "assert broken_func() == 2"],
    )
    assert res.is_passing is False
    assert res.state == (False, False)
    assert "SyntaxError" in res.feedback or "invalid syntax" in res.feedback or res.feedback == "True"


def test_executor_passing_tests():
    exe = PyExecutor()
    res = exe.execute(
        func="def add(a, b):\n    return a + b",
        tests=["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    )
    assert res.is_passing is True
    assert res.state == (True, True)
    assert "Tested passed:" in res.feedback
    assert "assert add(1, 2) == 3" in res.feedback


def test_executor_failing_assert_test():
    exe = PyExecutor()
    res = exe.execute(
        func="def add(a, b):\n    return a - b",
        tests=["assert add(1, 2) == 3", "assert add(2, 2) == 0"],
    )
    assert res.is_passing is False
    assert res.state == (False, True)
    assert "Tests failed:" in res.feedback
    assert "Tested passed:" in res.feedback
    assert "assert add(1, 2) == 3 # incorrect program's output:" in res.feedback


def test_executor_failing_check_candidate_style():
    exe = PyExecutor()
    test = "def check(candidate):\n    assert candidate(2) == 4\ncheck(double)"
    res = exe.execute(
        func="def double(x):\n    return x + 1",
        tests=[test],
    )
    assert res.is_passing is False
    assert res.state == (False,)
    assert "AssertionError" in res.feedback


def test_executor_timeout_expired():
    exe = PyExecutor()
    func = "import time\ndef infinite_loop():\n    time.sleep(2)\n    return 0"
    res = exe.execute(
        func=func,
        tests=["assert infinite_loop() == 0"],
        timeout=1,
    )
    assert res.is_passing is False
    assert "Program Timed Out after 1 seconds" in res.feedback


def test_executor_with_prev_code():
    exe = PyExecutor()
    prev = "GLOBAL_OFFSET = 10"
    func = "def add_offset(x):\n    return x + GLOBAL_OFFSET"
    res = exe.execute(
        func=func,
        tests=["assert add_offset(5) == 15"],
        prev=prev,
    )
    assert res.is_passing is True


def test_executor_evaluate_method():
    exe = PyExecutor()
    func = "def square(x):\n    return x ** 2"
    pass_test = "assert square(3) == 9"
    fail_test = "assert square(3) == 10"

    assert exe.evaluate("square", func, pass_test) is True
    assert exe.evaluate("square", func, fail_test) is False
    assert exe.evaluate("square", "def broken(:", pass_test) is False
