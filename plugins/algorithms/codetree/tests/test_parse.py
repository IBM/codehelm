# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codetree.generators.parse."""

import pytest
from codetree.generators.parse import add_code_block, parse_code_block, parse_first_func


def test_add_code_block():
    code = "def f(x):\n    return x + 1"
    fenced = add_code_block(code, "python")
    assert fenced == f"```python\n{code}\n```"


def test_parse_code_block_with_language_fence():
    text = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nHope it helps!"
    parsed = parse_code_block(text, "python")
    assert parsed == "def add(a, b):\n    return a + b"


def test_parse_code_block_with_generic_fence():
    text = "Here is the code:\n```\ndef multiply(a, b):\n    return a * b\n```"
    parsed = parse_code_block(text, "python")
    assert parsed == "def multiply(a, b):\n    return a * b"


def test_parse_code_block_with_whitespace_and_newlines():
    text = "```python  \n\ndef add(a, b):\n    return a + b\n\n```"
    parsed = parse_code_block(text, "python")
    assert parsed.strip() == "def add(a, b):\n    return a + b"


def test_parse_code_block_multiple_blocks_returns_first():
    text = (
        "First block:\n```python\ndef first():\n    return 1\n```\n"
        "Second block:\n```python\ndef second():\n    return 2\n```"
    )
    parsed = parse_code_block(text, "python")
    assert parsed == "def first():\n    return 1"


def test_parse_code_block_unfenced_fallback():
    text = "Sure, here is the function:\ndef solve(n):\n    return n ** 2\n\nThis completes the task."
    parsed = parse_code_block(text, "python")
    assert parsed.strip() == "def solve(n):\n    return n ** 2"


def test_parse_first_func_single_function_with_blank_line():
    code = (
        "Some comments\n"
        "def compute(x):\n"
        "    res = x * 2\n"
        "    return res\n"
        "\n"
        "# extra text"
    )
    extracted = parse_first_func(code, "python")
    assert extracted.strip() == "def compute(x):\n    res = x * 2\n    return res"


def test_parse_first_func_single_function_without_blank_line():
    code = "def compute(x):\n    return x * 2"
    extracted = parse_first_func(code, "python")
    assert extracted == "def compute(x):\n    return x * 2"


def test_parse_first_func_no_def_returns_empty():
    code = "x = 1\ny = 2\nprint(x + y)"
    extracted = parse_first_func(code, "python")
    assert extracted == ""


def test_parse_first_func_removes_python_close_tag():
    code = "def fn():\n    return True\n[/PYTHON]"
    extracted = parse_first_func(code, "python")
    assert extracted.rstrip() == "def fn():\n    return True"


def test_parse_first_func_non_python_raises_assertion_error():
    with pytest.raises(AssertionError, match="Only python is supported"):
        parse_first_func("fn main() {}", "rust")
