#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from codehelm.algorithms.base import DirectSolution
from codehelm.algorithms.validator_feedback.feeback import Feedback
from codehelm.orchestration.base import BaseRunner


class DummyModel:
    def __init__(self, responses=None):
        self.responses = list(responses or ["default_code"])
        self.call_count = 0
        self.calls = []

    def generate(self, prompt, verbose=False, **kwargs):
        self.call_count += 1
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if self.responses:
            return self.responses.pop(0)
        return "fallback_code"


class DummyValidator:
    def __init__(self, validate_results=None):
        self.validate_results = list(validate_results or [(True, None)])
        self.validations = []
        self.setups = []

    def setup(self, **kwargs):
        self.setups.append(kwargs)

    def validate(self, raw):
        self.validations.append(raw)
        if self.validate_results:
            return self.validate_results.pop(0)
        return True, None


@dataclass
class DummyPrefixPromptResult:
    prompt_for_model: str
    response_prefix: str


def _make_runner(model, validator, strategy, results_dir="results"):
    runner = BaseRunner.__new__(BaseRunner)
    runner.model = model
    runner.results_dir = str(results_dir)
    runner.validator = validator
    runner.repair_strategy = strategy
    runner.verbose = False
    runner.test_format = "exec"
    if strategy and getattr(strategy, "requires_model", False):
        strategy.base_model = model
    return runner


def test_owns_generation_skips_initial_generation_and_validates_code(tmp_path):
    """When strategy has owns_generation=True, runner skips model.generate and validates repair().code."""
    model = DummyModel()
    validator = DummyValidator([(True, None)])

    strategy = MagicMock()
    strategy.owns_generation = True
    strategy.requires_model = False
    strategy.repair.return_value = DirectSolution(code="def solved(): return 42")

    runner = _make_runner(model, validator, strategy, tmp_path)

    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "prompt": "write a function",
                    "test": "assert solved() == 42",
                    "entry_point": "solved",
                }
            ]
        )
    )

    runner.parse_data(str(dataset_path))

    # Model was NOT called because strategy owns generation
    assert model.call_count == 0
    # Strategy setup received test and entry_point
    strategy.setup.assert_called_once()
    _, kwargs = strategy.setup.call_args
    assert kwargs["test"] == "assert solved() == 42"
    assert kwargs["entry_point"] == "solved"
    # Strategy repair called with initial empty failure_info
    strategy.repair.assert_called_once_with("write a function", "")
    # Validator received the code from DirectSolution
    assert validator.validations == ["def solved(): return 42"]


def test_unittest_key_and_sample_dir_forwarded_to_setup(tmp_path):
    """A 'unittest' field is used as the test, and sample_dir is forwarded to strategy.setup."""
    model = DummyModel(["def initial(): pass"])
    validator = DummyValidator([(True, None)])

    strategy = MagicMock()
    strategy.owns_generation = False
    strategy.requires_model = False

    runner = _make_runner(model, validator, strategy, tmp_path)

    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "prompt": "my_prompt",
                    "unittest": "def check(): pass",
                    "entry_point": "my_entry",
                }
            ]
        )
    )

    runner.parse_data(str(dataset_path))

    strategy.setup.assert_called_once()
    _, kwargs = strategy.setup.call_args
    assert kwargs["test"] == "def check(): pass"
    assert kwargs["entry_point"] == "my_entry"
    assert kwargs["sample_dir"].endswith("sample_0")


@pytest.mark.parametrize(
    ("failure_info", "expected"),
    [
        (["AssertionError", "line 5 failed"], "AssertionError\nline 5 failed"),
        ("Entry point 'foo' not defined", "Entry point 'foo' not defined"),
        (None, ""),
    ],
    ids=["list", "str", "none"],
)
def test_failure_info_normalization(tmp_path, failure_info, expected):
    """failure_info is passed to repair() as a string: lists joined, str as-is, None -> ''."""
    model = DummyModel(["def attempt0(): pass", "def attempt1(): pass"])
    validator = DummyValidator([(False, failure_info), (True, None)])

    strategy = MagicMock()
    strategy.owns_generation = False
    strategy.requires_model = False
    strategy.repair.return_value = "fixed prompt"

    runner = _make_runner(model, validator, strategy, tmp_path)

    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(
        json.dumps([{"prompt": "my_prompt", "test": "assert False"}])
    )

    runner.parse_data(str(dataset_path))

    strategy.repair.assert_called_once_with("my_prompt", expected)


def test_direct_solution_inside_retry_loop(tmp_path):
    """When a strategy returns DirectSolution during retries, runner validates code directly and stops retrying."""
    model = DummyModel(["def bad_initial(): pass"])
    validator = DummyValidator([(False, "Error"), (True, None)])

    strategy = MagicMock()
    strategy.owns_generation = False
    strategy.requires_model = False
    strategy.repair.return_value = DirectSolution(code="def direct_fix(): return True")

    runner = _make_runner(model, validator, strategy, tmp_path)

    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(
        json.dumps([{"prompt": "my_prompt", "test": "assert direct_fix()"}])
    )

    runner.parse_data(str(dataset_path))

    assert model.call_count == 1
    assert validator.validations == [
        "def bad_initial(): pass",
        "def direct_fix(): return True",
    ]


def test_prompt_prefix_result_inside_retry_loop(tmp_path):
    """When repair returns an object with prompt_for_model and response_prefix, prefix is prepended."""
    model = DummyModel(["def bad_initial(): pass", "def continuation(): return True"])
    validator = DummyValidator([(False, "Error"), (True, None)])

    strategy = MagicMock()
    strategy.owns_generation = False
    strategy.requires_model = False
    strategy.repair.return_value = DummyPrefixPromptResult(
        prompt_for_model="steered_prompt",
        response_prefix="def prefix():\n    ",
    )

    runner = _make_runner(model, validator, strategy, tmp_path)

    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(
        json.dumps([{"prompt": "my_prompt", "test": "assert True"}])
    )

    runner.parse_data(str(dataset_path))

    assert model.calls[1]["prompt"] == "steered_prompt"
    assert validator.validations == [
        "def bad_initial(): pass",
        "def prefix():\n    def continuation(): return True",
    ]


def test_feedback_strategy_flow(tmp_path):
    """Verify standard Feedback strategy works cleanly with the updated setup kwargs."""
    feedback = Feedback()
    model = DummyModel(["def initial(): pass", "def repaired(): return 1"])
    validator = DummyValidator([(False, "SyntaxError on line 1"), (True, None)])

    runner = _make_runner(model, validator, feedback, tmp_path)

    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(
        json.dumps([{"prompt": "Write foo", "test": "assert foo() == 1"}])
    )

    runner.parse_data(str(dataset_path))

    assert model.call_count == 2
    assert "SyntaxError on line 1" in model.calls[1]["prompt"]
    assert "Write foo" in model.calls[1]["prompt"]
    assert validator.validations == ["def initial(): pass", "def repaired(): return 1"]


def test_none_repair_strategy(tmp_path):
    """When repair_strategy is None, runner completes initial generation without error."""
    model = DummyModel(["def initial(): pass"])
    validator = DummyValidator([(True, None)])

    runner = _make_runner(model, validator, None, tmp_path)

    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(
        json.dumps([{"prompt": "Write foo", "test": "assert True"}])
    )

    runner.parse_data(str(dataset_path))

    assert model.call_count == 1
    assert validator.validations == ["def initial(): pass"]


def test_sample_without_test_field_raises(tmp_path):
    """A sample with neither 'unittest' nor 'test' fails with a clear error."""
    runner = _make_runner(DummyModel(), DummyValidator(), None, tmp_path)

    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(json.dumps([{"prompt": "Write foo"}]))

    with pytest.raises(ValueError, match="neither a 'unittest' nor a 'test'"):
        runner.parse_data(str(dataset_path))
