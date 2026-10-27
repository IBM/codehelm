#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#
# Derived from codespace-optimization/sfs (Apache-2.0); modified for CodeHelm.
#

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, NamedTuple


@dataclass
class Message:
    """Chat message for LLM interactions."""

    role: str
    content: str


@dataclass
class LLMResponse:
    """Response returned by a model adapter."""

    content: str
    logprob: float = 0.0


@dataclass
class StrategyReturn:
    """Return structure for generated code strategy."""

    code: str
    logprob: float = 0.0


class ExecuteResult(NamedTuple):
    """Result of executing test suite on a solution candidate."""

    is_passing: bool
    feedback: str
    state: tuple[bool, ...]


@dataclass
class Problem:
    """Base problem dataclass."""

    task_id: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Problem):
            return NotImplemented
        return self.task_id == other.task_id

    @staticmethod
    def filter_out_unrelated_kwargs(
        clss: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        valid_keys = {f.name for f in fields(clss)}
        return {key: value for key, value in kwargs.items() if key in valid_keys}

    def __init__(self, task_id: str, **kwargs: Any) -> None:
        self.task_id = task_id
        filtered = Problem.filter_out_unrelated_kwargs(self.__class__, kwargs)
        for key, value in filtered.items():
            setattr(self, key, value)


@dataclass
class CodingProblem(Problem):
    """Coding problem specification."""

    prompt: str
    entry_point: str
    visible_tests: tuple[str, ...] = field(default_factory=tuple)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_unwritten_kwargs: dict[str, Any] = field(default_factory=dict)
    test: str = ""
    canonical_solution: str = ""
    base_input: list = field(default_factory=list)
    plus_input: list = field(default_factory=list)
    contract: str = ""
    assertion: str = ""

    def __init__(
        self,
        task_id: str,
        prompt: str,
        entry_point: str = "",
        test: str = "",
        visible_tests: tuple[str, ...] | None = None,
        canonical_solution: str = "",
        base_input: list | None = None,
        plus_input: list | None = None,
        contract: str = "",
        assertion: str = "",
        **kwargs: Any,
    ):
        super().__init__(task_id, **kwargs)
        self.prompt = prompt
        self.entry_point = entry_point
        self.test = test
        self.visible_tests = visible_tests if visible_tests is not None else ()
        self.canonical_solution = canonical_solution
        self.base_input = base_input if base_input is not None else []
        self.plus_input = plus_input if plus_input is not None else []
        self.contract = contract
        self.assertion = assertion
        self.extra_kwargs = kwargs
        self.extra_unwritten_kwargs = {}


@dataclass
class CodingProblemAndSolution(CodingProblem):
    """A coding problem together with a candidate solution and evaluation state."""

    solution: str | None = None
    is_solved: bool = False
    is_passing: bool = False
    feedback_string: str = ""
    self_reflection: str = ""
    test_results: tuple[bool, ...] = field(default_factory=tuple)
    generation: int = 0
    pass_on_first_try: bool = False

    @staticmethod
    def init_from_coding_problem(problem: CodingProblem) -> CodingProblemAndSolution:
        return CodingProblemAndSolution(
            task_id=problem.task_id,
            prompt=problem.prompt,
            entry_point=problem.entry_point,
            test=problem.test,
            visible_tests=problem.visible_tests,
            canonical_solution=problem.canonical_solution,
            base_input=problem.base_input,
            plus_input=problem.plus_input,
            extra_kwargs=dict(problem.extra_kwargs),
        )

    def __hash__(self) -> int:
        assert isinstance(self.prompt, str), f"Prompt is not a string: {self.prompt}"
        assert isinstance(self.solution, str), (
            f"Solution is not a string: {self.solution}"
        )
        return hash((self.prompt, self.solution))

    def get_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "visible_tests": self.visible_tests,
            "solution": self.solution,
            "is_solved": self.is_solved,
            "is_passing": self.is_passing,
            "feedback_string": self.feedback_string,
            "self_reflection": self.self_reflection,
            "test_results": self.test_results,
            "generation": self.generation,
            "pass_on_first_try": self.pass_on_first_try,
            "extra_kwargs": self.extra_kwargs,
        }

    def get_prop_test_passed(self) -> float:
        if not self.visible_tests:
            return 0.0
        return sum(self.test_results) / len(self.visible_tests)
