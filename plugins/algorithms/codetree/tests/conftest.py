"""Shared fixtures for the CodeTree plugin tests (the ``codetree`` package must be installed)."""

import json
from pathlib import Path

import pytest

DATASET = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "test_data"
    / "dataset_qiskit_test_human_eval_small.json"
)


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """PyExecutor writes ``temp_files/`` into the current directory; keep it out of the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session")
def sample():
    """First problem of the small Qiskit HumanEval dataset (create_quantum_circuit)."""
    if not DATASET.exists():
        pytest.skip(f"dataset not found: {DATASET}")
    with open(DATASET) as f:
        return json.load(f)[0]


GOOD_CODE = "def create_quantum_circuit(n_qubits):\n    return QuantumCircuit(n_qubits)"
BAD_CODE = "def create_quantum_circuit(n_qubits):\n    return QuantumCircuit(1)"


def fenced(code: str) -> str:
    return f"```python\n{code}\n```"


@pytest.fixture
def good_code():
    return GOOD_CODE


@pytest.fixture
def bad_code():
    return BAD_CODE


class FakeModel:
    """Stands in for LiteLLMModel; routes on the system prompt like the CodeTree agents do.

    ``first_answer`` is returned for the plain-string prompt that BaseRunner sends first;
    every CodeTree "solver" call returns ``fixed_answer``. The "debugger" returns
    ``debugger_answer`` (defaults to ``fixed_answer``). The "tests" scorer ends its reply with
    ``score`` and the reflection prompt returns a one-item numbered list.
    """

    def __init__(self, first_answer=fenced(BAD_CODE), fixed_answer=fenced(GOOD_CODE), tests="",
                 debugger_answer=None, score=0):
        self.first_answer = first_answer
        self.fixed_answer = fixed_answer
        self.debugger_answer = debugger_answer or fixed_answer
        self.tests = tests
        self.score = score
        self.strategies = "1. just build the circuit with n qubits"
        self.review = "True"  # last line of the "stop" reviewer's reply: True accepts, False rejects
        self.calls = []

    def generate(self, prompt, **kwargs):
        if isinstance(prompt, str):
            self.calls.append("plain")
            return self.first_answer
        first = prompt[0]
        system = first.content if hasattr(first, "content") else first["content"]
        if "reflect on problem-solving" in system:
            self.calls.append("reflection")
            return "1. call QuantumCircuit with n_qubits"
        if "evaluate the execution outputs" in system:
            self.calls.append("score")
            return f"the code crashes\n{self.score}"
        if "past function implementation" in system:
            self.calls.append("debug")
            return self.debugger_answer
        if "provides strategy" in system:
            self.calls.append("strategy")
            return self.strategies
        if "further review" in system:
            self.calls.append("stop")
            return f"looks fine\n{self.review}"
        if "unique, diverse, and intuitive unit tests" in system:
            self.calls.append("tests")
            return self.tests
        self.calls.append("impl")
        return self.fixed_answer


@pytest.fixture
def fake_model():
    return FakeModel
