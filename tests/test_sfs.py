#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import json

from codehelm.algorithms.sfs import SFSRepair
from codehelm.algorithms.sfs.code_safety import check_code_safety
from codehelm.algorithms.sfs.executor import PyExecutor
from codehelm.algorithms.sfs.model_adapter import ModelAdapter
from codehelm.algorithms.sfs.parse import (
    parse_code_block,
    parse_improvement_ideas,
    parse_tests,
)
from codehelm.algorithms.sfs.prompts_python import PyGenerator
from codehelm.algorithms.sfs.search import Graph
from codehelm.algorithms.sfs.synthesis import SynthesisSolver
from codehelm.algorithms.sfs.types import (
    CodingProblem,
    CodingProblemAndSolution,
)
from codehelm.orchestration.base import BaseRunner


def test_parse_code_block():
    code_md = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nExplanation."
    assert parse_code_block(code_md, "python") == "def add(a, b):\n    return a + b"

    code_generic = "```\ndef foo():\n    return 1\n```"
    assert parse_code_block(code_generic, "python") == "def foo():\n    return 1"

    code_raw = "def bar():\n    return 2"
    assert parse_code_block(code_raw, "python") == "def bar():\n    return 2"


def test_parse_tests():
    single_asserts = "assert add(1, 2) == 3\nassert add(0, 0) == 0"
    parsed = parse_tests(single_asserts)
    assert len(parsed) == 2
    assert parsed[0] == "assert add(1, 2) == 3"

    multi_line = "x = [1, 2]\nassert len(x) == 2\n\ny = [3]\nassert len(y) == 1"
    parsed_multi = parse_tests(multi_line)
    assert len(parsed_multi) == 2
    assert "x = [1, 2]" in parsed_multi[0]

    no_assert = "import os\nx = 10"
    assert parse_tests(no_assert) == []


def test_parse_improvement_ideas():
    text_pattern1 = (
        "Reflection: failed test.\nIdea 1: Use a dictionary.\nIdea 2: Use recursion."
    )
    ideas1 = parse_improvement_ideas(text_pattern1, 2)
    assert len(ideas1) == 2
    assert "Use a dictionary." in ideas1[0]
    assert "Use recursion." in ideas1[1]

    text_pattern2 = "1. **Check boundaries**\n2. **Handle empty list**"
    ideas2 = parse_improvement_ideas(text_pattern2, 2)
    assert len(ideas2) == 2
    assert ideas2[0] == "Check boundaries"

    text_short = "Idea 1: Only one idea provided."
    ideas_padded = parse_improvement_ideas(text_short, 3)
    assert len(ideas_padded) == 3
    assert ideas_padded[2] == "Improve the code."


def test_code_safety_scanner():
    safe_code = "def add(a, b):\n    return a + b"
    assert check_code_safety(safe_code).is_safe

    dangerous_exec = "def bad():\n    exec('import os')"
    report = check_code_safety(dangerous_exec)
    assert not report.is_safe
    assert any("exec" in str(f) for f in report.flags)

    dangerous_popen = "import subprocess\ndef bad():\n    subprocess.Popen(['ls'])"
    report2 = check_code_safety(dangerous_popen)
    assert not report2.is_safe


def test_executor_import_hoisting_and_execution():
    executor = PyExecutor(safe_mode=False)

    problem = CodingProblem(
        task_id="t1",
        prompt="from math import sqrt\ndef hypotenuse(a, b):\n    pass",
        entry_point="hypotenuse",
    )
    sol = CodingProblemAndSolution.init_from_coding_problem(problem)
    # Solution uses sqrt without explicit import in solution body
    sol.solution = "def hypotenuse(a, b):\n    return sqrt(a*a + b*b)"
    sol.visible_tests = ("assert hypotenuse(3, 4) == 5.0",)

    result = executor.execute(sol)
    assert result.is_passing
    assert result.state == (True,)


def test_executor_test_function_invocation():
    executor = PyExecutor(safe_mode=False)

    problem = CodingProblem(
        task_id="t2",
        prompt="def is_even(n):\n    pass",
        entry_point="is_even",
        test="def test_is_even():\n    assert is_even(2) is True\n    assert is_even(3) is False",
    )
    sol = CodingProblemAndSolution.init_from_coding_problem(problem)
    sol.solution = "def is_even(n):\n    return n % 2 == 0"

    is_solved = executor.evaluate(sol)
    assert is_solved is True
    assert sol.is_solved is True


def test_search_graph_and_node():
    problem = CodingProblem(
        task_id="test_mcts", prompt="def f(): pass", entry_point="f"
    )
    sol1 = CodingProblemAndSolution.init_from_coding_problem(problem)
    sol1.solution = "def f(): return 1"
    sol1.visible_tests = ("assert f() == 1", "assert f() == 2")
    sol1.test_results = (True, False)

    sol2 = CodingProblemAndSolution.init_from_coding_problem(problem)
    sol2.solution = "def f(): return 2"
    sol2.visible_tests = ("assert f() == 1", "assert f() == 2")
    sol2.test_results = (True, True)

    graph = Graph.init_from_seed_solutions([sol1])
    assert len(graph.nodes) == 1

    child_node = graph.add_child_solution(
        parent_solution=sol1,
        action="Idea 1",
        solution=sol2,
        score=1.0,
    )
    assert child_node.score == 1.0
    assert child_node.depth == 2

    child_node.backpropogate()
    assert graph.get_best_score_solution() == sol2


class ScriptedFakeModel:
    """Deterministic fake model for testing SFS Synthesis loop without network."""

    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0
        self.output_tokens = 0
        self.total_requests = 0

    def generate(self, prompt, **kwargs):
        self.call_count += 1
        prompt_text = str(prompt)

        # Seed generation
        if "[object description]:" in prompt_text and "[object impl]:" in prompt_text:
            return "```python\ndef solve_task(x):\n    return x + 1\n```"

        # Scout idea generation
        if "[general insights]:" in prompt_text or "improvement ideas" in prompt_text:
            return "Reflection: Off by one.\nIdea 1: Multiply instead of add.\nIdea 2: Return x * 2."

        # Strategist implementation
        if "[improvement idea]:" in prompt_text:
            return "```python\ndef solve_task(x):\n    return x * 2\n```"

        # Insight update
        if "[updated insights]:" in prompt_text or "Reflect on how" in prompt_text:
            return "*Reflections*\nMultiplication worked.\n*Updated Insights*\nAlways check multipliers."

        return "```python\ndef solve_task(x):\n    return x\n```"


def test_synthesis_solver_full_loop():
    fake_model = ScriptedFakeModel()
    adapter = ModelAdapter(fake_model)
    generator = PyGenerator()
    executor = PyExecutor(safe_mode=False)

    solver = SynthesisSolver(
        generator=generator,
        model=adapter,
        executor=executor,
        max_iters=2,
        num_seeds=1,
        temperature=0.0,
        use_problem_tests=True,
    )

    problem = CodingProblem(
        task_id="task_double",
        prompt="def solve_task(x):\n    '''Double x.'''",
        entry_point="solve_task",
        test="assert solve_task(3) == 6",
    )

    best = solver.solve(problem)
    assert best is not None
    assert "return x * 2" in (best.solution or "")


def test_sfs_repair_runner_integration(tmp_path):
    class LocalDummyValidator:
        def __init__(self):
            self.validations = []

        def setup(self, **kwargs):
            pass

        def validate(self, raw):
            self.validations.append(raw)
            return True, None

    fake_model = ScriptedFakeModel()
    strategy = SFSRepair(
        generator="python",
        model=fake_model,
        max_iters=2,
        num_seeds=1,
        use_problem_tests=True,
    )

    validator = LocalDummyValidator()

    runner = BaseRunner.__new__(BaseRunner)
    runner.model = fake_model
    runner.results_dir = str(tmp_path / "results")
    runner.validator = validator
    runner.repair_strategy = strategy
    runner.verbose = False
    runner.test_format = "exec"

    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "prompt": "def solve_task(x):\n    '''Double x.'''",
                    "test": "assert solve_task(3) == 6",
                    "entry_point": "solve_task",
                }
            ]
        )
    )

    runner.parse_data(str(dataset_path))

    # Validator received the synthesized code
    assert len(validator.validations) == 1
    assert "solve_task" in validator.validations[0]


def test_solve_writes_graph_json(tmp_path):
    solver = SynthesisSolver(
        generator=PyGenerator(),
        model=ModelAdapter(ScriptedFakeModel()),
        executor=PyExecutor(safe_mode=False),
        max_iters=2,
        num_seeds=1,
        temperature=0.0,
        use_problem_tests=True,
    )
    problem = CodingProblem(
        task_id="task_double",
        prompt="def solve_task(x):\n    '''Double x.'''",
        entry_point="solve_task",
        test="assert solve_task(3) == 6",
    )

    solver.solve(problem, results_path=str(tmp_path))

    graph = json.loads((tmp_path / "graph.json").read_text())
    ids = {n["id"] for n in graph["nodes"]}
    assert len(graph["nodes"]) > 1
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    assert sum(n["is_best"] for n in graph["nodes"]) == 1


def test_model_adapter_appends_message_log(tmp_path):
    log_path = tmp_path / "sample_0" / "messages.jsonl"
    adapter = ModelAdapter(ScriptedFakeModel(), log_path=str(log_path))

    adapter.generate("first prompt")
    adapter.generate("second prompt")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [r["request_num"] for r in records] == [1, 2]
    assert records[0]["messages"] == [{"role": "user", "content": "first prompt"}]
    assert records[0]["response"]


def test_apply_problem_tests_uses_whole_test_as_single_oracle():
    solver = SynthesisSolver(
        generator=PyGenerator(),
        model=ModelAdapter(ScriptedFakeModel()),
        executor=PyExecutor(safe_mode=False),
        use_problem_tests=True,
    )
    problem = CodingProblem(
        task_id="t",
        prompt="def f(): pass",
        entry_point="f",
        test="assert f() == 1\nassert f() == 2",
    )

    solver.apply_problem_tests(problem)

    assert problem.visible_tests == ("assert f() == 1\nassert f() == 2",)
