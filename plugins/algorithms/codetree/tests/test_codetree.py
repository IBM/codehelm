"""Tests for the CodeTree plugin (AgentGuide) wired into codehelm's BaseRunner.

All tests run offline: the LLM is replaced by a fake model (see conftest.py).
Run from the repository root:  pytest codehelm/plugins/algorithms/codetree/tests
"""

import json

import pytest
from codetree.methods.llm_agent_guide import AgentGuide, gen_test_eval
from codetree.results_summary import STATS_FILE, SUMMARY_FILE, summarize_results

from codehelm.orchestration.base import BaseRunner, FinalResponse
from codehelm.validator.validator_exec import ValidatorExec

GENERATED_TESTS = (
    "assert create_quantum_circuit(3).num_qubits == 3\n"
    "assert create_quantum_circuit(1).num_qubits == 1\n"
    "not a test\n"
    "assert (("  # invalid syntax: filtered out by internal_tests
)


def _setup(guide, sample, sample_dir="unused", **kwargs):
    guide.setup(
        str(sample_dir),
        test=sample["test"],
        entry_point=sample["entry_point"],
        prompt=sample["prompt"],
        **kwargs,
    )
    return guide


# ---------------------------------------------------------------------------
# AgentGuide.setup: where the visible tests come from
# ---------------------------------------------------------------------------


def test_setup_dataset_mode_builds_check_runner(sample):
    guide = _setup(AgentGuide(visible_tests="dataset"), sample)
    assert len(guide.item["given_tests"]) == 1
    assert guide.item["given_tests"][0].rstrip().endswith("check(create_quantum_circuit)")
    # the code-stub prompt is passed as `prev` so its imports are available when tests run
    assert guide.item["prev"] == sample["prompt"]
    assert guide.item["test"] == sample["test"]


def test_setup_generated_mode_excludes_dataset_test(sample, fake_model):
    guide = AgentGuide(visible_tests="generated")
    guide.model = fake_model(tests=GENERATED_TESTS)
    _setup(guide, sample)
    assert len(guide.item["given_tests"]) == 2
    assert all("def check(" not in t for t in guide.item["given_tests"])
    # the hidden test is still available for scoring
    assert guide.item["test"] == sample["test"]


def test_setup_combined_mode(sample, fake_model):
    guide = AgentGuide(visible_tests="dataset+generated")
    guide.model = fake_model(tests=GENERATED_TESTS)
    _setup(guide, sample)
    assert len(guide.item["given_tests"]) == 3
    assert any("check(create_quantum_circuit)" in t for t in guide.item["given_tests"])


def test_dataset_mode_does_not_call_the_model_for_tests(sample, fake_model):
    guide = AgentGuide(visible_tests="dataset")
    guide.model = fake_model()
    _setup(guide, sample)
    assert guide.model.calls == []


def test_invalid_visible_tests_value_is_rejected():
    with pytest.raises(ValueError, match="visible_tests"):
        AgentGuide(visible_tests="bogus")


def test_setup_stops_when_no_tests_are_available(fake_model):
    guide = AgentGuide(visible_tests="dataset")
    guide.model = fake_model()
    with pytest.raises(ValueError, match="given_tests"):
        guide.setup("unused")  # no test / entry_point -> nothing to run


def test_setup_stops_when_generated_tests_are_all_unusable(sample, fake_model):
    guide = AgentGuide(visible_tests="generated")
    guide.model = fake_model(tests="no asserts here")
    with pytest.raises(ValueError, match="given_tests"):
        _setup(guide, sample)


def test_setup_filters_assert_false_tests():
    guide = AgentGuide()
    with pytest.raises(ValueError, match="given_tests"):
        guide.setup("unused", item={"given_tests": ["assert False"]})


# ---------------------------------------------------------------------------
# PyExecutor on check(candidate)-style tests
# ---------------------------------------------------------------------------


def test_executor_passes_correct_solution(sample, good_code):
    guide = _setup(AgentGuide(), sample)
    passing, feedback, reward = gen_test_eval(
        guide.exe, good_code, guide.item["given_tests"], prev=guide.item["prev"]
    )
    assert passing is True
    assert reward == 1
    assert "Tested passed" in feedback


def test_executor_reports_assertion_traceback_for_wrong_solution(sample, bad_code):
    guide = _setup(AgentGuide(), sample)
    passing, feedback, reward = gen_test_eval(
        guide.exe, bad_code, guide.item["given_tests"], prev=guide.item["prev"]
    )
    assert passing is False
    assert reward == 0.0
    assert "AssertionError" in feedback
    assert "Expected 3 qubits, got 1" in feedback


# ---------------------------------------------------------------------------
# repair() hands back a finished answer instead of a prompt
# ---------------------------------------------------------------------------


def test_repair_returns_final_response(sample, fake_model, tmp_path):
    guide = AgentGuide()
    guide.model = fake_model()
    _setup(guide, sample, tmp_path)
    result = guide.repair(sample["prompt"], "failure")
    assert isinstance(result, FinalResponse)
    assert result.text.startswith("```python\n")
    assert "QuantumCircuit(n_qubits)" in result.text


# ---------------------------------------------------------------------------
# Per-search stats (codetree_stats.json)
# ---------------------------------------------------------------------------


def _read_stats(sample_dir):
    return json.loads((sample_dir / STATS_FILE).read_text())["rounds"]


def test_repair_records_stats_for_solved_search(sample, fake_model, tmp_path):
    guide = AgentGuide()
    guide.model = fake_model()
    _setup(guide, sample, tmp_path)
    guide.repair(sample["prompt"], "failure")
    (round_,) = _read_stats(tmp_path)
    assert round_ == {
        "num_try": 1,
        "max_depth": 3,
        "search_width": 10,
        "max_iters": 20,
        "exhausted": False,
        "passed_visible": True,
        "weak_solved": True,
        "end_reason": "accepted",
        "trace": [{"depth": 1, "reward": 1, "go_on": False, "reason": "accepted", "children": 0}],
    }


def test_repair_records_exhausted_search_without_crashing(sample, fake_model, bad_code, tmp_path):
    """Regression: the exhausted counter used to raise UnboundLocalError."""
    from conftest import fenced

    guide = AgentGuide()
    guide.model = fake_model(fixed_answer=fenced(bad_code))
    _setup(guide, sample, tmp_path)
    result = guide.repair(sample["prompt"], "failure", max_iters=1, max_depth=1, search_width=1)
    assert isinstance(result, FinalResponse)
    (round_,) = _read_stats(tmp_path)
    assert round_["num_try"] == 1
    assert round_["exhausted"] is True
    assert round_["passed_visible"] is False
    assert round_["weak_solved"] is False
    assert round_["end_reason"] == "max_iters"


# ---------------------------------------------------------------------------
# Search budget (max_depth / search_width / max_iters) comes from the constructor
# ---------------------------------------------------------------------------


def test_search_budget_defaults_are_the_original_codetree_values():
    guide = AgentGuide()
    assert (guide.max_depth, guide.search_width, guide.max_iters) == (3, 10, 20)


def test_constructor_search_budget_is_used_and_recorded(sample, fake_model, bad_code, tmp_path):
    """BaseRunner calls repair() with no budget arguments, so the constructor must supply it."""
    from conftest import fenced

    guide = AgentGuide(max_depth=2, search_width=1, max_iters=1)
    model = guide.model = fake_model(fixed_answer=fenced(bad_code))
    _setup(guide, sample, tmp_path)
    guide.repair(sample["prompt"], "failure")  # no budget arguments, as BaseRunner does it
    (round_,) = _read_stats(tmp_path)
    assert (round_["max_depth"], round_["search_width"], round_["max_iters"]) == (2, 1, 1)
    assert round_["num_try"] == 1
    assert round_["exhausted"] is True
    assert model.calls.count("impl") == 1


def test_repair_arguments_override_the_constructor_budget(sample, fake_model, bad_code, tmp_path):
    from conftest import fenced

    guide = AgentGuide(max_iters=20)
    guide.model = fake_model(fixed_answer=fenced(bad_code))
    _setup(guide, sample, tmp_path)
    guide.repair(sample["prompt"], "failure", max_iters=1)
    (round_,) = _read_stats(tmp_path)
    assert round_["max_iters"] == 1


def test_search_width_limits_the_number_of_first_attempts(sample, fake_model, bad_code, tmp_path):
    from conftest import fenced

    model = fake_model(fixed_answer=fenced(bad_code))
    model.strategies = "1. first idea here\n2. second idea here\n3. third idea here"
    guide = AgentGuide(search_width=2)
    guide.model = model
    _setup(guide, sample, tmp_path)
    guide.repair(sample["prompt"], "failure")
    assert model.calls.count("impl") == 2


@pytest.mark.parametrize("name", ["max_depth", "search_width", "max_iters"])
@pytest.mark.parametrize("value", [0, -1, 2.5, "3", None, True])
def test_invalid_search_budget_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        AgentGuide(**{name: value})


def test_search_that_runs_out_of_nodes_is_not_reported_as_exhausted(
    sample, fake_model, bad_code, tmp_path
):
    """The failure seen on a real run: a wrong answer, the evaluator prunes the only node, and the
    search stops after one try even though 20 were allowed."""
    from conftest import fenced

    guide = AgentGuide()
    guide.model = fake_model(fixed_answer=fenced(bad_code))
    _setup(guide, sample, tmp_path)
    guide.repair(sample["prompt"], "failure")  # default max_iters=20
    (round_,) = _read_stats(tmp_path)
    assert round_["num_try"] == 1
    assert round_["exhausted"] is False
    assert round_["passed_visible"] is False
    assert round_["end_reason"] == "no_nodes_left"
    (node,) = round_["trace"]
    assert node["go_on"] is False
    assert node["reason"] == "value_equals_parent_low_reward"
    assert node["children"] == 0


def test_answer_rejected_by_the_reviewer_is_expanded_not_dropped(sample, fake_model, tmp_path):
    """An answer that passes every visible test but is rejected by the reviewer has reward 1 and
    is_passing False. It is kept (as in the original CodeTree) and the search goes deeper; with
    max_depth=3 the node at the last level is accepted once its parent had passed the visible tests."""
    model = fake_model()
    model.review = "False"
    guide = AgentGuide()
    guide.model = model
    _setup(guide, sample, tmp_path)
    guide.repair(sample["prompt"], "failure")
    (round_,) = _read_stats(tmp_path)
    assert [(n["depth"], n["reason"]) for n in round_["trace"]] == [
        (1, "passed_visible_rejected_by_review"),
        (2, "passed_visible_rejected_by_review"),
        (3, "accepted"),
    ]
    assert round_["end_reason"] == "accepted"
    assert round_["num_try"] == 3
    assert "debug" in model.calls


def test_eval_node_reports_the_rule_that_decided(fake_model):
    from codetree.generators.py_generate import PyGenerator
    from codetree.methods.llm_agent_guide import eval_node
    from codetree.node import Node

    node = Node("code", depth=3)
    go_on, _, _, reason = eval_node("p", node, PyGenerator(), fake_model(), max_depth=3)
    assert (go_on, reason) == (False, "max_depth")


# ---------------------------------------------------------------------------
# prune_rule: what happens to a first attempt that fails every visible test
# ---------------------------------------------------------------------------


def test_invalid_prune_rule_is_rejected():
    with pytest.raises(ValueError, match="prune_rule"):
        AgentGuide(prune_rule="nope")


def test_original_rule_prunes_a_failing_root_so_the_debugger_never_runs(
    sample, fake_model, bad_code, good_code, tmp_path
):
    from conftest import fenced

    guide = AgentGuide(prune_rule="original")
    model = guide.model = fake_model(fixed_answer=fenced(bad_code), debugger_answer=fenced(good_code))
    _setup(guide, sample, tmp_path)
    guide.repair(sample["prompt"], "failure")
    (round_,) = _read_stats(tmp_path)
    assert round_["end_reason"] == "no_nodes_left"
    assert {node["reason"] for node in round_["trace"]} == {"value_equals_parent_low_reward"}
    assert "debug" not in model.calls


def test_keep_failing_roots_lets_the_debugger_fix_the_root(
    sample, fake_model, bad_code, good_code, tmp_path
):
    from conftest import fenced

    guide = AgentGuide(prune_rule="keep_failing_roots")
    model = guide.model = fake_model(fixed_answer=fenced(bad_code), debugger_answer=fenced(good_code))
    _setup(guide, sample, tmp_path)
    result = guide.repair(sample["prompt"], "failure")
    (round_,) = _read_stats(tmp_path)
    assert round_["end_reason"] == "accepted"
    assert round_["passed_visible"] is True
    assert round_["trace"][0] == {
        "depth": 1, "reward": 0.0, "go_on": True, "reason": "kept_failing_root", "children": 1,
    }
    assert "debug" in model.calls
    assert "QuantumCircuit(n_qubits)" in result.text


def test_keep_failing_roots_only_exempts_roots(fake_model):
    from codetree.generators.py_generate import PyGenerator
    from codetree.methods.llm_agent_guide import eval_node
    from codetree.node import Node

    gen = PyGenerator()
    root = Node("code", depth=1)
    go_on, _, _, reason = eval_node("p", root, gen, fake_model(), keep_failing_roots=True)
    assert (go_on, reason) == (True, "kept_failing_root")

    parent = Node("code", depth=1)
    parent.value = 1
    child = Node("code", parent=parent, depth=2)
    go_on, _, _, reason = eval_node("p", child, gen, fake_model(), keep_failing_roots=True)
    assert (go_on, reason) == (False, "value_below_parent")

    child = Node("code", parent=Node("code", depth=1), depth=2)
    go_on, _, _, reason = eval_node("p", child, gen, fake_model(), keep_failing_roots=True)
    assert (go_on, reason) == (False, "value_equals_parent_low_reward")


def test_each_repair_round_is_appended_and_setup_resets(sample, fake_model, tmp_path):
    guide = AgentGuide()
    guide.model = fake_model()
    _setup(guide, sample, tmp_path)
    guide.repair(sample["prompt"], "failure")
    guide.repair(sample["prompt"], "failure")
    assert len(_read_stats(tmp_path)) == 2
    _setup(guide, sample, tmp_path)  # next sample starts clean
    guide.repair(sample["prompt"], "failure")
    assert len(_read_stats(tmp_path)) == 1


# ---------------------------------------------------------------------------
# parse_first_func: fallback extraction when the reply has no code fence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "last_line",
    ["return N", "return [1, 2, 3]", "return HELLO", "return PYTHON", "return x[0]"],
)
def test_unfenced_code_keeps_trailing_characters_of_the_last_line(last_line):
    """Regression: rstrip("[/PYTHON]") stripped any trailing run of those letters, not the tag."""
    from codetree.generators.parse import parse_code_block

    code = f"def f(x):\n    {last_line}"
    assert parse_code_block(code, "python") == code


def test_unfenced_code_still_drops_a_trailing_python_tag():
    from codetree.generators.parse import parse_code_block

    reply = "def f(x):\n    return 1\n[/PYTHON]"
    assert parse_code_block(reply, "python").rstrip() == "def f(x):\n    return 1"


def test_fenced_code_does_not_use_the_fallback():
    from codetree.generators.parse import parse_code_block

    reply = "text\n```python\ndef f(x):\n    return N\n```\nmore text"
    assert parse_code_block(reply, "python") == "def f(x):\n    return N"


# ---------------------------------------------------------------------------
# summarize_results: joins validator_result.json and codetree_stats.json
# ---------------------------------------------------------------------------


def _make_sample(results_dir, n, passed, rounds=None):
    d = results_dir / f"sample_{n}"
    d.mkdir(parents=True)
    (d / "validator_result.json").write_text(json.dumps({"passed": passed}))
    if rounds is not None:
        (d / STATS_FILE).write_text(json.dumps({"rounds": rounds}))


def _round(num_try, exhausted=False, passed_visible=True, weak_solved=True, end_reason=None):
    round_ = {"num_try": num_try, "max_iters": 20, "exhausted": exhausted,
              "passed_visible": passed_visible, "weak_solved": weak_solved}
    if end_reason:  # rounds written before this field existed have none
        round_["end_reason"] = end_reason
    return round_


def test_summarize_results_matches_original_metrics(tmp_path):
    _make_sample(tmp_path, 0, True)                                                   # zero-shot hit
    _make_sample(tmp_path, 1, True, [_round(3)])                                      # solved by search
    _make_sample(tmp_path, 2, False, [_round(20, True, passed_visible=False,
                                             weak_solved=False)])                     # exhausted
    _make_sample(tmp_path, 3, False, [_round(2)])                                     # passed visible only
    _make_sample(tmp_path, 4, True, [_round(20, True, passed_visible=False),
                                     _round(5)])                                      # solved on retry
    _make_sample(tmp_path, 10, False, [_round(4)])                                    # numeric (not text) ordering

    summary = summarize_results(tmp_path, verbose=False)

    assert summary["samples"] == 6
    assert summary["solved"] == 3                      # samples 0, 1, 4
    assert summary["acc"] == 0.5
    assert summary["weak_acc"] == pytest.approx(5 / 6, abs=1e-3)   # all but sample 2
    assert summary["solved_before_codetree"] == 1
    assert summary["passed_at_sample"] == [3, 5]       # last search of each solved sample
    assert summary["passed_visible_not_solved"] == 2   # samples 3 and 10
    assert summary["searches"] == 6
    assert summary["exhausted_searches"] == 2          # per search: sample 2, and sample 4's first
    assert summary["searches_without_answer"] == 2     # same two: the rest were accepted
    assert summary["end_reasons"] == {"unknown": 6}    # old-format rounds carry no end_reason
    assert json.loads((tmp_path / SUMMARY_FILE).read_text()) == summary


def test_summarize_results_separates_early_stops_from_exhausted_searches(tmp_path):
    """Mirrors the real run: a sample failing three searches that each stopped after 3 tries."""
    stuck = _round(3, passed_visible=False, weak_solved=False, end_reason="no_nodes_left")
    _make_sample(tmp_path, 0, False, [stuck, stuck, stuck])
    _make_sample(tmp_path, 1, True, [_round(2, end_reason="accepted")])
    _make_sample(tmp_path, 2, False, [_round(20, True, passed_visible=False, weak_solved=False,
                                             end_reason="max_iters")])

    summary = summarize_results(tmp_path, verbose=False)

    assert summary["searches"] == 5
    assert summary["exhausted_searches"] == 1            # only the max_iters search
    assert summary["searches_without_answer"] == 4       # the stuck ones count as well
    assert summary["end_reasons"] == {"no_nodes_left": 3, "accepted": 1, "max_iters": 1}


def test_summarize_results_with_no_samples(tmp_path):
    summary = summarize_results(tmp_path, verbose=False)
    assert summary["samples"] == 0
    assert summary["acc"] == 0.0


# ---------------------------------------------------------------------------
# End to end through BaseRunner + ValidatorExec
# ---------------------------------------------------------------------------


def _run(tmp_path, sample, model, guide):
    dataset = tmp_path / "one.json"
    dataset.write_text(json.dumps([sample]))
    results_dir = tmp_path / "results"
    BaseRunner._build_model = staticmethod(lambda cfg: model)
    runner = BaseRunner(
        {"model": {}, "test_format": "human_eval"},
        results_dir=str(results_dir),
        validator=ValidatorExec(),
        repair_strategy=guide,
    )
    runner.parse_data(str(dataset), 1)
    return json.loads((results_dir / "sample_0" / "validator_result.json").read_text())


@pytest.fixture
def restore_build_model():
    original = BaseRunner._build_model
    yield
    BaseRunner._build_model = original


def test_end_to_end_search_fixes_wrong_first_answer(
    tmp_path, sample, fake_model, restore_build_model
):
    model = fake_model()
    result = _run(tmp_path, sample, model, AgentGuide())
    assert result == {"passed": True}
    # The tree-search answer is validated directly: no generation after the "stop" review.
    assert model.calls == ["plain", "strategy", "impl", "stop"]
    # ...and the run can be summarised from what the pipeline wrote.
    summary = summarize_results(tmp_path / "results", verbose=False)
    assert summary["acc"] == 1.0
    assert summary["passed_at_sample"] == [1]
    assert summary["solved_before_codetree"] == 0


def test_end_to_end_no_repair_when_first_answer_passes(
    tmp_path, sample, fake_model, restore_build_model
):
    from conftest import GOOD_CODE, fenced

    model = fake_model(first_answer=fenced(GOOD_CODE))
    result = _run(tmp_path, sample, model, AgentGuide())
    assert result == {"passed": True}
    assert model.calls == ["plain"]
