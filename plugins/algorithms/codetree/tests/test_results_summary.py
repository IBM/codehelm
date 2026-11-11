# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codetree.results_summary."""

import json
from pathlib import Path
import pytest
from codetree.results_summary import (
    STATS_FILE,
    SUMMARY_FILE,
    summarize_results,
)


def _make_sample(results_dir: Path, n: int, passed: bool, rounds=None):
    d = results_dir / f"sample_{n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "validator_result.json").write_text(json.dumps({"passed": passed}))
    if rounds is not None:
        (d / STATS_FILE).write_text(json.dumps({"rounds": rounds}))


def test_summarize_results_empty_dir(tmp_path):
    summary = summarize_results(tmp_path, verbose=False)
    assert summary["samples"] == 0
    assert summary["solved"] == 0
    assert summary["acc"] == 0.0
    assert summary["weak_acc"] == 0.0
    assert summary["searches"] == 0
    assert summary["passed_at_sample"] == []
    assert (tmp_path / SUMMARY_FILE).exists()


def test_summarize_results_ignores_non_sample_dirs(tmp_path):
    (tmp_path / "other_dir").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "stats").mkdir()
    _make_sample(tmp_path, 0, True)

    summary = summarize_results(tmp_path, verbose=False)
    assert summary["samples"] == 1
    assert summary["solved"] == 1
    assert summary["acc"] == 1.0


def test_summarize_results_skips_samples_without_validator_result(tmp_path):
    d = tmp_path / "sample_0"
    d.mkdir(parents=True)
    # No validator_result.json

    _make_sample(tmp_path, 1, True)
    summary = summarize_results(tmp_path, verbose=False)
    assert summary["samples"] == 1


def test_summarize_results_comprehensive_metrics(tmp_path, capsys):
    # Sample 0: solved zero-shot (no codetree_stats.json)
    _make_sample(tmp_path, 0, True)

    # Sample 1: solved in 2 tries
    _make_sample(
        tmp_path,
        1,
        True,
        [
            {
                "num_try": 2,
                "exhausted": False,
                "passed_visible": True,
                "weak_solved": True,
                "end_reason": "accepted",
            }
        ],
    )

    # Sample 2: failed visible tests, exhausted search
    _make_sample(
        tmp_path,
        2,
        False,
        [
            {
                "num_try": 20,
                "exhausted": True,
                "passed_visible": False,
                "weak_solved": False,
                "end_reason": "max_iters",
            }
        ],
    )

    # Sample 3: passed visible tests and reviewer, but failed hidden test
    _make_sample(
        tmp_path,
        3,
        False,
        [
            {
                "num_try": 4,
                "exhausted": False,
                "passed_visible": True,
                "weak_solved": True,
                "end_reason": "accepted",
            }
        ],
    )

    # Sample 4: solved on retry round
    _make_sample(
        tmp_path,
        4,
        True,
        [
            {
                "num_try": 20,
                "exhausted": True,
                "passed_visible": False,
                "weak_solved": False,
                "end_reason": "max_iters",
            },
            {
                "num_try": 3,
                "exhausted": False,
                "passed_visible": True,
                "weak_solved": True,
                "end_reason": "accepted",
            },
        ],
    )

    summary = summarize_results(tmp_path, verbose=True)
    captured = capsys.readouterr().out

    assert summary["samples"] == 5
    assert summary["solved"] == 3  # samples 0, 1, 4
    assert summary["acc"] == 0.6
    assert summary["weak_acc"] == 0.8  # samples 0, 1, 3, 4
    assert summary["solved_before_codetree"] == 1
    assert summary["passed_at_sample"] == [2, 3]
    assert summary["passed_visible_not_solved"] == 1
    assert summary["searches"] == 5
    assert summary["exhausted_searches"] == 2
    assert summary["searches_without_answer"] == 2
    assert summary["end_reasons"] == {"accepted": 3, "max_iters": 2}

    assert "completed 5/5: acc = 0.6, weak_acc = 0.8" in captured
    assert (tmp_path / SUMMARY_FILE).exists()
