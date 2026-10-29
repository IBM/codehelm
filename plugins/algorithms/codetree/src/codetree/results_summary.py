# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0

"""Aggregate CodeTree results over a finished run.

Two files per ``sample_<n>`` directory are joined:

* ``validator_result.json`` - written by the validator; says whether the final answer passed the
  hidden test (this is the accuracy, the same number the original CodeTree reports as ``acc``).
* ``codetree_stats.json`` - written by ``AgentGuide.repair``; one entry per tree search ("round")
  with ``num_try``, ``max_iters``, ``exhausted``, ``passed_visible`` and ``weak_solved``.
  A sample that passed on the first zero-shot attempt never reaches ``repair`` and has no such file.
"""

import json
from collections import Counter
from pathlib import Path

STATS_FILE = "codetree_stats.json"
SUMMARY_FILE = "codetree_summary.json"


def summarize_results(results_dir, verbose: bool = True) -> dict:
    """Print (and save as ``codetree_summary.json``) the metrics of the original CodeTree run.

    Metrics, using the last tree search of each sample when retries happened:
      acc / weak_acc             - fraction solved on the hidden test / on the weaker test.
      passed_at_sample           - ``num_try`` of the search that produced each solved answer.
      passed_visible_not_solved  - answer passed the visible tests (and the review) but failed the
                                   hidden test; shows how well the visible tests predict the hidden one.
      exhausted_searches         - searches that used all ``max_iters`` tries (counted per search).
      searches_without_answer    - searches that never found an answer the visible tests and the
                                   review accepted (counted per search). Unlike ``exhausted``, this
                                   also covers searches that ran out of nodes before ``max_iters``.
      end_reasons                - why searches ended: ``accepted``, ``max_iters`` or ``no_nodes_left``
                                   (searches recorded before this field existed are ``unknown``).
      solved_before_codetree     - solved by the first attempt, so no search was ever run.
    Samples solved before CodeTree ran count as weakly solved, as the weaker test cannot be stricter.
    """
    results_dir = Path(results_dir)
    sample_dirs = sorted(
        (d for d in results_dir.glob("sample_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[-1]),
    )

    samples = solved = weak_solved = solved_before_codetree = passed_visible_not_solved = 0
    searches = exhausted = without_answer = 0
    end_reasons = Counter()
    passed_at_sample = []
    for sample_dir in sample_dirs:
        validator_file = sample_dir / "validator_result.json"
        if not validator_file.exists():
            continue
        is_solved = json.loads(validator_file.read_text())["passed"] is True
        stats_file = sample_dir / STATS_FILE
        rounds = json.loads(stats_file.read_text())["rounds"] if stats_file.exists() else []
        last = rounds[-1] if rounds else None

        samples += 1
        solved += is_solved
        searches += len(rounds)
        exhausted += sum(r["exhausted"] for r in rounds)
        without_answer += sum(not r["passed_visible"] for r in rounds)
        end_reasons.update(r.get("end_reason", "unknown") for r in rounds)
        weak_solved += last["weak_solved"] if last else is_solved
        if last is None:
            solved_before_codetree += is_solved
        elif is_solved:
            passed_at_sample.append(last["num_try"])
        elif last["passed_visible"]:
            passed_visible_not_solved += 1

    summary = {
        "samples": samples,
        "solved": solved,
        "acc": round(solved / samples, 3) if samples else 0.0,
        "weak_acc": round(weak_solved / samples, 3) if samples else 0.0,
        "solved_before_codetree": solved_before_codetree,
        "passed_visible_not_solved": passed_visible_not_solved,
        "searches": searches,
        "exhausted_searches": exhausted,
        "searches_without_answer": without_answer,
        "end_reasons": dict(end_reasons),
        "passed_at_sample": passed_at_sample,
    }
    (results_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=4))

    if verbose:
        print("_______________________________")
        print(passed_at_sample)
        print(sorted(passed_at_sample))
        print(len(passed_at_sample))
        print(Counter(passed_at_sample))
        print("Solved before CodeTree ran:", solved_before_codetree)
        print("Passed visible tests but not solved:", passed_visible_not_solved)
        print(f"Searches that used all their tries: {exhausted}/{searches}")
        print(f"Searches that found no accepted answer: {without_answer}/{searches}")
        print("Why searches ended:", dict(end_reasons))
        print(f"completed {samples}/{samples}: acc = {summary['acc']}, weak_acc = {summary['weak_acc']}")
    return summary
