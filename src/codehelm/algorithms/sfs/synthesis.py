#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#
# Derived from codespace-optimization/sfs (Apache-2.0); modified for CodeHelm.
#

"""MCTS Synthesis Solver.

Iterative Monte Carlo Tree Search solver with Scout (idea exploration) and
Strategist (code improvement) phases, guided by test feedback.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .executor import PyExecutor
from .generator import GenericChatGenerator
from .model_adapter import ModelAdapter
from .search import Graph
from .types import CodingProblem, CodingProblemAndSolution

logger = logging.getLogger(__name__)


def _write_visible_tests(results_path: str, visible_tests: tuple[str, ...]) -> None:
    """Write visible_tests to ``<results_path>/visible_tests.json`` for inspection."""
    os.makedirs(results_path, exist_ok=True)
    out = [{"index": i, "test": t} for i, t in enumerate(visible_tests)]
    path = os.path.join(results_path, "visible_tests.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _write_graph(
    results_path: str,
    graph: Graph,
    best_solution: CodingProblemAndSolution,
) -> None:
    """Serialize the MCTS search tree to ``<results_path>/graph.json``.

    Produces a node/edge list suitable for rendering as a tree visualisation.
    Each node record contains: id, depth, score, visits, is_best, label
    (first 60 chars of the solution code or "root"/"seed N").
    """
    from collections import deque

    nodes: list[dict] = []
    edges: list[dict] = []

    # BFS from root
    queue: deque = deque()
    queue.append((graph.root, None, "root"))
    node_id = 0
    node_map: dict[int, int] = {id(graph.root): node_id}

    seed_counter = 0

    while queue:
        node, parent_id, action_label = queue.popleft()
        nid = node_map[id(node)]
        sol = node.solution

        if sol is None:
            label = "root"
            code_snippet = ""
        else:
            code = sol.solution or ""
            code_snippet = code[:120].replace("\n", " ")
            if node.depth == 1:
                seed_counter += 1
                label = f"seed {seed_counter}"
            else:
                label = f"d{node.depth}"

        nodes.append(
            {
                "id": nid,
                "depth": node.depth,
                "score": round(node.score, 4),
                "value": round(node.value, 4),
                "visits": node.visits,
                "is_best": sol is not None and sol is best_solution,
                "label": label,
                "action": (
                    action_label[:80]
                    if isinstance(action_label, str)
                    else str(action_label)[:80]
                ),
                "code_snippet": code_snippet,
            }
        )

        if parent_id is not None:
            edges.append({"source": parent_id, "target": nid})

        for action, child in node.action_to_node.items():
            child_id = node_id + 1
            node_id += 1
            node_map[id(child)] = child_id
            queue.append((child, nid, str(action)[:80]))

    out = {"nodes": nodes, "edges": edges}
    os.makedirs(results_path, exist_ok=True)
    path = os.path.join(results_path, "graph.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


class SynthesisSolver:
    """MCTS-based synthesis and repair solver."""

    def __init__(
        self,
        generator: GenericChatGenerator,
        model: ModelAdapter,
        executor: PyExecutor,
        max_iters: int = 7,
        num_seeds: int = 3,
        temperature: float = 0.6,
        use_problem_tests: bool = True,
        number_of_tests: int = 3,
        testgen_method: str = "simple",
        codegen_method: str = "outcome",
        num_ideas_per_strategy: int = 3,
        is_puct: bool = False,
        return_any_solved: bool = False,
        preamble_mode: str = "none",
        **kwargs: Any,
    ) -> None:
        self.generator = generator
        self.model = model
        self.executor = executor
        self.max_iters = max_iters
        self.num_seeds = num_seeds
        self.temperature = temperature
        self.use_problem_tests = use_problem_tests
        self.number_of_tests = number_of_tests
        self.testgen_method = testgen_method
        self.codegen_method = codegen_method
        self.num_ideas_per_strategy = num_ideas_per_strategy
        self.is_puct = is_puct
        self.return_any_solved = return_any_solved
        self.preamble_mode = preamble_mode
        self.do_fill_with_tests = not use_problem_tests

    def apply_problem_tests(self, problem: CodingProblem) -> None:
        """Use problem's built-in test suite as the single visible test."""
        if problem.test:
            problem.visible_tests = (problem.test,)
        self.do_fill_with_tests = False

    def fill_with_tests(
        self, problem: CodingProblem, num_tests: int, max_tries: int = 20
    ) -> None:
        """Populate visible_tests using LLM test generation until num_tests are obtained."""
        if self.do_fill_with_tests:
            counter = 0
            visible = list(problem.visible_tests)
            while len(visible) < num_tests:
                generated = self.generator.generate_internal_tests(
                    func_sig=problem.prompt,
                    model=self.model,
                    max_num_tests=num_tests,
                    entry_point=problem.entry_point,
                    temperature=self.temperature,
                    method=self.testgen_method,
                )
                visible.extend(generated)
                counter += 1
                if counter > max_tries:
                    break
            problem.visible_tests = tuple(visible[:num_tests])

    def generate_seed_functions(
        self, num_seeds: int, problem: CodingProblem
    ) -> list[CodingProblemAndSolution]:
        """Generate initial candidate solutions."""
        seed_functions: list[CodingProblemAndSolution] = []
        for i in range(num_seeds):
            logger.info(
                "[SFS] Generating seed candidate %d/%d for task '%s'...",
                i + 1,
                num_seeds,
                problem.task_id,
            )
            cur_solution = CodingProblemAndSolution.init_from_coding_problem(problem)
            counter = 0
            while cur_solution.solution is None:
                strategies = self.generator.generate_func_impl(
                    func_sig=problem.prompt,
                    model=self.model,
                    strategy="simple",
                    temperature=self.temperature,
                    entry_point=problem.entry_point,
                )
                if strategies:
                    cur_solution.solution = strategies[0].code
                    cur_solution.extra_kwargs["logprob"] = strategies[0].logprob
                counter += 1
                if counter > 10:
                    break

            cur_solution.extra_kwargs["int_output_tokens"] = self.model.output_tokens
            cur_solution.extra_kwargs["int_total_tokens"] = self.model.total_tokens
            cur_solution.extra_kwargs["int_total_requests"] = self.model.total_requests

            self.executor.execute(cur_solution, will_modify=True)
            vis_len = len(cur_solution.visible_tests)
            vis_pass = sum(cur_solution.test_results)
            logger.info(
                "[SFS] Seed %d/%d evaluated: passing=%s, score=%d/%d",
                i + 1,
                num_seeds,
                cur_solution.is_passing,
                vis_pass,
                vis_len,
            )
            seed_functions.append(cur_solution)

            if cur_solution.test_results and all(cur_solution.test_results):
                logger.info(
                    "[SFS] Seed %d passed all visible tests! Skipping remaining seeds.",
                    i + 1,
                )
                break

        return seed_functions

    def is_any_solved(
        self, solutions: list[CodingProblemAndSolution]
    ) -> tuple[CodingProblemAndSolution | None, bool, int | None]:
        """Check if any candidate solution in the search passed the evaluation."""
        for i, solution in enumerate(solutions):
            if self.executor.evaluate(solution, will_modify=False):
                return solution, True, i
        return None, False, None

    def generate_updated_summary(
        self,
        old_summary: str,
        old_strategy: CodingProblemAndSolution,
        idea: str,
        new_strategy: CodingProblemAndSolution,
    ) -> str:
        """Update running insight summary."""
        old_impl = old_strategy.solution or ""
        new_impl = new_strategy.solution or ""
        old_tests_count = len(old_strategy.test_results) or 1
        score_diff = (
            sum(new_strategy.test_results) - sum(old_strategy.test_results)
        ) / old_tests_count

        return self.generator.generate_updated_summary(
            summary=old_summary,
            prompt=old_strategy.prompt,
            prev_impl=old_impl,
            prev_feedback=old_strategy.feedback_string,
            improvement_idea=idea,
            new_impl=new_impl,
            model=self.model,
            new_feedback=new_strategy.feedback_string,
            temperature=self.temperature,
            num_comps=1,
            is_few_shot=True,
            score_difference=score_diff,
        )

    def generate_improvement_ideas_with_summary(
        self,
        strategy: CodingProblemAndSolution,
        num_ideas: int,
        summary: str,
    ) -> list[str]:
        """Scout step: generate improvement ideas."""
        cur_impl = strategy.solution or ""
        return self.generator.generate_ideas_with_summary(
            func_sig=strategy.prompt,
            prev_func_impl=cur_impl,
            num_ideas=num_ideas,
            model=self.model,
            feedback=strategy.feedback_string,
            temperature=self.temperature,
            summary=summary,
        )

    def generate_strategy(
        self,
        old_strategy: CodingProblemAndSolution,
        improvement_idea: str,
    ) -> CodingProblemAndSolution:
        """Strategist step: implement code modification based on improvement idea."""
        cur_impl = old_strategy.solution or ""
        strat_return = self.generator.strategist_generate_strategy(
            func_sig=old_strategy.prompt,
            prev_func_impl=cur_impl,
            model=self.model,
            strategy=self.codegen_method,
            feedback=old_strategy.feedback_string,
            self_reflection="No reflections.",
            idea_to_implement=improvement_idea,
            num_comps=1,
            temperature=self.temperature,
        )

        new_strategy = CodingProblemAndSolution.init_from_coding_problem(old_strategy)
        new_strategy.solution = strat_return.code
        new_strategy.generation = old_strategy.generation + 1
        new_strategy.extra_kwargs["logprob"] = strat_return.logprob
        return new_strategy

    def solve(
        self, problem: CodingProblem, results_path: str = ""
    ) -> CodingProblemAndSolution:
        """Run MCTS synthesis search loop."""
        logger.info(
            "[SFS] Starting MCTS Synthesis for task '%s' (max_iters=%d, num_seeds=%d, use_problem_tests=%s)",
            problem.task_id,
            self.max_iters,
            self.num_seeds,
            self.use_problem_tests,
        )
        self.model.reset_stats()
        all_solutions: list[CodingProblemAndSolution] = []

        if self.use_problem_tests:
            self.apply_problem_tests(problem)
        else:
            self.fill_with_tests(problem, self.number_of_tests)

        if results_path:
            _write_visible_tests(results_path, problem.visible_tests)
            if not problem.visible_tests:
                logger.warning(
                    "[SFS] No visible tests available for task '%s' — scoring will be blind.",
                    problem.task_id,
                )

        seed_functions = self.generate_seed_functions(
            num_seeds=self.num_seeds, problem=problem
        )
        all_solutions.extend(seed_functions)

        if not seed_functions:
            logger.warning(
                "[SFS] No seed functions generated for task '%s'", problem.task_id
            )
            fallback = CodingProblemAndSolution.init_from_coding_problem(problem)
            fallback.solution = ""
            return fallback

        graph = Graph.init_from_seed_solutions(seed_functions)
        graph.is_puct = self.is_puct
        general_insights = "None"

        for iteration in range(self.max_iters):
            cur_node = graph.simulate_trajectory(
                stop_condition="has_unvisited_actions_or_no_better_child"
            )
            cur_solution = cur_node.solution
            if cur_solution is None:
                break

            if cur_solution.test_results and all(cur_solution.test_results):
                logger.info(
                    "[SFS] Node at depth %d already passes all visible tests. Ending search early.",
                    cur_node.depth,
                )
                break

            logger.info(
                "[SFS] --- MCTS Iteration %d/%d (Node depth=%d, visits=%d, score=%.2f) ---",
                iteration + 1,
                self.max_iters,
                cur_node.depth,
                cur_node.visits,
                cur_node.get_score(),
            )

            counter = 0
            while not cur_node.has_unvisited_actions():
                if counter > 5:
                    logger.warning(
                        "[SFS] Could not generate new unvisited actions after 5 attempts"
                    )
                    break
                directions = self.generate_improvement_ideas_with_summary(
                    cur_solution, self.num_ideas_per_strategy, general_insights
                )
                logger.info(
                    "[SFS] Scout generated %d improvement ideas", len(directions)
                )
                cur_node.actions.update(directions)
                counter += 1

            unvisited = cur_node.actions - cur_node.action_to_node.keys()
            if not unvisited:
                logger.info("[SFS] No unvisited actions left for current node")
                break

            action = next(iter(unvisited))
            action_str = str(action)
            logger.info(
                "[SFS] Strategist implementing idea: '%s...'",
                action_str[:70].replace("\n", " "),
            )
            new_solution = self.generate_strategy(cur_solution, action_str)
            self.executor.execute(new_solution)

            new_solution.extra_kwargs["int_output_tokens"] = self.model.output_tokens
            new_solution.extra_kwargs["int_total_tokens"] = self.model.total_tokens
            new_solution.extra_kwargs["int_total_requests"] = self.model.total_requests
            all_solutions.append(new_solution)

            vis_count = len(new_solution.visible_tests) or 1
            score = sum(new_solution.test_results) / vis_count
            logger.info(
                "[SFS] Candidate evaluated: passing=%s, score=%d/%d (%.2f)",
                new_solution.is_passing,
                sum(new_solution.test_results),
                vis_count,
                score,
            )

            new_node = graph.add_child_solution(
                cur_solution,
                action,
                new_solution,
                score,
                logp=new_solution.extra_kwargs.get("logprob", 1.0),
            )
            new_node.backpropogate()

            general_insights = self.generate_updated_summary(
                general_insights, cur_solution, action_str, new_solution
            )

        best_solution = graph.get_best_score_solution()
        if best_solution is None:
            best_solution = seed_functions[0]

        # Optional fallback: if best MCTS node fails hidden test but another solution passed
        if self.return_any_solved:
            solved_solution, any_solved, _ = self.is_any_solved(all_solutions)
            if (
                any_solved
                and solved_solution is not None
                and not self.executor.evaluate(best_solution, will_modify=False)
            ):
                best_solution = solved_solution

        best_solution.extra_kwargs["total_requests"] = self.model.total_requests
        best_solution.extra_kwargs["total_tokens"] = self.model.total_tokens
        best_solution.extra_kwargs["output_tokens"] = self.model.output_tokens

        logger.info(
            "[SFS] Finished search for task '%s'. Best solution score: %.2f (total candidates explored: %d, requests: %d, tokens: %d)",
            problem.task_id,
            best_solution.get_prop_test_passed(),
            len(all_solutions),
            self.model.total_requests,
            self.model.total_tokens,
        )

        if results_path:
            _write_graph(results_path, graph, best_solution)

        return best_solution
