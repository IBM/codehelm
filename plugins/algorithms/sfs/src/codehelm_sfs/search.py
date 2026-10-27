#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#
# Derived from codespace-optimization/sfs (Apache-2.0); modified for CodeHelm.
#

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

from .types import CodingProblemAndSolution

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Node in the MCTS search tree."""

    solution: CodingProblemAndSolution | None
    parent: Node | None
    score: float
    depth: int
    visits: int
    action_to_node: dict[Hashable, Node]
    action_to_logp: defaultdict[Hashable, float]
    actions: set[Hashable]
    value: float

    def __init__(
        self,
        solution: CodingProblemAndSolution | None,
        parent: Node | None = None,
        score: float = 0.0,
        depth: int = 0,
        value: float = 0.0,
    ):
        self.solution = solution
        self.parent = parent
        self.score = score
        self.depth = depth
        self.value = value
        self.action_to_node = {}
        self.action_to_logp: defaultdict[Hashable, float] = defaultdict(lambda: 1.0)
        self.actions: set[Hashable] = set()
        self.visits = 1

    def __str__(self) -> str:
        return f"Node(strategy={self.solution}, score={self.score}, depth={self.depth})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.solution)

    def add_child_solution(
        self,
        action: Hashable,
        solution: CodingProblemAndSolution,
        score: float,
        logp: float = 1.0,
    ) -> Node:
        logger.debug(
            "Adding child solution: %s with score: %s, action: %s, logp: %s to parent: %s",
            hash(solution),
            score,
            action,
            logp,
            hash(self),
        )
        child = Node(
            solution, parent=self, score=score, depth=self.depth + 1, value=score
        )
        self.action_to_node[action] = child
        self.actions.add(action)
        self.action_to_logp[action] = logp
        return child

    def get_best_child_uct(self, is_puct: bool = False) -> Node | None:
        """Returns the best child according to UCT values of the actions. None if no children."""
        best_child = None
        best_uct = float("-inf")
        for action, child in self.action_to_node.items():
            uct = self.get_puct(action) if is_puct else self.get_uct(action)
            if uct > best_uct:
                best_uct = uct
                best_child = child
        return best_child

    def has_unvisited_actions(self) -> bool:
        """Returns True if there are unvisited actions."""
        return len(self.actions) > len(self.action_to_node)

    def get_random_unvisited_action(self) -> Hashable | None:
        """Returns a random unvisited action. None if no unvisited actions."""
        unvisited_actions = self.actions - self.action_to_node.keys()
        if len(unvisited_actions) == 0:
            return None
        return next(iter(unvisited_actions))

    def has_no_children(self) -> bool:
        """Returns True if the node has no children."""
        return len(self.action_to_node) == 0

    def has_no_better_child(self) -> bool:
        """Returns whether any child has a better score than the current node."""
        for child in self.action_to_node.values():
            if child.get_score() > self.get_score():
                return False
        return True

    def get_uct(
        self,
        action: Hashable,
        c: float = math.sqrt(2),
    ) -> float:
        """Computes the UCT value for the given action."""
        if action not in self.action_to_node:
            logger.debug("Action: %s not in action_to_node", action)
            return float("inf")
        child = self.action_to_node[action]
        return child.value + c * math.sqrt(math.log(self.visits) / child.visits)

    def get_puct(self, action: Hashable, c: float = 2.0, cbase: float = 10.0) -> float:
        """Computes the PUCT value for the given action."""
        if action not in self.action_to_node:
            logger.debug("Action: %s not in action_to_node", action)
            return float("inf")
        child = self.action_to_node[action]
        beta = math.log((1 + self.visits + cbase) / cbase) + c
        out = child.value + beta * self.get_pvalue(action) * math.sqrt(
            math.log(self.visits) / (1 + child.visits)
        )
        logger.debug("PUCT value for action: %s is %s", action, out)
        return out

    def get_pvalue(self, action: Hashable) -> float:
        """Converts logprob to pvalue with softmax normalization."""
        max_logp = max(self.action_to_logp.values())
        denom = sum(math.exp(logp - max_logp) for logp in self.action_to_logp.values())
        if denom == 0:
            return 1.0 / len(self.action_to_logp)
        out = math.exp(self.action_to_logp[action] - max_logp) / denom
        logger.debug("Pvalue for action: %s is %s", action, out)
        return out

    def backpropogate(self, alpha: float = 1.0) -> None:
        """Backpropogates the value up the tree."""
        if self.parent is not None:
            self.parent.value += alpha * max(0.0, self.value - self.parent.value)
            self.parent.backpropogate(alpha=alpha)

    def get_score(self) -> float:
        """Returns the score of the node."""
        return self.score


@dataclass
class Graph:
    """MCTS Search graph."""

    root: Node
    nodes: dict[CodingProblemAndSolution, Node] = field(default_factory=dict)
    is_puct: bool = False

    def __init__(self, root: Node, is_puct: bool = False):
        self.root = root
        self.nodes = {}
        self.is_puct = is_puct

    @staticmethod
    def init_from_seed_solutions(
        seed_solutions: list[CodingProblemAndSolution],
    ) -> Graph:
        """Initializes the graph from seed solutions."""
        root = Node(None, score=-1.0)
        graph = Graph(root)
        for seed_solution in seed_solutions:
            visible_count = len(seed_solution.visible_tests)
            score = (
                sum(seed_solution.test_results) / visible_count
                if visible_count > 0
                else 0.0
            )
            logp = seed_solution.extra_kwargs.get("logprob", 1.0)
            root.add_child_solution(
                hash(seed_solution),
                seed_solution,
                score,
                logp=logp,
            )
            logger.debug(
                "Added seed solution: %s with score: %s",
                hash(seed_solution),
                score,
            )
            graph.nodes[seed_solution] = root.action_to_node[hash(seed_solution)]
        return graph

    def add_child_solution(
        self,
        parent_solution: CodingProblemAndSolution,
        action: Hashable,
        solution: CodingProblemAndSolution,
        score: float,
        logp: float = 1.0,
    ) -> Node:
        """Adds a child solution to the parent solution."""
        parent_node = self.nodes[parent_solution]
        child_node = parent_node.add_child_solution(
            action=action, solution=solution, score=score, logp=logp
        )
        self.nodes[solution] = child_node
        return child_node

    def simulate_trajectory(self, stop_condition: str = "has_unvisited_action") -> Node:
        """Simulates a trajectory from the root node until stop_condition is met."""
        do_stop: Callable[[Node], bool]
        if stop_condition == "has_unvisited_action":
            do_stop = lambda node: node.has_unvisited_actions()
        elif stop_condition == "has_no_better_child":
            do_stop = lambda node: node.has_no_better_child()
        elif stop_condition == "has_unvisited_actions_or_no_better_child":
            do_stop = lambda node: (
                node.has_unvisited_actions() or node.has_no_better_child()
            )
        else:
            raise ValueError(f"Invalid stop_condition: {stop_condition}")

        cur_node = self.root
        cur_node.visits += 1

        logger.debug("Simulating trajectory")
        counter = 0
        while not cur_node.has_no_children() and not do_stop(cur_node):
            if counter >= 100:
                raise RuntimeError("MCTS Infinite loop trajectory")
            logger.debug("Current node: %s", hash(cur_node))
            next_node = cur_node.get_best_child_uct(is_puct=self.is_puct)
            if next_node is None:
                raise ValueError("No children")
            cur_node = next_node
            cur_node.visits += 1
            counter += 1
        logger.debug("Simulated trajectory ended at node: %s", hash(cur_node))
        return cur_node

    def get_best_score_solution(self) -> CodingProblemAndSolution | None:
        """Returns the best score solution. None if no solutions."""
        best_score = float("-inf")
        best_solution = None
        for solution, node in self.nodes.items():
            if node.get_score() > best_score:
                best_score = node.get_score()
                best_solution = solution
        return best_solution

    def get_all_solutions(self) -> list[CodingProblemAndSolution]:
        """Returns all solutions in the graph."""
        return list(self.nodes.keys())

    def get_all_best_solutions(self) -> list[CodingProblemAndSolution]:
        """Returns all solutions with the best score."""
        best_score = float("-inf")
        best_solutions = []
        for solution, node in self.nodes.items():
            if node.get_score() > best_score:
                best_score = node.get_score()
                best_solutions = [solution]
            elif node.get_score() == best_score:
                best_solutions.append(solution)
        return best_solutions
