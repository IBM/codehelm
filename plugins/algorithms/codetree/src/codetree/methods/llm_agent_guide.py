# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
# Derived from https://github.com/SalesforceAIResearch/CodeTree (Apache-2.0); modified for CodeHelm.
# Original License below.

# LICENSE HEADER MANAGED BY add-license-header
#
# /*
#  * Copyright (c) 2023, Salesforce, Inc.
#  * SPDX-License-Identifier: Apache-2
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  */
#
import json
import os
import sys

from codehelm.orchestration.base import FinalResponse
from codetree.executors import PyExecutor
from codetree.generators.py_generate import PyGenerator
from codetree.node import Node
from codetree.results_summary import STATS_FILE

sys.set_int_max_str_digits(100000)  # Increase the limit to 10000 digits

ADD_HINT = "To solve the problem, you can refer the hint given by an expert, and complete the details by analyzing it first.\nHint:"

def eval_node(prompt, node:Node, gen, model, max_depth=3, keep_failing_roots=False):
    """
    The evaluation shows not fully correct, decide whether to go on.

    Returns (go_on, value, analysis, reason); ``reason`` says which rule decided, for the search log.

    A node is dropped when it is no better than its parent. A root (first attempt) has no parent,
    so it is compared with an artificial parent value of 0. A root that failed every visible test and
    was scored 0 (typically code that crashes, e.g. a TypeError from a wrong API call) therefore
    ties with that 0 and is dropped at depth 1: it never gets reflections or a debugging attempt,
    and the search can end after a few tries with no nodes left. This is how the original CodeTree
    behaves (``keep_failing_roots=False``). With ``keep_failing_roots=True`` such a root is expanded
    anyway so the debugger gets a chance to fix it; only roots are exempt, deeper nodes that make no
    progress over their real parent are still dropped.
    """
    if node.parent is None: parent_value=0
    else: parent_value =  node.parent.value
    if node.depth >= max_depth: return False, node.value, "", "max_depth"
    agent_reward, analysis = gen.agent_eval(prompt, model, prev_func_impl=node.solution, task="tests",
                                  feedback=node.test_feedback.split("[additional review]:")[0].strip())
    node.value += float(agent_reward) / 15
    if node.value < parent_value: pruned_by = "value_below_parent"
    elif node.value == parent_value and node.depth > agent_reward: pruned_by = "value_equals_parent_low_reward"
    else: return True, node.value, analysis, "expanded"
    if keep_failing_roots and node.parent is None:
        return True, node.value, analysis, "kept_failing_root"
    return False, node.value, "", pruned_by

def step_verify(gen, exe, item, solution, model):
    """
    if pass all public test cases, run one agent review step
    """
    is_passing, feedback, reward = gen_test_eval(exe, solution, item["given_tests"], prev=item["prev"])
    if not is_passing:
        return False, feedback, reward
    else:
        reward = 1
        option, analysis = gen.agent_eval(item["prompt"], model, prev_func_impl=solution,
                                      task="stop", feedback=feedback, temperature=0)
        if option: return True, feedback, reward
        else:
            return False, f"{feedback}\n\n[additional review]:\n\n{analysis}", reward

def gen_test_eval(exe, solution, test_cases, prev=""):
    is_passing, feedback, _ = exe.execute(solution, test_cases, timeout=60, prev=prev)
    if is_passing: reward = 1
    else:
        reward = _.count(True)/len(_)
    return is_passing, feedback, reward

VISIBLE_TEST_MODES = ("dataset", "generated", "dataset+generated")
PRUNE_RULES = ("original", "keep_failing_roots")


class AgentGuide:
    def __init__(self, visible_tests: str = "dataset", max_generated_tests: int = 12,
                 prune_rule: str = "original") -> None:
        """
        visible_tests controls which tests drive the tree search (the hidden test is always the
        dataset test, applied by the validator):
          - "dataset":           the dataset test is also the visible test (oracle feedback).
          - "generated":         only tests produced by PyGenerator.internal_tests (no leakage).
          - "dataset+generated": both of the above.

        prune_rule controls what happens to a first attempt that fails every visible test and is
        scored 0 by the evaluator (see ``eval_node`` for the details):
          - "original":           it is dropped, so the debugger never runs (original CodeTree).
          - "keep_failing_roots": it is expanded with reflections so the debugger can fix it.
        """
        if visible_tests not in VISIBLE_TEST_MODES:
            raise ValueError(f"visible_tests must be one of {VISIBLE_TEST_MODES}, got {visible_tests!r}")
        if prune_rule not in PRUNE_RULES:
            raise ValueError(f"prune_rule must be one of {PRUNE_RULES}, got {prune_rule!r}")
        self.visible_tests = visible_tests
        self.max_generated_tests = max_generated_tests
        self.prune_rule = prune_rule
        self.requires_model = True
        self._model = None
        self.exe = PyExecutor()
        self._sample_dir = None
        self._rounds = []
        self._trace = []

    @property
    def model(self):
        return getattr(self, "base_model", None) or self._model

    @model.setter
    def model(self, val):
        self._model = val

    def setup(self, sample_dir, **kwargs):
        self._attempt = 0
        self._sample_dir = sample_dir
        self._rounds = []
        self.item = kwargs.get("item") or {}
        test = kwargs.get("test")
        entry_point = kwargs.get("entry_point")
        prompt = kwargs.get("prompt") or ""
        if test and entry_point:
            self.item.setdefault("test", test)
            self.item.setdefault("entry_point", entry_point)
            # Mirror ValidatorExec: code-stub prompts (imports + signature) are prepended to the code.
            if prompt.lstrip().startswith(("from ", "import ", "def ")):
                self.item.setdefault("prev", prompt)
            # Qiskit HumanEval defines check(candidate); QuantumKatas define test_<entry_point>().
            # The test must be *called* to run anything, so the runner line is appended.
            runner = f"check({entry_point})" if "def check(" in test else f"test_{entry_point}()"
            runnable_test = f"{test}\n{runner}"
            # No separate weaker benchmark exists for this data: it is the dataset test itself.
            self.item.setdefault("weaker_test", runnable_test)
            if not self.item.get("given_tests"):
                given_tests = []
                if "dataset" in self.visible_tests:
                    given_tests.append(runnable_test)
                if "generated" in self.visible_tests:
                    given_tests.extend(
                        PyGenerator().internal_tests(prompt, self.model, max_num_tests=self.max_generated_tests)
                    )
                self.item["given_tests"] = given_tests
        self.item.setdefault("prev", "")
        self.item.setdefault("given_tests", [])
        self.item["given_tests"] = [test for test in self.item["given_tests"] if 'assert False' not in test]
        if not self.item["given_tests"]:
            raise ValueError(
                f"AgentGuide (visible_tests={self.visible_tests!r}) requires at least one entry in "
                "item['given_tests']; none were provided (or all were filtered out as 'assert False')."
            )
    
    def repair(
            self,
            prompt: str,
            failure_info: str = "",
            verbose: bool = False,
            max_depth: int = 3,
            search_width: int = 10,
            max_iters: int = 20,
            **kwargs,
    ):
        print("max_depth", max_depth, "search_width", search_width)
        gen = PyGenerator()

        self.item["prompt"] = prompt
        # Thinker Agent Preparation
        hints = gen.strategy(prompt, self.model, num_strategy="multiple", task="strategy", temperature=0)
        if len(hints) > search_width: hints = hints[:search_width] # width cut
        stack = []
        is_passing = False
        self.num_try = 0
        self._trace = []  # one entry per node visited, so a short search can be explained afterwards
        for hint in reversed(hints):
            new_node = Node("", strategy=hint, depth=1)
            stack.append(new_node) # initial placeholders for new nodes

        # Tree Search Start
        found_solution = None
        candidate_solution = None
        while stack and self.num_try < max_iters and not is_passing:
            if len(stack) == 0: break
            this_node = stack.pop()
            if this_node.depth > max_depth: continue
            # Solver Agent
            if not this_node.solution:
                cur_solution = gen.func_impl(prompt + f"{ADD_HINT}{this_node.strategy}\n",
                                                    self.model, "simple", temperature=0)
                if not candidate_solution: candidate_solution = cur_solution

            # Debugger Agent
            else:
                cur_solution = gen.func_impl(
                    func_sig=prompt,
                    model=self.model,
                    strategy="reflexion",
                    prev_func_impl=this_node.solution,
                    feedback=this_node.test_feedback.split("[additional review]:")[0].strip(),
                    self_reflection=this_node.reflection,
                    temperature=0
                )
            self.num_try += 1

            # Execute and get Feedback
            is_passing, feedback, reward = step_verify(gen, self.exe, self.item, cur_solution, self.model)
            print("cur solution judge", is_passing)

            # Update node information as parent
            this_node.solution = cur_solution # update the solution to real solution
            this_node.test_feedback = feedback # With additional critic feedback
            this_node.value = reward

            if reward > 0.99:
                this_node.pass_visible = True
                if this_node.parent and this_node.parent.pass_visible and this_node.depth==max_depth:
                        is_passing = True
                this_node.value += 5/15 # reward for passing all visible
            if is_passing:
                found_solution = cur_solution
                self._trace.append({"depth": this_node.depth, "reward": reward, "go_on": False,
                                    "reason": "accepted", "children": 0})
                break

            """
            NB bug in original code snippet. 
            if is_passing:
                found_solution = cur_solution
                break

            elif reward <= 0.99: # didn't pass, need debugging
                candidate_solution = cur_solution
                go_on, values, analysis = eval_node(prompt=item["prompt"],node=this_node,gen=gen,model=model,max_depth=max_depth)
                this_node.value = values

            # Continue on this node
            # GZ: this is never reached. if is_passing is false, then reward is always under 0.99
            else: go_on = this_node.pass_visible

            fix below
            """


            # Decide whether to keep expanding this node.
            # go_on is always assigned here; we only reach this point when is_passing is False.
            if reward > 0.99:
                # Passed visible tests but full stopping criterion not met — still worth exploring children.
                go_on, reason = True, "passed_visible_rejected_by_review"
            else:
                # Didn't pass visible tests; ask the evaluator whether to continue.
                candidate_solution = cur_solution
                go_on, values, analysis, reason = eval_node(
                    prompt=prompt, node=this_node, gen=gen, model=self.model, max_depth=max_depth,
                    keep_failing_roots=self.prune_rule == "keep_failing_roots",
                )
                this_node.value = values

            n_children_before = len(this_node.children)
            if go_on:
                # Thinker Agent, init startegies for potential agents
                reflections = gen.strategy(prompt,
                                            self.model, task="reflection",
                                            num_strategy="one or multiple (if there is)",
                                            prev_func_impl=this_node.solution,
                                            feedback=this_node.test_feedback,
                                            temperature=0,
                                            given_strategy=this_node.strategy)
                if len(reflections) > search_width: reflections = reflections[:search_width]
                for reflection in reversed(reflections):
                    if not reflection: continue
                    new_node = Node(cur_solution, reflection=reflection, parent=this_node, strategy=this_node.strategy, depth=this_node.depth + 1) # init with previous code
                    new_node.test_feedback = this_node.test_feedback
                    this_node.children.append(new_node) # children in a reverse order
            stack.extend(this_node.children)
            self._trace.append({
                "depth": this_node.depth,
                "reward": reward,
                "go_on": go_on,
                "reason": reason,
                "children": len(this_node.children) - n_children_before,
            })
        # found_solution is only set when the search accepted an answer (visible tests + review);
        # otherwise fall back to the first candidate that was generated.
        passed_visible = found_solution is not None
        if found_solution is None: found_solution = candidate_solution
        if passed_visible: end_reason = "accepted"
        elif self.num_try >= max_iters: end_reason = "max_iters"
        else: end_reason = "no_nodes_left"  # every node was pruned or hit max_depth before max_iters
        self._record_round(found_solution, max_iters, passed_visible, end_reason)

        # The tree search already produced the final answer; hand it to the runner as such
        # (fenced so the validator's code extraction picks it up unambiguously).
        return FinalResponse(f"```python\n{found_solution or ''}\n```")

    def _record_round(self, solution, max_iters, passed_visible, end_reason):
        """Append this search's stats and rewrite ``codetree_stats.json`` (see results_summary.py).

        Whether the answer solves the hidden test is left to the validator; only what the search
        itself knows is recorded here. ``end_reason`` is one of ``accepted``, ``max_iters`` or
        ``no_nodes_left``; ``trace`` lists every node visited with the rule that decided its fate.
        """
        weak_solved = self.exe.evaluate(
            self.item.get("entry_point", ""), solution or "", self.item.get("weaker_test", ""),
            timeout=60, prev=self.item["prev"],
        )
        self._rounds.append({
            "num_try": self.num_try,
            "max_iters": max_iters,
            "exhausted": self.num_try >= max_iters,
            "passed_visible": passed_visible,
            "weak_solved": weak_solved,
            "end_reason": end_reason,
            "trace": self._trace,
        })
        os.makedirs(self._sample_dir, exist_ok=True)
        with open(os.path.join(self._sample_dir, STATS_FILE), "w") as f:
            json.dump({"rounds": self._rounds}, f, indent=4)
