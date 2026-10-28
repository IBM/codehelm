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
import os
import sys

from codehelm.orchestration.base import FinalResponse
from utils import make_printv, write_jsonl
from executors import PyExecutor
from generators.py_generate import PyGenerator
from typing import List

from collections import Counter
from node import Node

sys.set_int_max_str_digits(100000)  # Increase the limit to 10000 digits

ADD_HINT = "To solve the problem, you can refer the hint given by an expert, and complete the details by analyzing it first.\nHint:"

def eval_node(prompt, node:Node, gen, model, max_depth=3):
    """
    The evaluation shows not fully correct, decide whether to go on.
    """
    if node.parent is None: parent_value=0
    else: parent_value =  node.parent.value
    if node.depth >= max_depth: return False, node.value, ""
    agent_reward, analysis = gen.agent_eval(prompt, model, prev_func_impl=node.solution, task="tests",
                                  feedback=node.test_feedback.split("[additional review]:")[0].strip())
    node.value += float(agent_reward) / 15
    if node.value < parent_value: return False, node.value, ""
    elif node.value == parent_value and node.depth > agent_reward: return False, node.value, ""
    return True, node.value, analysis

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


class AgentGuide:
    def __init__(self, visible_tests: str = "dataset", max_generated_tests: int = 12) -> None:
        """
        visible_tests controls which tests drive the tree search (the hidden test is always the
        dataset test, applied by the validator):
          - "dataset":           the dataset test is also the visible test (oracle feedback).
          - "generated":         only tests produced by PyGenerator.internal_tests (no leakage).
          - "dataset+generated": both of the above.
        """
        if visible_tests not in VISIBLE_TEST_MODES:
            raise ValueError(f"visible_tests must be one of {VISIBLE_TEST_MODES}, got {visible_tests!r}")
        self.visible_tests = visible_tests
        self.max_generated_tests = max_generated_tests
        self.requires_model = True
        self._model = None
        self.num_success = 0
        self.weak_success = 0
        self.sad_case, self.debug_thoroughed_case = 0, 0
        self.exe = PyExecutor()
        self.passed_at_sample = []

    @property
    def model(self):
        return getattr(self, "base_model", None) or self._model

    @model.setter
    def model(self, val):
        self._model = val

    def setup(self, sample_dir, **kwargs):
        self._attempt = 0
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
            if not self.item.get("given_tests"):
                given_tests = []
                if "dataset" in self.visible_tests:
                    # Qiskit HumanEval defines check(candidate); QuantumKatas define test_<entry_point>().
                    runner = f"check({entry_point})" if "def check(" in test else f"test_{entry_point}()"
                    given_tests.append(f"{test}\n{runner}")
                if "generated" in self.visible_tests:
                    given_tests.extend(
                        PyGenerator().internal_tests(prompt, self.model, max_num_tests=self.max_generated_tests)
                    )
                self.item["given_tests"] = given_tests
        self.item.setdefault("prev", "")
        self.item.setdefault("given_tests", [])
        self.item.setdefault("weaker_test", self.item.get("test", ""))
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
        is_passing, is_solved, is_weaker_solved = False, False, False
        self.num_try = 0
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
                go_on = True
            else:
                # Didn't pass visible tests; ask the evaluator whether to continue.
                candidate_solution = cur_solution
                go_on, values, analysis = eval_node(prompt=prompt, node=this_node, gen=gen, model=self.model, max_depth=max_depth)
                this_node.value = values

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
        if self.num_try >= max_iters: self.debug_thoroughed_case += 1
        # Verify that values are actually fair for all nodes.

        if found_solution is None: found_solution = candidate_solution

        # The tree search already produced the final answer; hand it to the runner as such
        # (fenced so the validator's code extraction picks it up unambiguously).
        return FinalResponse(f"```python\n{found_solution or ''}\n```")

    def score(self, found_solution, idx, item, log_dir, log_path):
        print_v = make_printv(True)

        is_solved = self.exe.evaluate(
            item["entry_point"], found_solution, item["test"], timeout=10, prev=item["prev"])  # early exit
        if "weaker_test" in item.keys():
            is_weaker_solved = self.exe.evaluate(
                item["entry_point"], found_solution, item["weaker_test"], timeout=10, prev=item["prev"])
        if is_solved:
            self.num_success += int(is_solved)
            self.passed_at_sample.append(self.num_try)
        else:
            self.sad_case += 1
            print("Sad, Pass but not solve")

        if is_weaker_solved:
            self.weak_success += int(is_weaker_solved)
        item["acc"] = round(self.num_success / (idx + 1), 3)
        item["weak_acc"] = round(self.weak_success / (idx + 1), 3)

        # Per-sample artefacts
        if log_dir is not None:
            sample_id = str(item.get("task_id", idx)).replace("/", "_")
            sample_dir = os.path.join(log_dir, f"sample_{sample_id}")
            os.makedirs(sample_dir, exist_ok=True)
            # 1. Final generated code
            with open(os.path.join(sample_dir, "solution.py"), "w") as f:
                f.write(found_solution or "")
            # 2. Test output (feedback from the last step_verify call)
            with open(os.path.join(sample_dir, "test_output.txt"), "w") as f:
                solved_line = f"is_solved: {is_solved}  |  is_weaker_solved: {is_weaker_solved}\n\n"
                f.write(solved_line + (this_node.test_feedback if 'this_node' in dir() else ""))

        write_jsonl(log_path, [item], append=True)
        print_v(
            f'completed {idx + 1}/{1}: acc = {round(self.num_success / (idx + 1), 3)}, weak_acc={item["weak_acc"]}, pass no solve: {self.sad_case}, exhaust: {self.debug_thoroughed_case}')
    
