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

from .message_utils import Message, messages_to_str
import random
from typing import List, Callable
PY_STRATEGY = ("You are an AI assistant that provides strategy for Python programmers to code. You will be given a function signature and its docstring by the user. "
             "Your goal is to think of {py_strategy_k} "
             "strategies in English(Not Code) on how to approach this problem and solve it. Describe each strategy with a FEW sentences in a SINGLE Line. List and Number your strategies line by line using \"1. \"; \"2. \"; \"3. \" and so on.")
Prompt_flexible = "The number of alternatives(either one or multiple) should be determined given this specific case."
PY_REFELCTION = ("You are an AI assistant who can reflect on problem-solving solution program. You will be given a task, an incorrect function implementation and feedbacks from executing the code. Your goal is to describe the existing issue(s) and suggest methods on how to improve the code. Rules:\n"
                 "1. From the algorithm and implementation level, there could be multiple methods to fix the error, you should provide {py_strategy_k} alternative reflections using various strategies. If the bug is clouded and ambigious, you can use alternatives as different interpretations, too.\n"
                 "2. Each reflection should briefly describe the issues and bugs, what kind of improvement is needed, then describe how to implement the correction. You are allowed to restate the bug for each reflection if needed. Each reflection should start be complete and self-contained. In other words, if there are more than one bugs, they should be presented in one reflection rather than separately.\n"
                 "3. Answer format: List and number your alternatives line by line, starting with \"1. \", \"2. \" and so on. Each reflection alternative is in a single line within a few sentences.\n")


system_stop_simplified = """The user will provide a programming task along with a solution that passes all visible test cases. Your task is to further review the solution before it is judged against hidden test cases. Determine whether the solution is robust and general enough to pass unseen, valid test cases. Guideline:
    1. Generalization Check: Verify that the solution uses general methods, avoiding hardcoding specific values or cases unless explicitly required. Confirm that the approach logically extends to unseen cases without special assumptions.
    2. Boundary Check: Ensure all boundaries are correctly handled, including list indexing, loop start and end points, if-else conditions, and recursion exits. Look for potential off-by-one errors or boundary misses that could cause functional errors.
	3. Edge Case Check: Confirm that the solution correctly handles valid edge/corner cases, such as zero, negative, empty, boundary values, or other special problem-specific situations. Note: All unseen test cases are guaranteed to follow stated data types, formats, conditions, and other constraints in the problem, no need to handle unallowed inputs. Do NOT apply redundant handling for cases that the current solution inherently manages, such as empty lists in sorting algorithms (`sorted([]) → []`), unless they explicitly fail (e.g., `max([]) → error`).
	4. Major Efficiency Check: Check if the solution is within polynomial time/space complexity, if NOT, fail this check.

**Response Format**:
Firstly, within several sentences, follow the guideline and briefly analyze.
On a new line, respond with “True” if the solution is ACCEPTABLE as-is, or “False” if NECESSARY modifications are required to handle unseen valid test cases.

The following is one example of how to review:
<EXAMPLE 1>:
```python
def find_first_unique(nums: list[int]) -> int:
    \"\"\"
    Find the first unique integer in a list of integers.
    Args: nums (list[int]): A list of integers to search through.
    Returns: int: The first unique integer in the list, or -1 if no unique integer is found.
    Examples:
        >>> find_first_unique([4, 5, 1, 2, 0, 4])  ==>  5
        >>> find_first_unique([7, 3]) ==> 7
    \"\"\"
    for i, num in enumerate(nums):
        if num not in nums[i:]: return num
    return -1
```
<EXAMPLE 1 Review>:
1. Generalization Check: `num not in nums[i:]` won’t handle cases where the number appears previous positions, `find_first_unique([7, 7]) ==> 7` instead of -1. Other checks are omitted for now since the solution logic is wrong.
False
"""

def generic_generate_func_impl(
    func_sig: str,
    model,
    strategy: str,
    prev_func_impl,
    feedback,
    self_reflection,
    temperature,
    reflection_chat_instruction: str,
    simple_chat_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str]
) -> str:
    if strategy not in ("reflexion", "simple"):
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
        raise ValueError(
            "Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    if strategy == "reflexion":
        messages = [
            Message(
                role="system",
                content=f"{reflection_chat_instruction}\n{code_block_instruction}",
            ),
            Message(
                role="user",
                content=f"Here's the challenge for you:\n{func_sig}\n[implement]:\n",
            ),
            Message(
                role="assistant",
                content=add_code_block(prev_func_impl),
            ),
            Message(
                role="user",
                content=f"[unit test results from previous implement]:\n{feedback}\n\n[reflection on previous implement]:\n",
            ),
            Message(
                role="assistant",
                content=self_reflection + "\n",
            ),
            Message(
                role="user",
                content="[improved implement]:\n",
            ),
        ]
        print(messages_to_str(messages))
    else:  # simple
        messages = [
            Message(
                role="system",
                content=f"{simple_chat_instruction}\n{code_block_instruction}",
            ),
            Message(
                role="user",
                content=func_sig,
            ),
        ]
    func_bodies = model.generate(prompt=messages, temperature=temperature, max_tokens=2048*4)
    assert isinstance(func_bodies, str)
    func_body_str = parse_code_block(func_bodies)
    print_generated_func_body(func_body_str, raw=func_bodies)
    # If parsing extracted nothing, fall back to the raw response so the
    # pipeline is not handed an empty string.
    return func_body_str or func_bodies


def generic_generate_internal_tests(
        func_sig: str,
        model,
        max_num_tests: int,
        test_generation_few_shot: str,
        test_generation_chat_instruction: str,
        parse_tests: Callable[[str], List[str]],
        is_syntax_valid: Callable[[str], bool],
) -> List[str]:
    """Generates tests for a function."""
    messages = [
        Message(role="system", content=f"{test_generation_chat_instruction}\n\n{test_generation_few_shot}"),
        Message(role="user", content=f"[func signature]:\n{func_sig}\n\n[unit tests]:"),
    ]
    output = model.generate(prompt=messages, max_tokens=2048*4)

    all_tests = parse_tests(output)  # type: ignore
    valid_tests = [test for test in all_tests if is_syntax_valid(test)]

    return sample_n_random(valid_tests, max_num_tests)


def generic_evaluate(
        func_sig: str,
        model,
        parse_response: Callable,
        task: str = "stop",
        code: str = "",
        temperature: float = 0.0,
        lang: str = "python",
        exe_feedback: str = "",
):
    if task == "stop":
        messages = [
            Message(role="system", content=system_stop_simplified),
            Message(role="user", content=f"Task Description:\n{func_sig}\n\nCode to Evaluate:\n```{lang}\n{code}\n```\nFeedback from executing the code on visible test cases:\n```\n{exe_feedback}\n```")
        ]
        print(messages_to_str(messages))
    elif task == "tests":
        messages = [
            Message(role="system",
                    content="Your task is to evaluate the execution outputs of a code implementation. The statement and code is given by the user, and the output/expected output on a set of test cases."
                            "Your should analyze the expected outputs and execution outputs. From a 0 to 5 range, you should give a score on how far the execution outputs are from the expected ones. Standards are below:\n"
                            "\n0: Errors or time out when executing.\n"
                            "\n1: No pattern found when comparing pairs of <output, expected_output>, errors are hard to interpret.\n"
                            "\n2: Results abnormal for a part of cases(e.g., cannot handle negative elements; only half of it sorted).\n"
                            "\n3: Result pairs have clear patterns(e.g., all elements offset by 1; all elements + 1; corp by value; reverse all elements...)\n"
                            "\n4: Lack consideration of edge condition/corner cases(e.g., error only when elements are equal), otherwise correct.\n"
                            "\n5: Results matched.\n"
                            "\nGive your brief analysis first. Afterwards, start a new line with A SINGLE INTEGER NUMBER as your final score(0 to 5)."),
            Message(role="user",
                    content=f"Task Description:\n{func_sig}\n\nCode to Evaluate:\n```{lang}\n{code}\n```\nFeedback from executing the code on visible test cases:\n\n{exe_feedback}")
        ]
    else:
        raise ValueError("task not in one of tests, stop")
    response = model.generate(prompt=messages, max_tokens=2048*4, temperature=temperature)
    result = parse_response(response)
    return result

def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)

def generic_gen_strategy(
        func_sig: str,
        model,
        parse_strategy: Callable[[str], List[str]],
        task: str = "strategy",
        given_strategy: str = "",
        incorrect_code: str = "",
        test_feedback: str = "",
        temperature: float = 0.0,
        lang: str = "python",
        num_list: str = "3",
) -> List[str]:
    if task == "strategy":
        system_prompt = PY_STRATEGY.replace("{py_strategy_k}", str(num_list))
        if "multiple" in str(num_list):
            system_prompt += Prompt_flexible
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=func_sig),
        ]
    elif task == "reflection":
        system_prompt = PY_REFELCTION.replace("{py_strategy_k}", str(num_list))
        if "multiple" in str(num_list):
            system_prompt += Prompt_flexible
        if given_strategy is None:
            given_strategy = ""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"[problem] {func_sig}\n\n[proposed strategy]{given_strategy}\n```{lang}\n{incorrect_code}\n```\n"),
            Message(role="user", content=f"[unit test results]:\n{test_feedback}"),
        ]
    else:
        raise ValueError("Must be in one of strategy/reflection")
    print(messages_to_str(messages))
    func_bodies = model.generate(prompt=messages, max_tokens=2048*4, temperature=temperature)
    assert isinstance(func_bodies, str)
    func_body_str = parse_strategy(func_bodies)
    print_generated_func_body(func_body_str)
    return func_body_str


def print_generated_func_body(func_body_str: str, raw: str = "") -> None:
    print(f"""--------------------- GENERATED FUNC BODY ---------------------
{func_body_str}
------------------------------------------""")
    if not func_body_str:
        if raw:
            print("WARNING: parse_code_block returned empty — raw model response was:\n")
            print(raw)
        else:
            print("WARNING: parse_code_block returned empty — no raw response available.")
