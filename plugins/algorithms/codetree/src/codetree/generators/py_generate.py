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

from .generator_utils import generic_generate_func_impl, generic_generate_internal_tests, generic_gen_strategy, generic_evaluate
from typing import Optional, List
import ast
import re
from .parse import parse_code_block, add_code_block


USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature). Don't include test cases or printing statements in the code block."
PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Python assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature)."
PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring."""

class PyGenerator:
    def func_impl(
        self,
        func_sig: str,
        model,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        temperature: float = 0.8,
    ) -> str:
        return generic_generate_func_impl(
            func_sig=func_sig,
            model=model,
            strategy=strategy,
            prev_func_impl=prev_func_impl,
            feedback=feedback,
            self_reflection=self_reflection,
            temperature=temperature,
            reflection_chat_instruction=PY_REFLEXION_CHAT_INSTRUCTION,
            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
            code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
            parse_code_block=lambda x: parse_code_block(x, "python"),
            add_code_block=lambda x: add_code_block(x, "python"),
        )

    def internal_tests(self, func_sig: str, model, max_num_tests: int = 12) -> List[str]:        
        """
        Generates tests for a function.
        """

        def parse_tests(tests: str) -> List[str]:
            return [test.strip() for test in tests.splitlines() if "assert" in test]

        return generic_generate_internal_tests(
            func_sig=func_sig,
            model=model,
            max_num_tests=max_num_tests,
            test_generation_few_shot=PY_TEST_GENERATION_CHAT_INSTRUCTION,
            test_generation_chat_instruction=PY_TEST_GENERATION_CHAT_INSTRUCTION,
            parse_tests=parse_tests,
            is_syntax_valid=py_is_syntax_valid,
        )

    def strategy(self,
                 func_sig: str,
                 model,
                 num_strategy: int=3,
                 temperature: float = 0.0,
                 prev_func_impl: Optional[str] = None,
                 feedback: Optional[str] = None,
                 given_strategy: Optional[str] = None,
                 task: str="strategy") -> List[str]:
        def parse_strategy(strategies: str) -> List[str]:
            pattern = r"^\s*<\d+>(.*)"
            pattern2 = r"\d+\.(.*)"
            new_strategies = []
            lines = strategies.splitlines()
            lines = [ele for ele in lines if ele.strip() != '']
            for line in lines:
                if len(line) < 5: continue
                a = re.search(pattern, line.strip())
                if a: new_strategies.append(a.groups()[0])
                else:
                    a = re.search(pattern2, line.strip())
                    if a: new_strategies.append(a.groups()[0])
                    # else: new_strategies.append(line)
            return new_strategies
        return generic_gen_strategy(
            func_sig=func_sig,
            model=model,
            parse_strategy=parse_strategy,
            task=task,
            incorrect_code=prev_func_impl,
            test_feedback=feedback,
            given_strategy=given_strategy,
            num_list=num_strategy
        )

    def agent_eval(self,
                 func_sig: str,
                 model,
                 temperature: float = 0.0,
                 prev_func_impl: Optional[str] = None,
                 feedback: Optional[str] = None,
                 given_strategy: Optional[str] = None,
                 task: str="stop") -> List[str]:

        def binary_stop_parser(response):
            lines_of_response = response.splitlines()
            lines_of_response = [ele for ele in lines_of_response if ele.strip() != '']
            if not lines_of_response:
                print("Sorry, empty response from model — defaulting judge to False.")
                return False, ""
            judge = True
            if "false" in lines_of_response[-1].lower(): judge=False
            elif "true" in lines_of_response[-1].lower(): judge=True
            else:
                print("Sorry, this parse of judgement doesn't seem to work.")
                print("Response:", response)
            if len(lines_of_response) > 2: return judge, "\n".join(lines_of_response[:-1])
            return judge, lines_of_response[0]


        def test_eval_parser(response):
            response_last_line = response.splitlines()[-1]
            score = 0
            for ele in ["0","1","2", "3", "4", "5"]:
                if ele in response_last_line: score = int(ele)
            return score, "\n".join(response.splitlines()[:-1])

        return generic_evaluate(
            func_sig=func_sig,
            model=model,
            parse_response=binary_stop_parser if task=="stop" else test_eval_parser,
            task=task,
            code=prev_func_impl,
            exe_feedback=feedback,
            lang="python",
        )

def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False
