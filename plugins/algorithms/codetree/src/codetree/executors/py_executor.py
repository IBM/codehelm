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

import ast
import astunparse
import subprocess
import os
import sys
from typing import NamedTuple, List, Tuple

class ExecuteResult(NamedTuple):
    is_passing: bool
    feedback: str
    state: Tuple[bool]

def check_syntax_error_source(source_code):
    try:
        ast.parse(source_code)
        return False, None
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return True, e

class PyExecutor():
    def execute(self, func: str, tests: List[str], timeout: int = 5, prev="") -> ExecuteResult:
        os.makedirs("temp_files", exist_ok=True)
        error, syn = check_syntax_error_source(func)
        if error:
            return ExecuteResult(False, str(error), tuple([False] * len(tests)))

        imports = (
            'from typing import *\nfrom collections import *\n'
            'import math\nimport sys\nimport random\nimport re\n'
            'import itertools, functools, bisect, heapq, string\n'
        )
        if prev:
            imports += f"{prev}\n"
        func_test_list = [f'{imports}\n{func}\n{test}\n' for test in tests]

        success_tests = []
        failed_tests = []
        pid = os.getpid()
        for i, test in enumerate(tests):
            try:
                path = f"temp_files/temp_test{pid}.py"
                with open(path, "w") as temp_file:
                    temp_file.write(func_test_list[i])

                result = subprocess.run(
                    [sys.executable, path],
                    capture_output=True, text=True, timeout=timeout,
                )
                if result.returncode == 0:
                    success_tests.append(test)
                    continue

                # check(candidate)-style tests have no top-level assert to re-evaluate;
                # the traceback of the failed run is the useful feedback.
                if not any(isinstance(n, ast.Assert) for n in ast.parse(test).body):
                    failed_tests.append(f"{test} # incorrect program's output: {result.stderr[-1500:]}")
                    continue

                # Build a diagnostic script to capture the actual output.
                call_str = get_call_str(test)
                setup_lines = [l for l in test.splitlines() if not l.strip().startswith("assert")]
                diag_script = f'{imports}\n{func}\n{chr(10).join(setup_lines)}\nprint({call_str})'

                path = f"temp_files/temp_test_output{pid}.py"
                with open(path, "w") as temp_file:
                    temp_file.write(diag_script)

                result = subprocess.run(
                    [sys.executable, path],
                    capture_output=True, text=True, timeout=timeout,
                )
                output = (result.stderr + result.stdout) if result.stderr else result.stdout
                failed_tests.append(f"{test} # incorrect program's output: {output}")
            except subprocess.TimeoutExpired:
                failed_tests.append(f"{test} # incorrect program's output: Program Timed Out after {timeout} seconds.")
            except Exception as e:
                failed_tests.append(f"{test} # incorrect program's output: {e}")

        success_set = set(success_tests)
        state = tuple(test in success_set for test in tests)
        is_passing = len(failed_tests) == 0

        feedback_parts = []
        if success_tests:
            feedback_parts.append("Tested passed:\n" + "\n".join(success_tests))
        if failed_tests:
            feedback_parts.append("Tests failed:\n" + "\n".join(failed_tests))
        feedback = "\n".join(feedback_parts)
        if len(feedback) > 2500:
            feedback = feedback[:2500]
        return ExecuteResult(is_passing, feedback, state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5, prev="") -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        os.makedirs("temp_files", exist_ok=True)
        code = f"""from typing import *
from collections import *
import math, itertools, functools, bisect, heapq, string
import sys
import re
{prev}
{func}

{test}
"""
        try:
            path = f"temp_files/temp_test{os.getpid()}.py"
            with open(path, "w") as temp_file:
                temp_file.write(code)
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=timeout,
            )
            return result.returncode == 0
        except Exception:
            return False

def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    # body[0] may be an Assign (setup line) rather than an Assert for multi-line tests;
    # find the first Assert node instead.
    assert_node = None
    for node in ast_parsed.body:
        if isinstance(node, ast.Assert):
            assert_node = node
            break
    if assert_node is None:
        # No assert found — return the raw statement for diagnostic purposes
        return assert_statement.strip()
    try:
        call_str = assert_node.test.left  # type: ignore
    except AttributeError:
        call_str = assert_node.test  # type: ignore
    return astunparse.unparse(call_str).strip()

