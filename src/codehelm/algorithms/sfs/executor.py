#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#
# Derived from codespace-optimization/sfs (Apache-2.0); modified for CodeHelm.
#

"""Scoring executor for candidate solutions in SFS Synthesis.

Executes candidate code against test suites with import hoisting, timeout isolation,
private execution namespace, and detailed per-test failure feedback.

WARNING: This executor runs model-generated code via Python's built-in exec().
Ensure execution occurs in a sandboxed environment when running untrusted models.
"""

from __future__ import annotations

import ast
import logging
import os
import textwrap
import traceback
from typing import Any

from codehelm.validator.validator_exec import timeout

from .code_safety import check_code_safety
from .types import CodingProblemAndSolution, ExecuteResult

os.environ.setdefault("MPLBACKEND", "Agg")
logger = logging.getLogger(__name__)


def _extract_imports(code: str) -> str:
    """Extract top-level imports from code, deduplicated and in source order."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return ""
    lines = code.splitlines()
    seen: set[str] = set()
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if (
                hasattr(node, "lineno")
                and hasattr(node, "end_lineno")
                and node.end_lineno is not None
            ):
                import_src = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            else:
                import_src = ast.unparse(node)
            if import_src not in seen:
                seen.add(import_src)
                result.append(import_src)
    return "\n".join(result)


def _with_test_calls(test: str, entry_point: str = "") -> str:
    """Ensure test functions (e.g. check() or test_*()) are invoked."""
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return test

    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("test"):
                calls.append(f"{node.name}()")
            elif node.name == "check":
                if entry_point:
                    calls.append(f"check({entry_point})")
                else:
                    # Look for functions defined in candidate or use check()
                    calls.append("check(candidate)")

    if calls:
        return test + "\n" + "\n".join(calls)
    return test


def get_call_str(assert_statement: str) -> str:
    """Extract callable expression from an assert statement."""
    try:
        ast_parsed = ast.parse(assert_statement)
        if not ast_parsed.body:
            return ""
        stmt = ast_parsed.body[0]
        if isinstance(stmt, ast.Assert):
            test_node = stmt.test
            if isinstance(test_node, ast.Compare):
                return ast.unparse(test_node.left).strip()
            return ast.unparse(test_node).strip()
    except (SyntaxError, ValueError):
        return ""
    return ""


def get_output(
    func: str,
    assert_statement: str,
    timeout_sec: int = 5,
    prompt_imports: str = "",
) -> str:
    """Evaluate callable expression in assert statement to produce diagnostic output."""
    try:
        output_globals: dict[str, Any] = {}
        exec_code = f"from __future__ import annotations\nfrom typing import *\n{prompt_imports}\n{func}"
        with timeout(timeout_sec):
            exec(exec_code, output_globals)  # nosec # noqa: S102

        func_call = get_call_str(assert_statement)
        if not func_call:
            return "N/A (complex statement)"

        with timeout(timeout_sec):
            out = eval(func_call, output_globals)  # nosec
        return str(out)
    except TimeoutError:
        return "TIMEOUT"
    except Exception as exc:  # noqa: BLE001 - candidate code may raise anything
        return f"{type(exc).__name__}: {exc}"


class PyExecutor:
    """Scoring executor for candidate Python / Qiskit solutions."""

    def __init__(
        self,
        safe_mode: bool = False,
        timeout_seconds: int = 5,
        **kwargs: Any,
    ) -> None:
        self.safe_mode = safe_mode
        self.timeout_seconds = timeout_seconds
        if self.safe_mode:
            logger.warning(
                "PyExecutor running in SAFE MODE - AST validation only, no execution"
            )

    def execute(
        self,
        problem_and_solution: CodingProblemAndSolution,
        timeout_sec: int | None = None,
        will_modify: bool = True,
    ) -> ExecuteResult:
        """Execute visible tests against proposed solution."""
        effective_timeout = timeout_sec or self.timeout_seconds
        func = problem_and_solution.solution
        if func is None or not func.strip():
            res = ExecuteResult(False, "No solution provided", ())
            if will_modify:
                problem_and_solution.is_passing = False
                problem_and_solution.feedback_string = res.feedback
                problem_and_solution.test_results = res.state
            return res

        tests = problem_and_solution.visible_tests
        if not tests:
            res = ExecuteResult(True, "No visible tests provided", ())
            if will_modify:
                problem_and_solution.is_passing = True
                problem_and_solution.feedback_string = res.feedback
                problem_and_solution.test_results = ()
            return res

        if self.safe_mode:
            res = self._validate_ast_only(func, tests)
            if will_modify:
                problem_and_solution.is_passing = res.is_passing
                problem_and_solution.feedback_string = res.feedback
                problem_and_solution.test_results = res.state
            return res

        safety_report = check_code_safety(func)
        if not safety_report.is_safe:
            logger.warning("Safety check warning: %s", safety_report.summary())

        base_imports = "from __future__ import annotations\nfrom typing import *"
        prompt_imports = _extract_imports(problem_and_solution.prompt)
        func_imports = _extract_imports(func)
        extra = "\n".join(filter(None, [prompt_imports, func_imports]))
        imports = base_imports + ("\n" + extra if extra else "")

        entry_point = problem_and_solution.entry_point
        combined_tests = "\n".join(_with_test_calls(t, entry_point) for t in tests)
        combined_code = f"{imports}\n{func}\n{combined_tests}"

        try:
            exec_globals: dict[str, Any] = {}
            with timeout(effective_timeout):
                exec(combined_code, exec_globals)  # nosec # noqa: S102

            state = (True,) * len(tests)
            feedback = "All tests passed successfully."
            res = ExecuteResult(True, feedback, state)
            if will_modify:
                problem_and_solution.is_passing = True
                problem_and_solution.feedback_string = feedback
                problem_and_solution.test_results = state
            return res
        except Exception as exc:  # noqa: BLE001 - candidate code may raise anything
            logger.debug("Test batch execution failed: %s", exc)

            # Re-execute individual tests to identify individual passes and failures
            success_tests = []
            failed_tests = []
            state_list: list[bool] = []

            for test in tests:
                test_code = f"{imports}\n{func}\n{_with_test_calls(test, entry_point)}"
                try:
                    test_globals: dict[str, Any] = {}
                    with timeout(effective_timeout):
                        exec(test_code, test_globals)  # nosec # noqa: S102
                    success_tests.append(test)
                    state_list.append(True)
                except NameError as ne:
                    failed_tests.append(f"{test} # error: undefined variable - {ne}")
                    state_list.append(False)
                except Exception as test_exc:  # noqa: BLE001 - candidate code may raise anything
                    output = get_output(
                        func,
                        test,
                        timeout_sec=effective_timeout,
                        prompt_imports=prompt_imports,
                    )
                    if output == "N/A (complex statement)":
                        tb_tail = traceback.format_exc().strip().splitlines()[-1]
                        output = f"{type(test_exc).__name__}: {test_exc} ({tb_tail})"
                    failed_tests.append(f"{test} # output: {output}")
                    state_list.append(False)

            feedback_lines = ["Tests failed:"]
            feedback_lines.extend(failed_tests[:5])
            feedback_lines.append("\nTests passed:")
            feedback_lines.extend(success_tests[:5])

            if not safety_report.is_safe:
                feedback_lines.append(f"\n{safety_report.summary()}")

            feedback = "\n".join(feedback_lines)
            res = ExecuteResult(False, feedback, tuple(state_list))
            if will_modify:
                problem_and_solution.is_passing = False
                problem_and_solution.feedback_string = feedback
                problem_and_solution.test_results = tuple(state_list)
            return res

    def _validate_ast_only(self, func: str, tests: tuple[str, ...]) -> ExecuteResult:
        """Validate code syntax without execution."""
        try:
            ast.parse(func)
            for test in tests:
                ast.parse(test)
            state = (True,) * len(tests)
            return ExecuteResult(
                True,
                "SAFE MODE: AST validation passed (code not executed).",
                state,
            )
        except SyntaxError as e:
            state = (False,) * len(tests)
            return ExecuteResult(
                False,
                f"SAFE MODE: Syntax error detected: {e}",
                state,
            )

    def evaluate(
        self,
        problem_and_solution: CodingProblemAndSolution,
        timeout_sec: int | None = None,
        will_modify: bool = True,
    ) -> bool:
        """Evaluate candidate against hidden test suite."""
        effective_timeout = timeout_sec or self.timeout_seconds
        test = problem_and_solution.test
        func = problem_and_solution.solution

        if not func:
            if will_modify:
                problem_and_solution.is_solved = False
            return False

        if not test:
            # If no hidden test is set, fall back to checking if visible tests pass
            is_solved = problem_and_solution.is_passing
            if will_modify:
                problem_and_solution.is_solved = is_solved
            return is_solved

        if self.safe_mode:
            try:
                ast.parse(f"{func}\n\n{test}")
                if will_modify:
                    problem_and_solution.is_solved = True
                return True
            except SyntaxError:
                if will_modify:
                    problem_and_solution.is_solved = False
                return False

        prompt_imports = _extract_imports(problem_and_solution.prompt)
        func_imports = _extract_imports(func)
        base_imports = "from __future__ import annotations\nfrom typing import *"
        extra = "\n".join(filter(None, [prompt_imports, func_imports]))
        imports = base_imports + ("\n" + extra if extra else "")

        entry_point = problem_and_solution.entry_point
        test_block = _with_test_calls(test, entry_point)
        code = f"{imports}\n{func}\n\n{test_block}"

        try:
            eval_globals: dict[str, Any] = {}
            with timeout(effective_timeout):
                exec(code, eval_globals)  # nosec # noqa: S102
            if will_modify:
                problem_and_solution.is_solved = True
            return True
        except Exception as exc:  # noqa: BLE001 - candidate code may raise anything
            logger.debug("Evaluation failed: %s", exc)
            if will_modify:
                problem_and_solution.is_solved = False
            return False
