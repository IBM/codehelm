#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#
# Derived from codespace-optimization/sfs (Apache-2.0); modified for CodeHelm.
#

"""SFS Synthesis Repair Strategy for CodeHelm.

Integrates Scattered Forest Search (MCTS Synthesis) as a repair strategy
in CodeHelm pipelines.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from codehelm.algorithms.base import DirectSolution

from .executor import PyExecutor
from .model_adapter import ModelAdapter
from .prompts_python import PyGenerator
from .prompts_qiskit import QiskitGenerator
from .synthesis import SynthesisSolver
from .types import CodingProblem

logger = logging.getLogger(__name__)


class SFSRepair:
    """CodeHelm repair strategy backed by SFS Synthesis (MCTS).

    Attributes:
        requires_model: True — indicates BaseRunner should provide the pipeline model.
        owns_generation: True — indicates SFS manages full generation and returns final code.
    """

    requires_model: bool = True
    owns_generation: bool = True

    def __init__(
        self,
        generator: str = "python",
        model: Any = None,
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
        safe_mode: bool = False,
        executor_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.generator_type = generator.lower()
        self.base_model = model
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
        self.safe_mode = safe_mode
        self.executor_params = executor_params or {}
        self.extra_kwargs = kwargs

        self._sample_dir: str = ""
        self._test: str = ""
        self._entry_point: str = ""

    def setup(
        self,
        sample_dir: str = "",
        test: str = "",
        entry_point: str = "",
        **kwargs: Any,
    ) -> None:
        """Reset per-sample state before solving a task."""
        self._sample_dir = sample_dir
        self._test = test
        self._entry_point = entry_point or ""
        if not self._entry_point:
            logger.debug(
                "SFSRepair.setup: entry_point is empty for sample_dir=%s", sample_dir
            )

    def _get_generator(self):
        if self.generator_type == "qiskit":
            return QiskitGenerator()
        return PyGenerator()

    def repair(self, prompt: str, failure_info: str = "") -> DirectSolution:
        """Run SFS Synthesis search to generate and refine a candidate solution."""
        if self.base_model is None:
            raise ValueError(
                "No model available for SFSRepair. Pass model to SFSRepair(...) or configure BaseRunner."
            )

        log_path = (
            os.path.join(self._sample_dir, "messages.jsonl") if self._sample_dir else ""
        )
        model_adapter = ModelAdapter(self.base_model, log_path=log_path)
        gen = self._get_generator()
        exec_kwargs = dict(self.executor_params)
        exec_kwargs.setdefault("safe_mode", self.safe_mode)
        executor = PyExecutor(**exec_kwargs)

        solver = SynthesisSolver(
            generator=gen,
            model=model_adapter,
            executor=executor,
            max_iters=self.max_iters,
            num_seeds=self.num_seeds,
            temperature=self.temperature,
            use_problem_tests=self.use_problem_tests,
            number_of_tests=self.number_of_tests,
            testgen_method=self.testgen_method,
            codegen_method=self.codegen_method,
            num_ideas_per_strategy=self.num_ideas_per_strategy,
            is_puct=self.is_puct,
            return_any_solved=self.return_any_solved,
        )

        task_id = os.path.basename(self._sample_dir) if self._sample_dir else "unknown"
        problem = CodingProblem(
            task_id=task_id,
            prompt=prompt,
            entry_point=self._entry_point,
            test=self._test,
        )

        result = solver.solve(problem, self._sample_dir)
        return DirectSolution(code=result.solution or "")
