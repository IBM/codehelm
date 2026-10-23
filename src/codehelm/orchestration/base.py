#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import json
import os

import jsonlines
import yaml
from tqdm import tqdm


class BaseRunner:
    """
    Supplies a simple loop over the data with the indicated validator and repair strategies.
    """

    def __init__(self, config, results_dir, validator, repair_strategy):
        self.model = self._build_model(config["model"])
        self.results_dir = results_dir
        self.repair_strategy = repair_strategy
        if repair_strategy and getattr(repair_strategy, "requires_model", False):
            self.repair_strategy.base_model = self.model
        self.validator = validator
        self.verbose = True
        self.test_format = config["test_format"]

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(model_cfg: dict):
        """Instantiate the backend class described in *model_cfg*.

        Expects:
            model_cfg["type"]         – dotted import path, e.g.
                                        ``"codehelm.models.litellm.LiteLLMModel"``
            model_cfg["model_config"] – dict forwarded verbatim to the class
                                        constructor.
        """
        dotted = model_cfg["type"]
        module_path, class_name = dotted.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        config = {"type": dotted, **model_cfg["model_config"]}
        return cls(config)

    # ------------------------------------------------------------------
    # Config loader (class-level convenience)
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls, path: str, results_dir, validator, repair_strategy
    ) -> "BaseRunner":
        """Create a :class:`BaseRunner` from a YAML config file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config, results_dir, validator, repair_strategy)

    def parse_data(self, dataset, num_samples=None):
        if dataset.endswith(".jsonl"):
            reader_ctx = jsonlines.open(dataset)
        else:
            # plain .json file — load the full array and iterate over it
            with open(dataset) as f:
                data = json.load(f)
            if num_samples is None:
                num_samples = len(data)

            reader_ctx = iter(data)

        if dataset.endswith(".jsonl"):
            with reader_ctx as reader:
                self._run_parse_loop(reader, num_samples)
        else:
            self._run_parse_loop(reader_ctx, num_samples)

    def _run_parse_loop(self, reader, num_samples):
        for sample_num, obj in enumerate(tqdm(reader, total=num_samples)):
            if num_samples and sample_num >= num_samples:
                break

            sample_dir = os.path.join(self.results_dir, "sample_" + str(sample_num))

            test = obj.get("unittest", obj.get("test"))
            if test is None:
                raise ValueError(
                    f"Sample {sample_num} has neither a 'unittest' nor a 'test' field"
                )

            prompt = obj["prompt"]
            entry_point = obj.get("entry_point")

            self.validator.setup(
                sample_dir=sample_dir,
                sample_num=sample_num,
                test=test,
                test_format=self.test_format,
                entry_point=entry_point,
                prompt=prompt,
            )
            if self.repair_strategy:
                self.repair_strategy.setup(
                    sample_dir=sample_dir,
                    test=test,
                    entry_point=entry_point,
                )

            if self.repair_strategy and getattr(
                self.repair_strategy, "owns_generation", False
            ):
                # Strategy owns the full generation loop (e.g. SFS Synthesis) —
                # call repair() once directly, skipping the initial model.generate() pass.
                repair_result = self.repair_strategy.repair(prompt, "")
                solution_code = (
                    repair_result.code
                    if hasattr(repair_result, "code")
                    else str(repair_result)
                )
                self.validator.validate(solution_code)
            else:
                # prepend imports to steer model if needed: prompt + "\n```python\n" + obj["imports"]
                raw = self.model.generate(prompt, verbose=self.verbose)
                # run unittest immediately; retry with feedback up to 3 times on failure
                passed, failure_info = self.validator.validate(raw)

                if self.repair_strategy:
                    for _ in range(3):
                        if passed:
                            break
                        failure_str = (
                            "\n".join(failure_info)
                            if isinstance(failure_info, list)
                            else (failure_info or "")
                        )
                        repair_result = self.repair_strategy.repair(prompt, failure_str)
                        if hasattr(repair_result, "code"):
                            # DirectSolution is already final code — validate directly.
                            self.validator.validate(repair_result.code)
                            break
                        # Entropy.repair() returns an EntropyRepairResult; other
                        # strategies (e.g. Feedback) return a plain string prompt.
                        elif hasattr(repair_result, "prompt_for_model") and hasattr(
                            repair_result, "response_prefix"
                        ):
                            prompt_for_model = repair_result.prompt_for_model
                            response_prefix = repair_result.response_prefix
                        else:
                            prompt_for_model = repair_result
                            response_prefix = ""
                        # todo: read apply_chat_template from config
                        raw = self.model.generate(
                            prompt_for_model,
                            verbose=self.verbose,
                            apply_chat_template=False,
                        )
                        # Prepend the forced prefix so the validator sees a complete
                        # code string, not just the continuation tail.
                        passed, failure_info = self.validator.validate(
                            response_prefix + raw
                        )
