#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
from typing import Annotated, Any, Literal

import jsonlines
from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.core import Requirement, ValidationResult
from mellea.stdlib.context import SimpleContext
from pydantic import Field
from tqdm import tqdm

from codehelm.algorithms.validator_feedback.mellea_strategy import (
    MelleaFeedbackStrategy,
)

# The entropy algorithm (codehelm.algorithms.entropy) is not included in this repository,
# so MelleaEntropyStrategy is imported only if available and its code paths are skipped otherwise.
try:
    from codehelm.algorithms.entropy.mellea_strategy import (  # ty: ignore[unresolved-import]
        MelleaEntropyStrategy,
    )
except ImportError:
    MelleaEntropyStrategy: Any = None


def _is_entropy_strategy(strategy):
    return MelleaEntropyStrategy is not None and isinstance(
        strategy, MelleaEntropyStrategy
    )


def _to_mellea_validation_fn(codehelm_validate_fn):
    """Wrap a codehelm validate(ctx) -> (passed, failure_info) into a Mellea validation_fn."""

    def _validation_fn(ctx):
        passed, failure_info = codehelm_validate_fn(ctx)
        if isinstance(failure_info, list):
            failure_info = "\n".join(failure_info)
        return ValidationResult(passed is True, reason=failure_info or "")

    return _validation_fn


class MelleaRunnerConfig:
    type: Literal["codehelm.orchestration.mellea.MelleaConnector"]
    connector_backend: Annotated[
        dict,
        Field(
            description="Backend to use for Mellea.",
        ),
    ]
    validator: Annotated[
        str | None,
        Field(
            description="dotted import path to codehelm validator",
        ),
    ] = None
    strategy: Annotated[
        str | None,
        Field(
            description="Dotted import path of a SamplingStrategy class to use when validators are present, "
            "e.g. 'mellea.stdlib.sampling.MultiTurnStrategy'. "
            "Defaults to RejectionSamplingStrategy when not set.",
        ),
    ] = None


class MelleaRunner:
    config: MelleaRunnerConfig

    def __init__(self, config, results_dir, validator, repair_strategy=None):
        if config["model"]["type"] in [
            "huggingface_backend",
            "codehelm.models.huggingface.HuggingFaceModel",
        ]:
            from mellea.backends.huggingface import LocalHFBackend

            model_id = config["model"]["model_config"]["pretrained_model_name_or_path"]
            m_backend = LocalHFBackend(model_id=model_id)
        elif config["model"]["type"] == "codehelm.models.litellm.LiteLLMModel":
            from mellea.backends.litellm import LiteLLMBackend

            model_config = config["model"]["model_config"]
            # Configs use either base_url (Mellea's name) or api_base (LiteLLM's name)
            m_backend = LiteLLMBackend(
                model_id=model_config["model_id"],
                base_url=model_config.get("base_url") or model_config.get("api_base"),
            )
        else:
            raise ValueError(
                f"Unkown backend_type: either huggingface_backend or litellm_backend. Got {config['model']['type']}"
            )

        self.m = MelleaSession(backend=m_backend, ctx=SimpleContext())
        self.results_dir = results_dir
        self.validator = validator
        self.test_format = config["test_format"]

        self.max_new_tokens = 4000
        self.repair_strategy = repair_strategy
        if repair_strategy and _is_entropy_strategy(repair_strategy):
            model_obj = getattr(m_backend, "model", getattr(m_backend, "_model", None))
            tokenizer_obj = getattr(
                m_backend, "tokenizer", getattr(m_backend, "_tokenizer", None)
            )
            if model_obj is None or tokenizer_obj is None:
                raise ValueError(
                    "MelleaEntropyStrategy requires a local HuggingFace backend with accessible "
                    "model and tokenizer weights (for logits-based entropy computation). "
                    f"The configured backend ({type(m_backend).__name__}) does not expose them. "
                    "Use a HuggingFace model config or remove the repair_strategy argument."
                )
            from codehelm.models.huggingface import HuggingFaceModel

            repair_strategy._entropy.base_model = (
                HuggingFaceModel.from_model_and_tokenizer(
                    model=model_obj, tokenizer=tokenizer_obj
                )
            )

    @classmethod
    def config_model(cls) -> type[MelleaRunnerConfig]:
        """
        Return the Pydantic model class for this attack strategy

        :return: Pydantic model class
        """
        return MelleaRunnerConfig

    @staticmethod
    def template() -> dict[str, Any]:
        return {
            "type": "codehelm.orchestration.mellea.MelleaConnector",
            "name": "mellea",
        }

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
                self.generate(reader, num_samples)
        else:
            self.generate(reader_ctx, num_samples)

    def generate(self, reader, num_samples):

        for sample_num, obj in enumerate(tqdm(reader, total=num_samples)):
            if num_samples and sample_num >= num_samples:
                break

            sample_dir = os.path.join(self.results_dir, "sample_" + str(sample_num))

            if "unittest" in obj:
                test = obj["unittest"]
            elif "test" in obj:
                test = obj["test"]

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

            model_options = (
                {ModelOption.MAX_NEW_TOKENS: self.max_new_tokens}
                if self.max_new_tokens is not None
                else None
            )

            # Wire feedback strategy for this sample, so its repair() feeds the
            # validator failures (tests, LintQ) back to the model with the prompt.
            if isinstance(self.repair_strategy, MelleaFeedbackStrategy):
                self.repair_strategy._current_prompt = prompt
                MelleaFeedbackStrategy.set_active(self.repair_strategy)

            # Wire entropy strategy for this sample if present.
            if _is_entropy_strategy(self.repair_strategy):
                entropy_strategy: Any = self.repair_strategy
                entropy_strategy._entropy.setup(sample_dir=sample_dir)
                entropy_strategy._current_prompt = prompt
                entropy_strategy._current_response_prefix = ""
                MelleaEntropyStrategy.set_active(entropy_strategy)
                validation_fn = entropy_strategy.make_validation_fn(
                    self.validator.validate
                )
                active_strategy = entropy_strategy
            else:
                validation_fn = _to_mellea_validation_fn(self.validator.validate)
                active_strategy = self.repair_strategy

            requirement = Requirement("Must pass tests.", validation_fn=validation_fn)
            result = self.m.instruct(
                prompt,
                requirements=[requirement],
                strategy=active_strategy,
                model_options=model_options,
            )

            # Clean up active instance for this sample.
            if isinstance(self.repair_strategy, MelleaFeedbackStrategy):
                MelleaFeedbackStrategy.set_active(None)
            if _is_entropy_strategy(self.repair_strategy):
                MelleaEntropyStrategy.set_active(None)

        return str(result)
