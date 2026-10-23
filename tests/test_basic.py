#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest
import yaml

from codehelm.algorithms.validator_feedback.feeback import Feedback
from codehelm.orchestration.base import BaseRunner
from codehelm.utils.utils import resolve_env_vars
from codehelm.validator.validator_exec import ValidatorExec
from codehelm.validator.validator_unittest import ValidatorUnitTest

os.environ["LITELLM_LOG"] = "DEBUG"


@pytest.mark.parametrize("validator", [ValidatorExec(), ValidatorUnitTest()])
def test_litellm(validator):
    with open("tests/test_configs/litellm.yaml") as file:
        config = yaml.safe_load(file)
    resolve_env_vars(config)

    results_dir = os.path.join("litellm_test", "base_runner")

    runner = BaseRunner(
        config, results_dir=results_dir, validator=validator, repair_strategy=Feedback()
    )

    runner.parse_data(dataset=config["dataset"], num_samples=config["num_samples"])

    _ = runner.validator.get_summary(results_dir)


def test_hf():
    with open("tests/test_configs/qwen.yaml") as file:
        config = yaml.safe_load(file)
    resolve_env_vars(config)
    results_dir = os.path.join("hf_test", "base_runner")

    runner = BaseRunner(
        config,
        results_dir=results_dir,
        validator=ValidatorExec(),
        repair_strategy=Feedback(),
    )
    runner.parse_data(dataset=config["dataset"], num_samples=config["num_samples"])

    _ = runner.validator.get_summary(results_dir)
