"""Run SFS Synthesis with CodeHelm.

Usage:
    # Run with Qiskit HumanEval config:
    python examples/sfs/run.py --config examples/sfs/config_humaneval.yaml

    # Run with QuantumKatas config:
    python examples/sfs/run.py --config examples/sfs/config_quantumkatas.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import yaml

from codehelm.algorithms.sfs import SFSRepair
from codehelm.orchestration.base import BaseRunner
from codehelm.utils.utils import resolve_env_vars, resolve_folder_savedir, urlopen
from codehelm.validator.validator_exec import ValidatorExec

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Run SFS Synthesis in CodeHelm")
    parser.add_argument(
        "--config",
        type=str,
        default=str(HERE / "config_humaneval.yaml"),
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    resolve_env_vars(config)

    results_dir = resolve_folder_savedir(
        str(HERE / config.get("result_path", "results"))
    )
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "run.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    # Resolve dataset
    if "dataset_hf" in config:
        # Downloaded once, then served from the local HuggingFace cache.
        from datasets import load_dataset

        hf_ds = load_dataset(
            config["dataset_hf"],
            split=config.get("dataset_hf_split", "test"),
            revision=config["dataset_hf_revision"],
        )
        dataset_path = os.path.join(results_dir, "dataset.jsonl")
        with open(dataset_path, "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in hf_ds)
    elif "dataset_url" in config:
        with urlopen(config["dataset_url"]) as response:
            dataset = json.load(response)[: config.get("num_samples", 10)]
        dataset_path = os.path.join(results_dir, "dataset.json")
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)
    elif "dataset_path" in config:
        raw_path = Path(config["dataset_path"])
        if not raw_path.is_absolute():
            raw_path = HERE / raw_path
        dataset_path = str(raw_path)
    else:
        raise ValueError(
            "Config must specify 'dataset_hf', 'dataset_url' or 'dataset_path'"
        )

    # Initialize SFS Repair strategy
    sfs_cfg = config.get("sfs", {})
    repair_strategy = SFSRepair(
        generator=sfs_cfg.get("generator", "python"),
        max_iters=sfs_cfg.get("max_iters", 5),
        num_seeds=sfs_cfg.get("num_seeds", 2),
        temperature=sfs_cfg.get("temperature", 0.6),
        use_problem_tests=sfs_cfg.get("use_problem_tests", True),
        safe_mode=sfs_cfg.get("safe_mode", False),
        number_of_tests=sfs_cfg.get("number_of_tests", 3),
        num_ideas_per_strategy=sfs_cfg.get("num_ideas_per_strategy", 3),
        testgen_method=sfs_cfg.get("testgen_method", "simple"),
        codegen_method=sfs_cfg.get("codegen_method", "outcome"),
        is_puct=sfs_cfg.get("is_puct", False),
        return_any_solved=sfs_cfg.get("return_any_solved", False),
    )

    validator = ValidatorExec()
    runner = BaseRunner(
        config=config,
        results_dir=results_dir,
        validator=validator,
        repair_strategy=repair_strategy,
    )

    runner.parse_data(
        dataset=dataset_path,
        num_samples=config.get("num_samples"),
    )
    ValidatorExec.get_summary(results_dir)
    print(f"Results written to: {results_dir}")


if __name__ == "__main__":
    main()
