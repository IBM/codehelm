import os
import yaml

from codetree.results_summary import summarize_results
from codehelm.orchestration.base import BaseRunner
from codehelm.utils.utils import resolve_env_vars, resolve_folder_savedir
from codehelm.validator.validator_exec import ValidatorExec

from codetree.methods.llm_agent_guide import AgentGuide
import litellm

litellm.drop_params = True
def main():
    # Open the file and parse its content
    with open("litellm.yaml", "r") as file:
        config = yaml.safe_load(file)
    resolve_env_vars(config)

    model_name = config["model"]["model_config"]["model_name"]

    dataset_name = os.path.splitext(os.path.basename(config["dataset"]))[0]
    base_results_dir = resolve_folder_savedir(default_results_prefix=os.path.join(config["result_path"], dataset_name, "eval"))
    results_dir = os.path.join(base_results_dir, f"{model_name}_qiskit_data")
    print(f"\n{'='*35}\nEvaluating Model: {model_name}\n{'='*35}")
    print(f"Dataset: {config['dataset']}")
    print(f"Number of samples: {config['num_samples']}")

    # Unset keys fall back to AgentGuide's defaults (the original CodeTree settings).
    search_options = {k: config[k] for k in ("max_depth", "search_width", "max_iters") if k in config}
    runner = BaseRunner(config,
                        results_dir=results_dir,
                        repair_strategy=AgentGuide(visible_tests=config.get("visible_tests", "dataset"),
                                                   prune_rule=config.get("prune_rule", "original"),
                                                   **search_options),
                        validator=ValidatorExec())
    
    runner.parse_data(dataset=config['dataset'],
                      num_samples=config["num_samples"])
    summarize_results(results_dir)

main()