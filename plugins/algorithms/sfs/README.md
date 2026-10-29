# codehelm-sfs

codehelm algorithm plugin implementing
[Scattered Forest Search (SFS)](https://github.com/codespace-optimization/sfs),
an MCTS-based synthesis repair strategy for CodeHelm pipelines.

## Install

From the repository root, with codehelm already installed:

```bash
pip install -e plugins/algorithms/sfs
```

For Qiskit support:

```bash
pip install -e "plugins/algorithms/sfs[qiskit]"
```

## Usage

```python
from codehelm_sfs import SFSRepair
from codehelm.orchestration.base import BaseRunner
from codehelm.validator.validator_exec import ValidatorExec

repair_strategy = SFSRepair(generator="python", max_iters=5, num_seeds=2)
runner = BaseRunner(
    config=config,
    results_dir=results_dir,
    validator=ValidatorExec(),
    repair_strategy=repair_strategy,
)
runner.parse_data(dataset=dataset_path)
```

## Tests

```bash
pytest plugins/algorithms/sfs/tests -v
```

## Notice

Derived from [codespace-optimization/sfs](https://github.com/codespace-optimization/sfs)
(Apache-2.0). See `src/codehelm_sfs/NOTICE`.
