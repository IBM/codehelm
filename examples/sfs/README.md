# SFS Synthesis in CodeHelm

This example demonstrates how to use **Scattered Forest Search (SFS) Synthesis** as a repair strategy within CodeHelm.

SFS Synthesis uses Monte Carlo Tree Search (MCTS) with Scout (idea exploration) and Strategist (targeted code improvement) loops, guided by test execution feedback.

---

## Quickstart with Ollama

1. Start your local Ollama server and pull a coding model:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

2. Run the Qiskit HumanEval example:
   ```bash
   python examples/sfs/run.py --config examples/sfs/config_humaneval.yaml
   ```

3. Or run the QuantumKatas format example:
   ```bash
   python examples/sfs/run.py --config examples/sfs/config_quantumkatas.yaml
   ```

---

## Configuration

In `config_humaneval.yaml` / `config_quantumkatas.yaml`:

```yaml
model:
  type: codehelm.models.litellm.LiteLLMModel
  model_config:
    model_name: "qwen2.5-coder:7b"
    api_base: "http://localhost:11434/v1"
    api_key: "ollama"
    temperature: 0.6
    max_tokens: 1024

sfs:
  generator: "qiskit"      # "python" or "qiskit"
  max_iters: 5             # Maximum MCTS iterations
  num_seeds: 2             # Initial seed implementations
  temperature: 0.6
  use_problem_tests: true  # Score candidates against problem test suite
```

### Dataset

Exactly one of the following keys selects the dataset:

| Key | Description |
|---|---|
| `dataset_hf` | HuggingFace dataset id, e.g. `Qiskit/Qiskit-QuantumKatas`. Requires `dataset_hf_revision` (a pinned commit hash, for reproducibility). Use `dataset_hf_split` to pick a split (default `test`). |
| `dataset_path` | Local `.jsonl` / `.json` file (relative paths resolve against `examples/sfs/`). |
| `dataset_url` | URL of a JSON file. |

Datasets need the fields `task_id`, `prompt`, `test` and `entry_point`.

`dataset_hf` requires the `datasets` package (`pip install -e ".[datasets]"`). The dataset is downloaded once and then served from the local HuggingFace cache (`~/.cache/huggingface`), so later runs work offline. `num_samples` limits how many rows are run, taken in dataset order.

---

## Outputs

Each run writes to `result_path` (under `examples/sfs/`):

| File | Description |
|---|---|
| `run.log` | Full run log. |
| `dataset.jsonl` / `dataset.json` | Copy of the dataset used: `.jsonl` for `dataset_hf`, `.json` for `dataset_url`. Not written for `dataset_path`. |
| `summary.json` | Pass/fail summary across samples. |
| `sample_N/messages.jsonl` | Every LLM call (prompt and response), appended as the run progresses. |
| `sample_N/graph.json` | The MCTS search tree. Written only when `solve()` finishes, so a run that crashes midway has no `graph.json`. |

### Viewers

Both viewers produce a self-contained HTML file:

```bash
# LLM conversation
python examples/sfs/view_messages.py results/sample_0/messages.jsonl -o messages.html

# MCTS search tree
python examples/sfs/view_graph.py results/sample_0/graph.json -o tree.html
```

---

## Security Note

SFS Synthesis evaluates candidate code solutions during the MCTS search using Python's `exec()`. Ensure execution occurs in an isolated/sandboxed environment when evaluating untrusted LLM outputs.

---

## Attribution

Derived from [Scattered Forest Search](https://github.com/codespace-optimization/sfs) under the Apache License 2.0.
Paper: *Light, Jonathan, et al. "SFS: Smarter code space search improves LLM inference scaling." International Conference on Learning Representations. Vol. 2025. 2025.*.
