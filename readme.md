# Skill-CoE

Official code for **BEYOND INITIAL SKILLS: CONTRAST-GUIDED SKILL EVOLUTION FOR ROBUST LEARNING IN LLM AGENTS**.

## Installation

Run the commands from the repository root, using Python 3.11. The GPU experiments
use Linux and CUDA. Install the package with the adapter needed for your benchmark:

```bash
python -m pip install -e '.[model,alfworld]'
python -m pip install -e '.[model,scienceworld]'
python -m pip install -e '.[model]'              # AppWorld
```

The benchmark dependencies are ALFWorld 0.4.2 and ScienceWorld 1.2.3. ScienceWorld
also requires Java. AppWorld 0.2.0.dev0 is installed from its official source:

```bash
python -m pip install 'git+https://github.com/stonybrooknlp/appworld.git@42b5bcf3cd334fee33f0c37c02070a9f5807add5'
appworld install
```

Download each benchmark's data separately. For ALFWorld, point `data_root` to
the directory containing `json_2.1.1/`, and provide the official `config_tw.yaml`.
For AppWorld, run `appworld download data` in its root directory and point
`data_root` to that root, which contains `data/tasks/`. ScienceWorld loads instances
from its installed environment by task name and variation ID. See
[Task identifiers](data/README.md#task-identifiers) for the exact lookup rules.

Download [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) separately,
including its tokenizer. Serve the model with vLLM or a compatible server that
supports `/chat/completions` and echoed prompt log-probabilities at `/completions`.
Adjust the inference dependencies and GPU settings for your environment.

## Files

```text
src/skill_coe/       Agent, environment adapters, analysis, scoring, and updates
configs/            Benchmark configuration templates
data/splits/        Ordered training batches, validation IDs, and test IDs
pyproject.toml      Dependencies and command-line entry points
```

`coe.py` runs batch learning. `grouped_probe.py` groups tasks and analyzes their
trajectories. `coe_skills.py` applies local Skill edits, `coe_evidence.py` constructs
preference pairs and scores actions, and `recombination.py` performs skill recombination.
The agent and benchmark adapters are in `agent.py` and `environments/`.

## Data

The repository includes the task lists used in the experiments, without benchmark
data or answers. Batch order is fixed because Skills are updated from the collected
trajectories after each batch and then used in the next batch. All tasks within a
batch use the same Skills, so they can run in any order or concurrently. See
[data/README.md](data/README.md) for the format and split rules.

| Benchmark | Training | Validation | Test |
| --- | --- | --- | --- |
| ALFWorld | 5 × 12 | 60 | 134 |
| AppWorld | 5 × 9 | 57 | 168 |
| ScienceWorld | 5 × 12 | 60 | 271 |

## Run

Copy `configs/coe.alfworld.example.json` to a local configuration file. Set the
model URL, tokenizer path, and benchmark paths. The examples retain the experiment
settings: a 20-round history window, a 50-round decision limit, and seed 42.
AppWorld and ScienceWorld use 9 episode workers; ALFWorld uses 8.

```bash
mkdir -p configs/local
cp configs/coe.alfworld.example.json configs/local/coe.alfworld.json
# Edit configs/local/coe.alfworld.json before running.
skill-coe-train --config configs/local/coe.alfworld.json \
  --data data/splits/alfworld.json --output outputs/alfworld-01
```

For AppWorld or ScienceWorld, copy and edit the corresponding config, then run:

```bash
skill-coe-train --config configs/local/coe.appworld.json \
  --data data/splits/appworld.json --output outputs/appworld-01
skill-coe-train --config configs/local/coe.scienceworld.json \
  --data data/splits/scienceworld.json --output outputs/scienceworld-01
```

The supplied configs start from empty Skills and skip the initial test. To continue
from an existing library, add `--state state.json` and set
`evaluation.initial_test` to `true` to enable the initial test.
Use a new output directory for each run. Resume an interrupted run with
`--resume-run outputs/previous-run` and a new `--output` directory.

The best Skills are saved in `<output>/best.json`; their final test results are
saved in `<output>/test_final/summary.json`.
