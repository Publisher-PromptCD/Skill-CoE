# Skill-CoE

Skill-CoE learns and updates reusable Skills from interactive tasks. The code supports ALFWorld, AppWorld, and ScienceWorld.

## Requirements

- Python 3.11 and an OpenAI-compatible model server. The server must expose `/chat/completions` and return prompt log-probabilities from `/completions`.
- A tokenizer matching the served model. Our runs use Qwen3.8-27B.
- Benchmark packages and data. ALFWorld needs its TextWorld configuration and game files; AppWorld needs its application data; ScienceWorld needs Java.

Install the package and the required adapter:

```bash
python -m pip install -e '.[model,alfworld]'      # ALFWorld
python -m pip install -e '.[model,scienceworld]'  # ScienceWorld
python -m pip install -e '.[model]'               # AppWorld
```

The package requires `transformers>=4.45`; the ALFWorld extra also requires `PyYAML>=6`. The extras install ALFWorld 0.4.2 or ScienceWorld 1.2.3. For AppWorld 0.2.0.dev0, install the [official source](https://github.com/stonybrooknlp/appworld/tree/42b5bcf3cd334fee33f0c37c02070a9f5807add5) separately:

```bash
python -m pip install 'git+https://github.com/stonybrooknlp/appworld.git@42b5bcf3cd334fee33f0c37c02070a9f5807add5'
appworld install
```

Install and start the model server separately.

## Run

Copy the matching file from `configs/` to `configs/local/` and fill in the model URL, tokenizer path, and benchmark paths. Prepare a JSON file with `batches`, `validation`, and `test` arrays. ALFWorld tasks use `gamefile` and `split`; AppWorld uses `task_id`; ScienceWorld uses `task_name` and `variation_id`.

```bash
skill-coe-train --config configs/local/coe.alfworld.json --data tasks.json --output outputs/run-01
```

Use `coe.appworld.json` or `coe.scienceworld.json` for the other benchmarks. Add `--state state.json` to start from existing Skills.
