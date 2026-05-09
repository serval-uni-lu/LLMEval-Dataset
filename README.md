# Benchmark Evaluation

End-to-end evaluation of LLM code generation on the LLMEval dataset, covering both clean and mutated task descriptions across HumanEval and MBPP.

## Dataset

Located in `LLMEval-Dataset/`. A unified benchmark dataset combining HumanEval, MBPP, and robustness-focused variants from multiple papers to evaluate how well LLMs handle imperfect programming task descriptions, including ambiguous, incomplete, contradictory, and underspecified prompts. The dataset supports research on code generation robustness and reliability under real-world task conditions.

Mutation variants are organised by paper:

| Paper | Mutation types |
|-------|---------------|
| paper_1 | `incomplete`, `ambiguous`, `contradictory` |
| paper_2 | `lexical_vagueness__lv`, `syntax_and_formatting_sf`, `under-specification_us` |
| paper_3 | HumanEval-only additional variants |

Each record also has an `original` variant (unmodified prompt).

## Files

| File | Description |
|------|-------------|
| `eval_benchmark.py` | Main evaluation script |
| `run_eval.sh` | Multi-model runner — edit and launch this |
| `results/` | Per-run output: `<model>__<dataset>.json` and `<model>__<dataset>__summary.csv` |
| `logs/` | Per-run stdout/stderr logs |
| `LLMEval-Dataset/humaneval.json` | HumanEval tasks with all mutation variants |
| `LLMEval-Dataset/mbpp.json` | MBPP tasks with all mutation variants |

## Quick start

### Single model

```bash
python benchmark/eval_benchmark.py \
    --benchmark benchmark/LLMEval-Dataset/humaneval.json \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --outputDir ./benchmark/results \
    --gpus 0
```

### Multiple models

Edit the `MODELS` and `BENCHMARKS` arrays at the top of `run_eval.sh`, then run from the repo root:

```bash
bash benchmark/run_eval.sh
```

## CLI reference (`eval_benchmark.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--benchmark` | *(required)* | Path to `humaneval.json` or `mbpp.json` |
| `--model` | *(required)* | HuggingFace model ID |
| `--outputDir` | `./benchmark/results` | Directory for output files |
| `--gpus` | `None` | `CUDA_VISIBLE_DEVICES` value (e.g. `"0"` or `"0,1"`) |
| `--variants` | all | Space-separated subset of variant keys to evaluate |
| `--maxNewTokens` | `512` | Max new tokens per generation |
| `--dtype` | `bfloat16` | Model weight dtype (`float16` or `bfloat16`) |
| `--timeout` | `30` | Wall-clock timeout per problem (seconds) |
| `--limit` | all | Evaluate only the first N tasks |
| `--seed` | `42` | Random seed |

## Output format

**`<model>__<dataset>.json`** — full results per task:
```json
[
  {
    "task_id": "HumanEval/0",
    "variants": {
      "original": {
        "GeneratedCode": "...",
        "GeneratedResponse": "...",
        "PromptUsed": "...",
        "n_Tests": 5,
        "Tests_Passed": 5,
        "Pass@1": true,
        "Eval_Status": "OK"
      },
      "paper_2.lexical_vagueness__lv": { "..." : "..." }
    }
  }
]
```

**`<model>__<dataset>__summary.csv`** — aggregated Pass@1 per variant:

| Model | Benchmark | Variant | Samples | Pass@1 | SuccessExecRate |
|-------|-----------|---------|---------|--------|-----------------|
| Qwen/... | humaneval | original | 164 | 0.823 | 0.951 |
| Qwen/... | humaneval | paper_2.lexical_vagueness__lv | 164 | 0.756 | 0.933 |

## Evaluation pipeline

1. Each task prompt is wrapped in a structured instruction asking for a single Python function.
2. The model generates code (greedy decoding, `do_sample=False`).
3. Code is extracted from markdown fences or `[PYTHON]` tags.
4. Tests are run in an isolated subprocess with a configurable timeout.
5. `Pass@1 = True` when all test cases pass.
