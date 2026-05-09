#!/usr/bin/env bash
# Run eval_benchmark.py for every (model, benchmark) combination.
# Edit MODELS and BENCHMARKS below, then: bash benchmark/run_eval.sh
set -euo pipefail

# ── configuration ────────────────────────────────────────────────────────────
MODELS=(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    # "deepseek-ai/deepseek-coder-6.7b-instruct"
    # "meta-llama/CodeLlama-7b-Instruct-hf"
)

BENCHMARKS=(
    "benchmark/LLMEval-Dataset/humaneval.json"
    "benchmark/LLMEval-Dataset/mbpp.json"
)

OUTPUT_DIR="./benchmark/results"
LOG_DIR="./benchmark/logs"
GPUS="0"
MAX_NEW_TOKENS=512
DTYPE="bfloat16"
TIMEOUT=30
LIMIT=""       # set to e.g. "10" to evaluate only first N tasks
SEED=42
# Optional: space-separated variant names to restrict evaluation
# e.g. VARIANTS="original paper_2.lexical_vagueness__lv"
VARIANTS=""
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

build_args() {
    local model="$1" bench="$2"
    local args=(
        "--benchmark" "$bench"
        "--model"     "$model"
        "--outputDir" "$OUTPUT_DIR"
        "--gpus"      "$GPUS"
        "--maxNewTokens" "$MAX_NEW_TOKENS"
        "--dtype"     "$DTYPE"
        "--timeout"   "$TIMEOUT"
        "--seed"      "$SEED"
    )
    [[ -n "$LIMIT"    ]] && args+=("--limit"    "$LIMIT")
    [[ -n "$VARIANTS" ]] && args+=("--variants" $VARIANTS)
    echo "${args[@]}"
}

total=$(( ${#MODELS[@]} * ${#BENCHMARKS[@]} ))
done_count=0

for model in "${MODELS[@]}"; do
    for bench in "${BENCHMARKS[@]}"; do
        done_count=$(( done_count + 1 ))
        model_slug="${model//\//_}"
        bench_slug="$(basename "$bench" .json)"
        log_file="$LOG_DIR/${model_slug}__${bench_slug}.log"

        echo "─────────────────────────────────────────────────────────"
        echo "[$done_count/$total] model=$model  benchmark=$bench_slug"
        echo "log → $log_file"

        # shellcheck disable=SC2046
        python benchmark/eval_benchmark.py $(build_args "$model" "$bench") \
            2>&1 | tee "$log_file"

        echo "Done [$done_count/$total]"
    done
done

echo "═════════════════════════════════════════════════════════"
echo "All evaluations complete. Results in $OUTPUT_DIR"
