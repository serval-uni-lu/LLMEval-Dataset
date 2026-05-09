#!/usr/bin/env python3
"""
Evaluate LLM code generation on the LLMEval benchmark dataset.

Supports benchmark/LLMEval-Dataset/humaneval.json and mbpp.json.
Each record is evaluated across all (non-empty) mutation variants:
  original, paper_1.incomplete, paper_1.ambiguous, paper_1.contradictory,
  paper_2.lexical_vagueness__lv, paper_2.syntax_and_formatting_sf,
  paper_2.under-specification_us, paper_3.* (HumanEval only)

Usage:
  python benchmark/eval_benchmark.py \
      --benchmark benchmark/LLMEval-Dataset/humaneval.json \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --outputDir ./benchmark/results \
      --gpus 0
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import random
import re
import signal
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from transformers.models.mistral3 import Mistral3Config, Mistral3ForConditionalGeneration
    AutoModelForCausalLM.register(Mistral3Config, Mistral3ForConditionalGeneration)
except (ImportError, Exception):
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark_evaluator")


# ──────────────────────────────── reproducibility ────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]


# ──────────────────────────────── generation helpers ─────────────────────────────

def build_chat_prompt(prompt: str, model_name: str, tokenizer=None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    name = model_name.lower()
    if "llama" in name:
        return f"<s>[INST] {prompt.strip()} [/INST]"
    if "deepseek" in name:
        return f"### Instruction:\n{prompt.strip()}\n### Response:\n"
    if ("starcoder" in name or "santacoder" in name) and "instruct" not in name:
        return prompt.strip()
    if "qwen" in name:
        return f"<|im_start|>user\n{prompt.strip()}\n<|im_end|>\n<|im_start|>assistant\n"
    if any(k in name for k in ("mistral", "zephyr", "phi")):
        return f"<s>[INST] {prompt.strip()} [/INST]"
    return prompt.strip()


_ASSISTANT_MARKERS = (
    "<|im_start|>assistant\n",                              # Qwen
    "<|start_header_id|>assistant<|end_header_id|>\n\n",   # Llama 3
    "[/INST]",                                              # Llama 2, Mistral, CodeLlama
    "<|assistant|>\n",                                      # Phi-3
    "<start_of_turn>model\n",                               # Gemma
    "### Response:\n",                                      # DeepSeek (legacy)
    "<|ASSISTANT|>",                                        # Falcon
)


def generate_response(
    prompt: str, generator, model_name: str, tokenizer, max_tokens: int = 512
) -> str:
    formatted = build_chat_prompt(prompt, model_name, tokenizer)
    # Use return_full_text=True and strip manually: some tokenizers cause
    # return_full_text=False to mis-slice the output.
    full = generator(
        formatted,
        max_new_tokens=max_tokens,
        do_sample=False,
        return_full_text=True,
    )[0]["generated_text"]

    # Primary strip: exact prefix match
    if full.startswith(formatted):
        return full[len(formatted):]

    # Some tokenizers prepend BOS during the pipeline's internal tokenization
    # even though apply_chat_template already embedded it as text.  Strip the
    # BOS from `full` before retrying the prefix check.
    bos = getattr(tokenizer, "bos_token", "") or ""
    full_no_bos = full[len(bos):] if bos and full.startswith(bos) else full
    if full_no_bos.startswith(formatted):
        return full_no_bos[len(formatted):]

    # Fallback: find the last assistant-turn marker and return what follows it
    for marker in _ASSISTANT_MARKERS:
        idx = full.rfind(marker)
        if idx != -1:
            return full[idx + len(marker):]

    return full


def extract_code(txt: str) -> str:
    # 1. Markdown code fence
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
    if m:
        return textwrap.dedent(m.group(1)).strip()
    # 2. [PYTHON] tags
    m = re.search(r"\[PYTHON\](.*?)\[/PYTHON\]", txt, re.DOTALL | re.IGNORECASE)
    if m:
        return textwrap.dedent(m.group(1)).strip()
    # 3. Find the first Python function/import block (model skipped the fence)
    m = re.search(
        r"^((?:(?:import|from)\s+\S[^\n]*\n)*[ \t]*def\s+\w+\s*\(.*)",
        txt,
        re.MULTILINE | re.DOTALL,
    )
    if m:
        return textwrap.dedent(m.group(1)).strip()
    return textwrap.dedent(txt).strip()


# ──────────────────────────────── test-code converters ───────────────────────────

SAFE_BUILTINS = {"sorted", "len", "sum", "min", "max", "any", "all"}


def convert_general_check_code_MBPP(test_code: str, func_name: str) -> Tuple[str, int]:
    if func_name in SAFE_BUILTINS:
        return test_code, test_code.count("assert")
    lines = [ln.rstrip() for ln in test_code.splitlines() if ln.lstrip().startswith("assert")]
    total = len(lines)
    body = ["def check(candidate):", "    passed = 0", f"    total  = {total}"]
    for ln in lines:
        ln = ln.lstrip()
        ln = re.sub(rf"\b{re.escape(func_name)}\s*\(", "candidate(", ln, count=1)
        body += [
            "    try:",
            f"        assert {ln[len('assert '):]}",
            "        passed += 1",
            "    except AssertionError:",
            "        pass",
        ]
    body.append("    return passed, total")
    return "\n".join(body), total


def convert_general_check_code_HumanEval(test_code: str, func_name: str) -> Tuple[str, int]:
    CHECK_RE = re.compile(r"^\s*def\s+check\s*\(\s*candidate\s*\)\s*:", re.M)
    if CHECK_RE.search(test_code):
        n_tests = len(re.findall(r"^\s*assert\b", test_code, re.M))
        wrapper = textwrap.dedent(f"""
            _original_check = check
            def check(candidate):
                try:
                    _original_check(candidate)
                    return {n_tests}, {n_tests}
                except AssertionError:
                    return 0, {n_tests}
        """)
        return test_code.rstrip() + "\n\n" + wrapper, n_tests
    lines = [ln.rstrip() for ln in test_code.splitlines() if ln.lstrip().startswith("assert")]
    total = len(lines)
    body = ["def check(candidate):", "    passed = 0", f"    total  = {total}"]
    for ln in lines:
        ln = re.sub(rf"\b{re.escape(func_name)}\s*\(", "candidate(", ln, count=1)
        body += [
            "    try:",
            f"        assert {ln.lstrip()[len('assert '):]}",
            "        passed += 1",
            "    except AssertionError:",
            "        pass",
        ]
    body.append("    return passed, total")
    return "\n".join(body), total


# ──────────────────────────────── sandbox execution ──────────────────────────────

class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Timeout!")


signal.signal(signal.SIGALRM, _timeout_handler)


def _safe_exec(
    candidate_code: str,
    check_code: str,
    queue: mp.Queue,
    entry_point: Optional[str] = None,
) -> None:
    try:
        env: Dict[str, Any] = {}
        exec(candidate_code, env)
        if entry_point:
            fn = env.get(entry_point)
            if not callable(fn):
                for k, v in env.items():
                    if k.lower() == entry_point.lower() and callable(v):
                        fn = v
                        break
            if not callable(fn):
                queue.put((0, f"Function `{entry_point}` not found"))
                return
            fns = [fn]
        else:
            fns = [v for v in env.values() if callable(v)]
        if not fns:
            queue.put((0, "Function not found"))
            return

        def run(fn) -> int:
            env["candidate"] = fn
            exec(check_code + "\n_result = check(candidate)", env)
            passed, _ = env["_result"]
            return passed

        queue.put((max(run(f) for f in fns), "OK"))
    except Exception as exc:
        queue.put((0, f"ERROR: {type(exc).__name__}: {exc}"))


def evaluate_with_timeout(
    candidate_code: str,
    check_code: str,
    *,
    timeout_seconds: int = 20,
    entry_point: Optional[str] = None,
) -> Tuple[int, str]:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_safe_exec, args=(candidate_code, check_code, queue, entry_point)
    )
    proc.start()
    proc.join(timeout=timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return 0, "ERROR: Timeout/Killed"
    try:
        return queue.get_nowait()
    except Exception:
        return 0, "ERROR: Unknown"


def cleanup_model(model, generator) -> None:
    del generator
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Model unloaded and memory freed.")


# ──────────────────────────── benchmark-specific helpers ─────────────────────────

def extract_mbpp_entry_point(test_cases: str) -> str:
    """Extract the function name from MBPP assert-based test cases."""
    m = re.search(r"\bassert\s+(\w+)\s*\(", test_cases)
    return m.group(1) if m else "solution"


def get_variants(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return (variant_key, prompt_text) for every non-empty mutation in a record.
    Variant keys: "original", "paper_1.incomplete", "paper_2.lexical_vagueness__lv", …
    """
    variants: List[Tuple[str, str]] = []
    orig = record.get("original", "").strip()
    if orig:
        variants.append(("original", orig))
    for paper in ("paper_1", "paper_2", "paper_3"):
        section = record.get(paper)
        if not isinstance(section, dict):
            continue
        for mut_type, text in section.items():
            if isinstance(text, str) and text.strip():
                variants.append((f"{paper}.{mut_type}", text.strip()))
    return variants


def build_generation_prompt(variant_text: str, name_for_prompt: str) -> str:
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", variant_text)
    func_name = m.group(1) if m else name_for_prompt
    func_line = (
        f"Write **one** function named `{func_name}` that solves the task."
        if func_name
        else "Write **one** Python function that solves the task."
    )
    return textwrap.dedent(f"""
        You are a senior Python developer.

        Task:
        {variant_text}

        {func_line}
        If helpers are needed, define them above the main function.

        **Use only the Python standard library and place every required `import` at the very top.**

        Return *only* valid Python code in a single code block:
        ```python
        <your code here>
        ```
    """).strip()


def evaluate_variant(
    variant_text: str,
    test_cases: str,
    original_entry: str,
    is_mbpp: bool,
    generator,
    model_name: str,
    tokenizer,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    """Generate code for one (task, variant) pair and run the test harness."""
    # Function name detected in the variant (e.g. "candidate" in paper_3 mutations)
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", variant_text)
    mutated_name: Optional[str] = m.group(1) if m else None
    name_for_prompt = mutated_name or original_entry or "solution"

    prompt = build_generation_prompt(variant_text, name_for_prompt)

    try:
        response = generate_response(prompt, generator, model_name, tokenizer, max_tokens)
        code = extract_code(response)

        # Always use the original entry point for building the check harness,
        # since test assertions reference the original function name.
        if is_mbpp:
            check_code, n_tests = convert_general_check_code_MBPP(test_cases, original_entry)
        else:
            check_code, n_tests = convert_general_check_code_HumanEval(test_cases, original_entry)

        # Try entry-point candidates in priority order
        entry_candidates: List[Optional[str]] = []
        if mutated_name:
            entry_candidates.append(mutated_name)
        if original_entry and original_entry not in entry_candidates:
            entry_candidates.append(original_entry)
        entry_candidates.append(None)  # last resort: auto-detect any callable

        passed, status = 0, "ERROR: Not evaluated"
        for ep in entry_candidates:
            passed, status = evaluate_with_timeout(
                code, check_code, timeout_seconds=timeout, entry_point=ep
            )
            if isinstance(status, str) and (
                "not found" in status or status.startswith("Function `")
            ):
                continue
            break

        pass_at_1 = passed == n_tests and status == "OK"

    except Exception as exc:
        code = response = ""
        n_tests = passed = 0
        status = f"ERROR: {type(exc).__name__}: {exc}"
        pass_at_1 = False
        logger.exception("Evaluation failed for variant")

    return {
        "GeneratedCode": code,
        "GeneratedResponse": response,
        "PromptUsed": variant_text,
        "n_Tests": n_tests,
        "Tests_Passed": passed,
        "Pass@1": pass_at_1,
        "Eval_Status": status,
    }


# ──────────────────────────────────── main loop ──────────────────────────────────

def evaluate_benchmark(args, generator, tokenizer) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = json.loads(
        Path(args.benchmark).read_text("utf-8")
    )
    if args.limit:
        records = records[: args.limit]

    is_mbpp = "mbpp" in Path(args.benchmark).stem.lower()
    dataset_label = "MBPP" if is_mbpp else "HumanEval"
    all_results: List[Dict[str, Any]] = []

    for record in tqdm(records, desc=dataset_label):
        task_id = record.get("task_id", "?")
        test_cases = record.get("test_cases", "")

        # Resolve original entry point
        original_entry = record.get("entry_point", "").strip()
        if not original_entry:
            original_entry = extract_mbpp_entry_point(test_cases)

        variants = get_variants(record)
        if args.variants:
            wanted = set(args.variants)
            variants = [(k, v) for k, v in variants if k in wanted]

        task_result: Dict[str, Any] = {"task_id": task_id, "variants": {}}

        for variant_key, variant_text in variants:
            logger.info("Task %-30s  variant=%s", task_id, variant_key)
            result = evaluate_variant(
                variant_text,
                test_cases,
                original_entry,
                is_mbpp,
                generator,
                model_name=args.model,
                tokenizer=tokenizer,
                max_tokens=args.maxNewTokens,
                timeout=args.timeout,
            )
            task_result["variants"][variant_key] = result
            logger.info(
                "  → Pass@1=%s  Tests=%d/%d  [%s]",
                result["Pass@1"],
                result["Tests_Passed"],
                result["n_Tests"],
                result["Eval_Status"],
            )

        all_results.append(task_result)

    return all_results


def compute_summary(
    results: List[Dict[str, Any]], model: str, benchmark: str
) -> pd.DataFrame:
    counts: Dict[str, Dict[str, int]] = {}
    for task in results:
        for vk, res in task["variants"].items():
            if vk not in counts:
                counts[vk] = {"total": 0, "pass1": 0, "ok": 0}
            counts[vk]["total"] += 1
            if res["Pass@1"]:
                counts[vk]["pass1"] += 1
            if res["Eval_Status"] == "OK":
                counts[vk]["ok"] += 1

    rows = []
    for vk, c in sorted(counts.items()):
        t = c["total"]
        rows.append(
            {
                "Model": model,
                "Benchmark": benchmark,
                "Variant": vk,
                "Samples": t,
                "Pass@1": round(c["pass1"] / t, 3) if t else 0.0,
                "SuccessExecRate": round(c["ok"] / t, 3) if t else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM code generation on the LLMEval benchmark dataset"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="path to humaneval.json or mbpp.json in benchmark/LLMEval-Dataset/",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument(
        "--outputDir",
        default="./benchmark/results",
        help="directory to write output files (default: ./benchmark/results)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help=(
            "subset of variants to evaluate (default: all). "
            "e.g. --variants original paper_1.incomplete paper_2.lexical_vagueness__lv"
        ),
    )
    parser.add_argument("--maxNewTokens", type=int, default=512,
                        help="max new tokens per generation (default: 512)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"],
                        help="model weight dtype (default: bfloat16)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="wall-clock timeout per problem in seconds (default: 30)")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N tasks (default: all)")
    parser.add_argument("--gpus", default=None,
                        help="CUDA_VISIBLE_DEVICES value (e.g. '0' or '0,1')")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info("CUDA_VISIBLE_DEVICES=%s", args.gpus)

    set_seed(args.seed)
    output_dir = Path(args.outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s  n_gpu=%s", device, torch.cuda.device_count())
    logger.info("Loading model: %s", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.maxNewTokens,
        do_sample=False,
    )

    results = evaluate_benchmark(args, generator, tokenizer)

    model_slug = args.model.replace("/", "_")
    dataset_slug = Path(args.benchmark).stem
    out_json = output_dir / f"{model_slug}__{dataset_slug}.json"
    out_json.write_text(json.dumps(results, indent=2), "utf-8")
    logger.info("Detailed results → %s", out_json)

    summary_df = compute_summary(results, args.model, dataset_slug)
    out_csv = output_dir / f"{model_slug}__{dataset_slug}__summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info("Summary → %s", out_csv)
    print("\n" + summary_df.to_string(index=False))

    cleanup_model(model, generator)


if __name__ == "__main__":
    main()
