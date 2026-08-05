"""Run one external engine configuration and print its wall time as JSON.

Executed by bench/throughput.py in a separate interpreter, because vLLM and
SGLang pin torch versions that would fight the locked environment this project's
claims are scoped to. Keeping them out of process keeps env.lock meaningful.
"""

from __future__ import annotations

import json
import os
import sys
import time


def run_vllm(spec: dict) -> float:
    if spec["deterministic"]:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=spec["weights"],
        dtype="float16",
        gpu_memory_utilization=0.55,
        max_model_len=512,
        enforce_eager=True,
        disable_log_stats=True,
    )
    prompts = [{"prompt_token_ids": tokens} for tokens, _ in spec["trace"]]
    params = [SamplingParams(temperature=0.0, max_tokens=n) for _, n in spec["trace"]]

    started = time.perf_counter()
    llm.generate(prompts, params)
    return time.perf_counter() - started


def run_sglang(spec: dict) -> float:
    import sglang as sgl

    engine = sgl.Engine(
        model_path=spec["weights"],
        dtype="float16",
        mem_fraction_static=0.55,
        enable_deterministic_inference=spec["deterministic"],
        disable_cuda_graph=True,
    )
    started = time.perf_counter()
    engine.generate(
        input_ids=[tokens for tokens, _ in spec["trace"]],
        sampling_params=[
            {"temperature": 0.0, "max_new_tokens": n} for _, n in spec["trace"]
        ],
    )
    elapsed = time.perf_counter() - started
    engine.shutdown()
    return elapsed


def main() -> int:
    spec = json.loads(sys.argv[1])
    seconds = run_vllm(spec) if spec["engine"] == "vllm" else run_sglang(spec)
    print(json.dumps({"seconds": seconds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
