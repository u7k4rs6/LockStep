# Thin wrappers over the lockstep CLI and module entry points. Nothing here
# reimplements anything; if a target and the CLI disagree, the CLI is right.
.PHONY: help setup check claim1 claim2 claim3 report replay certify clean

help:
	@echo "setup    install the pinned environment and fetch weights"
	@echo "check    everything that needs no GPU (what CI runs)"
	@echo "claim1   invariance, 13 relations across 5 block sizes      [GPU]"
	@echo "claim2   mutation campaign, 10 operators over 192 cases     [GPU]"
	@echo "claim3   throughput, interleaved, both graph modes          [GPU + vLLM]"
	@echo "report   build results/report.html from committed evidence"
	@echo "replay   re-run the committed replay-determinism witness    [GPU]"
	@echo "certify  black-box differential test of a local vLLM        [GPU + vLLM]"

setup:
	uv sync
	uv run python3 scripts/download_weights.py
	OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python3 scripts/build_fp64_reference.py

# The CPU-only gate. Identical to what .github/workflows/ci.yml runs.
check:
	uv run python3 scripts/static_checks.py
	uv run python3 scripts/secret_scan.py
	uv run python3 scripts/verify_no_gpu.py
	uv run python3 -m pytest -m "not gpu" -q

claim1:
	uv run ./lockstep verify --block-sizes 8 16 32 64 128

claim2:
	uv run ./lockstep fuzz --seeds 12 --cases-per-seed 6 --eviction-campaign 120 --seeded-faults

claim3:
	uv run ./lockstep bench --external-python $$HOME/lockstep-extenv/vllmdet/bin/python --runs 5

report:
	uv run ./lockstep report

replay:
	uv run ./lockstep replay evidence/case-witness.json

certify:
	uv run ./lockstep certify --repeats 3

clean:
	rm -rf results/*/ results/report.html
