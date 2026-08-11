"""Every numeric literal in the comment, beside the artifact field it should match.

Not an assertion. A report to read in one pass before publishing, because every
error in the last three rounds was a number in prose while the artifact under it
was correct. The artifacts have tests; the prose layer had nothing.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

COMMENT = Path(sys.argv[1] if len(sys.argv) > 1
               else REPO_ROOT / "report" / "upstream-comment-51187.md")
ARTIFACT = REPO_ROOT / "evidence" / "rmsnorm-blocksize.json"

doc = json.loads(ARTIFACT.read_text())
p = doc["payload"]
rows = p["threshold_table"]["rows"]
arms = p["arms"]


def row(name):
    return next(v for k, v in rows.items() if k.startswith(name))


small = [v for k, v in rows.items() if v["co_resident"] <= 16]
b31, b32 = row("batch 31"), row("batch 32")
cache = row("cache hit")
others = [v for v in small if v is not cache]

# field -> (value, where it came from)
EXPECT = {
    "1120": (p["claim"]["across_every_arm"]["pairs"], "claim.across_every_arm.pairs"),
    "112": (sum(len(r["cases"]) for r in p["lifetimes"]), "sum of case-lifetimes"),
    "16": (len(p["lifetimes"]), "len(lifetimes)"),
    "10": (arms["main"]["lifetimes"], "arms.main.lifetimes"),
    "2": (arms["max_num_seqs_8"]["lifetimes"], "arms.max_num_seqs_8.lifetimes"),
    "4": (arms["flashinfer_sampler_disabled"]["lifetimes"],
          "arms.flashinfer_sampler_disabled.lifetimes"),
    "5": (5, "repeats per lifetime (certify --repeats)"),
    "270": (b31["uncached_prefill_tokens"], "batch 31 uncached"),
    "274": (b32["uncached_prefill_tokens"], "batch 32 uncached"),
    "268": (b31["max_tokens_in_any_launch"], "batch 31 max launch"),
    "272": (b32["max_tokens_in_any_launch"], "batch 32 max launch"),
    "117": (cache["uncached_prefill_tokens"], "cache-hit uncached"),
    "147": (cache["max_tokens_in_any_launch"], "cache-hit max launch"),
    "130": (min(v["uncached_prefill_tokens"] for v in others),
            "min uncached of the other four small cases"),
    "140": (max(v["uncached_prefill_tokens"] for v in others),
            "max uncached of the other four small cases"),
    "128": (min(v["max_tokens_in_any_launch"] for v in others),
            "min max-launch of the other four small cases"),
    "136": (max(v["max_tokens_in_any_launch"] for v in others),
            "max max-launch of the other four small cases"),
    "85": (arms["max_num_seqs_8"]["max_step_tokens_observed"],
           "arms.max_num_seqs_8.max_step_tokens_observed"),
    "44": (b31["co_resident"], "batch 31 co-resident"),
    "45": (b32["co_resident"], "batch 32 co-resident"),
    "15": (cache["co_resident"], "cache-hit co-resident"),
    "8": (8, "--max-num-seqs 8, and the 8 sampler-off lifetimes across two sets"),
    "9": (round(sum(len(f) % 16 for f in __import__(
        "certify.run", fromlist=["x"]).filler_requests(13, 16, 0)) / 13, 1),
        "uncached tokens per filler sequence"),
    "4.9": (p["threshold_table"]["per_sequence_contribution"]
            ["prefix_sharing_targets"]["each"], "uncached per target sequence"),
    "256": (256, "the kernel's threshold, literal in the source"),
    "57": (57, "RMSNorm launches per forward pass at hidden 1024 (56 C++ + 1 Triton)"),
    "56": (56, "of those, the C++ fused_add_rms_norm launches"),
}

UNCHECKED = {
    "0.26.0": "version string", "0.27.0": "version string",
    "48391": "PR number", "51187": "issue number",
    "12.9": "SyaOtiLan CUDA", "13.0": "my CUDA",
    "4090": "their GPU", "4060": "my GPU",
    "4.685e-02": "their reported delta, quoted from the thread",
    "3.906e-02": "their reported delta, quoted from the thread",
    "32/32": "their operator table", "16/32": "their operator table",
    "5/5": "their nightly lifetimes", "35/35": "their nightly cases",
    "2026-08-10": "v0.27.0 release date",
    "16": "also the vec_size numerator 16/sizeof(scalar_t)",
    "512": "hidden size in their operator test", "4096": "hidden size in theirs",
    "64": "512/8 in the vec_size explanation",
    "1024": "Qwen3-0.6B hidden size and the wide block",
    "1024/8": "the hidden-1024 argument, 1024/8 = 128",
    "3": "3 of the earlier 4 sampler-off lifetimes diverged; artifact "
         "instruments.perturbation.measured_rather_than_argued."
         "the_earlier_run_by_arm.flashinfer_sampler_disabled",
    "2": "also co-residency of 2 in the hypothetical",
}

text = COMMENT.read_text()
def _isnum(x):
    try:
        float(x); return True
    except ValueError:
        return False


# Version strings tokenize into meaningless fragments; drop them first.
text_for_scan = re.sub(r"\d+\.\d+\.\d+", " ", text)
literals = re.findall(r"\d+\.\d+e-\d+|\d+\.\d+|\d+/\d+|\d{4}-\d{2}-\d{2}|\d+", text_for_scan)
seen, ok, mismatch, unknown = [], [], [], []
for lit in dict.fromkeys(literals):
    if lit in EXPECT:
        value, where = EXPECT[lit]
        same = str(value) == lit or (
            _isnum(lit) and _isnum(str(value)) and float(lit) == float(value))
        (ok if same else mismatch).append((lit, value, where))
    elif lit in UNCHECKED:
        seen.append((lit, UNCHECKED[lit]))
    else:
        unknown.append(lit)

print(f"comment: {COMMENT}\nartifact: {ARTIFACT.name}\n")
print(f"MATCHES ARTIFACT ({len(ok)})")
for lit, value, where in ok:
    print(f"  {lit:<10} == {where}")
print(f"\nMISMATCH ({len(mismatch)})")
for lit, value, where in mismatch:
    print(f"  {lit:<10} != {value}   ({where})")
print(f"\nNOT AN ARTIFACT FIELD, quoted from elsewhere ({len(seen)})")
for lit, why in seen:
    print(f"  {lit:<10} {why}")
print(f"\nNO CORRESPONDING FIELD, check by hand ({len(unknown)})")
for lit in unknown:
    for line in text.splitlines():
        if re.search(rf"(?<!\d){re.escape(lit)}(?!\d)", line):
            print(f"  {lit:<10} {line.strip()[:88]}")
            break
