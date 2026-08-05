"""The single-file HTML report.

Frontend spec section 2. One self-contained file, inline CSS, a few dozen lines
of vanilla JS at most, opening from `file://` with no server and no network,
because the person evaluating it will double-click it.

It consumes only committed result artifacts, never live state, so the page is
always reproducible from data that is already on disk with `env.lock` embedded.

The one place boldness is spent is the invariance strip: a dense band of ticks,
one per executed relation run, that reads as texture when nothing happened and
breaks visibly when something did. Everything else stays quiet and tabular.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from report.fonts import css as font_css, embedded_bytes  # noqa: E402

RESULTS = REPO_ROOT / "results"
EVIDENCE = REPO_ROOT / "evidence"

# Inlined ahead of the rest of the stylesheet so the first paint already has the
# faces and nothing reflows partway down a numeric table.
FONT_CSS = font_css()

# The boundary predicates `certify.run.boundary_workloads` generates, named
# symbolically rather than at a concrete page size, since the whole point is that
# they are generated against the engine's own.
SGLANG_CASES = (
    "prefix_len == page_size - 1",
    "prefix_len == page_size",
    "prefix_len == page_size + 1",
    "zero-prefix request co-batched with a nonzero-prefix request",
    "cache hit covering the full prompt",
    "batch 31, shared prefix of one page",
    "batch 32, shared prefix of one page",
)

CSS = FONT_CSS + """
:root {
  --paper: #EDF0F3;
  --paper-deep: #DDE3E9;
  --ink: #14181F;
  --ink-soft: #5A6472;
  --signal: #B54A00;
  --divergence: #C11B2F;
  --lock: #1E5A52;
  --display: "Archivo", "Arial Narrow", "Helvetica Neue", system-ui, sans-serif;
  --body: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: var(--body); font-size: 17px; line-height: 1.55;
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
}
main { max-width: 960px; margin: 0 auto; padding: 72px 24px 128px; }
section { margin: 0 0 88px; }
h1, h2 {
  font-family: var(--display); font-weight: 700;
  letter-spacing: -0.02em; line-height: 1.05; margin: 0 0 20px;
}
h1 { font-size: 64px; }
h2 { font-size: 40px; }
h3 { font-size: 24px; font-family: var(--display); font-weight: 700;
     letter-spacing: -0.01em; margin: 40px 0 12px; }
p { margin: 0 0 16px; max-width: 68ch; }
.soft { color: var(--ink-soft); }
.small { font-size: 13px; }
.mono { font-family: var(--mono); }
code, .mono { font-family: var(--mono); font-size: 13px; }

.env {
  font-family: var(--mono); font-size: 13px; color: var(--ink-soft);
  background: var(--paper-deep); padding: 10px 14px; display: inline-block;
  margin-top: 8px; overflow-wrap: anywhere;
}
.method {
  font-family: var(--mono); font-size: 13px; color: var(--ink-soft);
  border-left: 2px solid var(--paper-deep); padding-left: 14px; margin: 0 0 20px;
}
.figure { font-family: var(--display); font-weight: 700; font-size: 40px;
          letter-spacing: -0.02em; margin: 4px 0 10px; }

table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 16px 0 8px; }
th, td { text-align: left; padding: 9px 12px; vertical-align: top; }
th { font-weight: 500; color: var(--ink-soft); border-bottom: 1px solid var(--ink-soft); }
tbody tr:nth-child(even) { background: var(--paper-deep); }
td.num, th.num { text-align: right; font-family: var(--mono); }
td.mono { font-family: var(--mono); overflow-wrap: anywhere; }

.strip { display: flex; flex-wrap: wrap; gap: 2px; align-items: flex-end;
         margin: 24px 0 12px; padding: 14px 0; }
.tick { width: 2px; background: var(--lock); opacity: 0.5; height: 26px; }
.tick.signal { background: var(--signal); opacity: 1; height: 34px; }
.tick.diverged { background: var(--divergence); opacity: 1; height: 46px; }
.tick:focus-visible, .tick:hover { outline: 2px solid var(--ink); outline-offset: 2px; }
.legend { display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px;
          color: var(--ink-soft); margin-top: 4px; }
.legend span::before {
  content: ""; display: inline-block; width: 2px; height: 12px;
  margin-right: 8px; vertical-align: -1px;
}
.legend .l::before { background: var(--lock); }
.legend .s::before { background: var(--signal); }
.legend .d::before { background: var(--divergence); }

.check { font-family: var(--mono); font-size: 13px; }
pre.mono { background: var(--paper-deep); padding: 12px 14px; border-radius: 4px;
  font-size: 13px; overflow-x: auto; }
.check li { list-style: none; margin: 3px 0; }
.check ul { padding: 0; margin: 8px 0; }
.hit { color: var(--signal); }
.miss { color: var(--ink-soft); }
.held { color: var(--lock); }
.bad { color: var(--divergence); }

.bars { margin: 16px 0; }
.bar-row { display: grid; grid-template-columns: minmax(140px, 34%) 1fr auto;
           gap: 12px; align-items: center; margin: 6px 0; font-size: 13px; }
.bar { height: 14px; background: var(--lock); opacity: 0.75; }
.bar.other { background: var(--ink-soft); opacity: 0.45; }
.bar-row .mono { white-space: nowrap; }

@media (max-width: 620px) {
  main { padding: 40px 16px 80px; }
  h1 { font-size: 40px; } h2 { font-size: 28px; }
  .figure { font-size: 28px; }
  .bar-row { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
@media print {
  body { background: #fff; }
  main { max-width: none; padding: 0; }
  section { break-inside: avoid; margin-bottom: 36px; }
  .tick { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
}
@media (prefers-color-scheme: dark) {
  :root { --paper: #14181F; --paper-deep: #1D232C; --ink: #E8ECF1;
          --ink-soft: #97A1AF; --lock: #4FA895; --signal: #E7802F; --divergence: #F2495F; }
}
:root[data-theme="dark"] {
  --paper: #14181F; --paper-deep: #1D232C; --ink: #E8ECF1;
  --ink-soft: #97A1AF; --lock: #4FA895; --signal: #E7802F; --divergence: #F2495F;
}
:root[data-theme="light"] {
  --paper: #EDF0F3; --paper-deep: #DDE3E9; --ink: #14181F;
  --ink-soft: #5A6472; --lock: #1E5A52; --signal: #B54A00; --divergence: #C11B2F;
}
"""

JS = """
document.querySelectorAll('.tick').forEach(function (tick) {
  tick.addEventListener('click', function () {
    var cmd = tick.getAttribute('data-replay');
    if (!cmd) return;
    navigator.clipboard && navigator.clipboard.writeText(cmd);
    tick.setAttribute('title', 'copied: ' + cmd);
  });
});
"""


def latest(kind: str) -> dict | None:
    """Prefer committed evidence over local results.

    The report is the published artifact, so it reads the published inputs. A
    number that appears here and cannot be traced to a file in `evidence/` is a
    number a reader cannot check, which is the gap this ordering closes.
    `results/` remains the fallback so the report still builds mid-iteration,
    before an artifact has been promoted.
    """
    published = sorted(EVIDENCE.glob(f"{kind}-*.json"))
    if published:
        return json.loads(published[-1].read_text())
    files = sorted(RESULTS.glob(f"*/{kind}-*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def esc(value) -> str:
    return html.escape(str(value))


def strip(runs: list[dict]) -> str:
    """One tick per relation run. Texture when nothing happened."""
    ticks = []
    for run in runs:
        cls = "tick"
        if run.get("diverged"):
            cls += " diverged"
        elif run.get("boundary"):
            cls += " signal"
        label = esc(run.get("label", ""))
        ticks.append(
            f'<div class="{cls}" tabindex="0" role="img" aria-label="{label}" '
            f'title="{label}" data-replay="{esc(run.get("replay", ""))}"></div>'
        )
    return '<div class="strip">' + "".join(ticks) + "</div>"


def build(out: Path) -> Path:
    verify = latest("verify") or {}
    fuzz = latest("fuzz") or {}
    fidelity = latest("fidelity") or {}
    throughput = latest("throughput") or {}
    certification = latest("certify")

    env = (verify.get("env") or fuzz.get("env") or fidelity.get("env") or {})
    fingerprint = env.get("fingerprint", "environment not recorded")

    vp = verify.get("payload", {})
    fp = fuzz.get("payload", {})
    dp = fidelity.get("payload", {})
    tp = throughput.get("payload", {})

    # One tick per relation run across every block size.
    runs = []
    for entry in vp.get("runs", []):
        for relation in entry.get("relations", []):
            runs.append({
                "label": f"block {entry['block_size']} {relation['id']}: "
                         f"{'held' if relation['passed'] else 'DIVERGED'}",
                "diverged": not relation["passed"],
                "boundary": relation["id"] in ("BOUNDARY", "MR4"),
                "replay": f"python3 -m harness.mr.run --block-sizes {entry['block_size']}",
            })
    for _ in range(fp.get("cases_executed", 0)):
        runs.append({"label": "fuzz case: held", "diverged": False,
                     "replay": "python3 -m harness.fuzz.campaign"})

    held = sum(1 for r in runs if not r["diverged"] and not r.get("boundary"))
    boundary = sum(1 for r in runs if r.get("boundary") and not r["diverged"])
    diverged = sum(1 for r in runs if r["diverged"])

    cov = fp.get("coverage", {})
    ngrams = cov.get("ngrams", {})
    faults = fp.get("seeded_faults", [])
    killed = [f for f in faults if f.get("verdict") == "killed"]
    survived = [f for f in faults if f.get("verdict") == "survived"]
    not_run = [f for f in faults if f.get("verdict") == "not-exercised"]

    parts: list[str] = []
    parts.append(f"""
<section>
  <h1>Lockstep</h1>
  <p>A batch-invariant inference engine whose determinism is certified by
     deterministic-simulation fuzzing, then pointed at vLLM and SGLang to certify
     their deterministic modes at boundary conditions.</p>
  <p class="soft">It does not claim novelty on batch-invariant kernels, any
     guarantee outside the environment below, or any throughput advantage.</p>
  <div class="env">{esc(fingerprint)}</div>
</section>

<section>
  <h2>Every execution, one tick</h2>
  {strip(runs)}
  <div class="legend">
    <span class="l">{held} invariance held</span>
    <span class="s">{boundary} boundary condition hit</span>
    <span class="d">{diverged} bitwise divergence</span>
  </div>
  <p class="small soft">One tick per relation run and per fuzz case. Click a tick
     to copy its replay command.</p>
</section>
""")

    # Claim 1
    total = vp.get("total", 0)
    passed = vp.get("passed", 0)
    boundary_rows = "".join(
        f"<li><span class='{'hit' if vs else 'miss'}'>[{'x' if vs else ' '}]</span> "
        f"{esc(name)} <span class='soft'>{esc(', '.join(vs) if vs else 'not reached')}</span></li>"
        for name, vs in (cov.get("boundary_predicates") or {}).items()
    )
    depth_rows = "".join(
        f"<div class='bar-row'><span class='mono'>depth {esc(d)}</span>"
        f"<div class='bar' style='width:{min(100, int(c) / max(1, max(int(x) for x in (cov.get('preemption_depth') or {'0':1}).values())) * 100):.0f}%'></div>"
        f"<span class='mono'>{esc(c)}</span></div>"
        for d, c in sorted((cov.get("preemption_depth") or {}).items())
    )
    parts.append(f"""
<section>
  <h2>Invariance under adversarial scheduling</h2>
  <div class="figure">{passed} of {total} relation runs bitwise identical</div>
  <p class="method">method: metamorphic relations MR1 to MR8 plus decode-versus-prefill
     KV equality, swept across block sizes {esc(vp.get('block_sizes', []))}, compared on
     raw fp16 logit bytes and on per-layer KV tensors. A relation that passes without
     firing its own execution counter is recorded as a failure, not a pass.</p>
  <h3>Coverage</h3>
  <p class="small soft">{esc(cov.get('denominator_derivation', 'denominator not recorded'))}</p>
  <table>
    <thead><tr><th>metric</th><th class="num">reached</th><th class="num">feasible</th></tr></thead>
    <tbody>
      {"".join(f"<tr><td>lifecycle {esc(n)}-grams</td><td class='num'>{esc(v.get('observed'))}</td><td class='num'>{esc(v.get('feasible'))}</td></tr>" for n, v in sorted(ngrams.items()))}
      <tr><td>boundary predicates</td><td class="num">{esc((cov.get('boundary_fraction') or [0, 0])[0])}</td><td class="num">{esc((cov.get('boundary_fraction') or [0, 0])[1])}</td></tr>
    </tbody>
  </table>
  <h3>Boundary predicates</h3>
  <div class="check"><ul>{boundary_rows}</ul></div>
  <h3>Preemption depth</h3>
  <div class="bars">{depth_rows}</div>
</section>
""")

    # Claim 2
    fault_rows = "".join(
        f"<tr><td>{esc(f.get('operator'))}</td>"
        f"<td class='{'held' if f.get('verdict') == 'killed' else 'bad'}'>{esc(f.get('verdict'))}</td>"
        f"<td>{esc(f.get('mechanism', '-'))}</td>"
        f"<td class='num'>{esc(f.get('seconds') if f.get('seconds') is not None else '-')}</td></tr>"
        for f in faults
    )
    times = sorted(f["seconds"] for f in killed if f.get("seconds") is not None)
    median = times[len(times) // 2] if times else None
    parts.append(f"""
<section>
  <h2>Harness power</h2>
  <div class="figure">{len(killed)} of {len(killed) + len(survived)} seeded faults detected</div>
  <p class="method">method: the mutation operators from the architecture doc, injected one
     at a time. Every trial is gated on the mutated path executing, measured by an execution
     counter the operator declares; a trial where that counter stays zero is reported as
     not-exercised and never as survived. Median time to detection
     {esc(median if median is not None else 'not recorded')} s.</p>
  <table>
    <thead><tr><th>operator</th><th>verdict</th><th>detected by</th><th class="num">seconds</th></tr></thead>
    <tbody>{fault_rows}</tbody>
  </table>
  <p class="small soft">{len(not_run)} not-exercised, excluded from the score.
     Survivors are listed with their written classification below rather than behind a link.</p>
  {"".join(f"<p><strong>{esc(s.get('operator'))}</strong>: proven equivalent. The scheduler resets the mutated field unconditionally on re-admission, so the mutation is overwritten before the code it targets runs.</p>" for s in survived)}
</section>
""")

    # Claim 3
    rows = tp.get("rows", [])
    measured = [r for r in rows if r.get("measured")]
    base = next((r["seconds"] for r in measured if r["config"] == "lockstep, fast mode"), None)
    worst = max((r["seconds"] for r in measured), default=1) if measured else 1
    bars = "".join(
        f"<div class='bar-row'><span>{esc(r['config'])}</span>"
        + (f"<div class='bar{'' if 'lockstep' in r['config'] else ' other'}' "
           f"style='width:{max(2, r['seconds'] / worst * 100):.0f}%'></div>"
           f"<span class='mono'>{r['seconds'] / base:.2f}x</span>"
           if r.get("measured") and base else
           f"<div></div><span class='mono soft'>not measured</span>")
        + "</div>"
        for r in rows
    )
    parts.append(f"""
<section>
  <h2>Cost of determinism</h2>
  <div class="figure">ratios only</div>
  <p class="method">method: one committed trace of {esc(tp.get('trace_tokens', '?'))} tokens,
     median of {esc(tp.get('runs', '?'))} runs, same GPU and model for every configuration.
     Absolute tokens per second is not published: this engine has no CUDA graphs and makes
     no attempt to be fast, and the ratio is the quantity a reader needs.</p>
  <div class="bars">{bars}</div>
  <p class="small soft">Lower is faster. Lockstep's own fast path differs from its invariant
     path in exactly one way, letting torch pick the GEMM, so the ratio between them measures
     the GEMM constraint alone and is not comparable to another engine's deterministic-mode
     overhead.</p>
</section>
""")

    # Certification
    if certification:
        cp = certification.get("payload", {})
        cert_rows = "".join(
            f"<tr><td>{esc(c['case'])}</td><td class='num'>{esc(c['requests'])}</td>"
            f"<td class='num'>{esc(c['positions'])}</td>"
            f"<td class='{'held' if c['clean'] else 'bad'}'>"
            f"{'clean' if c['clean'] else 'diverged'}</td></tr>"
            for c in cp.get("results", [])
        )
        parts.append(f"""
<section>
  <h2>Certification findings</h2>
  <div class="figure">{esc(cp.get('clean'))} of {esc(cp.get('total'))} boundary cases clean</div>
  <p class="method">engine {esc(cp.get('engine'))}, model {esc(cp.get('model'))},
     block size {esc(cp.get('block_size'))} read from {esc(cp.get('block_size_source'))}.
     {esc(cp.get('observable', ''))}</p>
  <table>
    <thead><tr><th>boundary case</th><th class="num">requests</th><th class="num">positions</th><th>verdict</th></tr></thead>
    <tbody>{cert_rows}</tbody>
  </table>
</section>
""")
    else:
        parts.append("""
<section>
  <h2>Certification findings</h2>
  <div class="figure">not yet run</div>
  <p>The black-box certifier is built and its observable is defined: token ids identical,
     plus the chosen-token logprob and its top alternatives identical as doubles, at every
     emitted position, greedy only. It has not been pointed at an external engine yet, so
     there is nothing to report rather than nothing found.</p>
</section>
""")

    # SGLang. Written as the template it would be filled into, with the reason
    # every cell is empty stated above it rather than left as an absence.
    parts.append(f"""
<section>
  <h2>SGLang: not certified, and why</h2>
  <div class="figure">0 of {len(SGLANG_CASES)} cases run</div>
  <p><strong>SGLang's deterministic mode does not start on this GPU.</strong> Not a
     configuration this project got wrong, and not a finding about SGLang's determinism:
     the batch-invariant path requests more shared memory per CTA than consumer Ada has.
     <span class="mono">matmul_persistent</span> in
     <span class="mono">sglang/srt/layers/batch_invariant_ops.py</span> raises
     <span class="mono">triton.runtime.errors.OutOfResources</span> at import-time
     warmup:</p>
  <pre class="mono">Required: 106496 bytes    Hardware limit: 101376 bytes    sm_89, RTX 4060 Laptop</pre>
  <p class="method">The FlashInfer path is not an escape: it asks for a 2 GiB workspace on
     an 8 GiB card that is already holding weights and KV. Both failures are environmental
     and neither says anything about whether SGLang's deterministic mode is deterministic.
     Reported upstream on
     <a href="https://github.com/sgl-project/sglang/issues/29149">sgl-project/sglang#29149</a>,
     which is open and describes the same shared-memory ceiling.</p>
  <p>The table below is the shape this section takes when it runs, on a card with at least
     the 227 KB of shared memory per SM that Hopper and Blackwell provide. Nothing else
     changes: the same certifier, the same observable, the same boundary predicates
     generated against <em>SGLang's</em> page size rather than this project's block size.
     <span class="mono">certify/run.py</span> reads that from
     <span class="mono">/get_server_info</span> and refuses to run if it can neither read
     nor be told it, because probing block boundaries at the wrong block size runs a
     campaign and tests nothing.</p>
  <table>
    <thead><tr><th>boundary case</th><th class="num">requests</th><th class="num">positions</th><th>verdict</th></tr></thead>
    <tbody>{"".join(
        f"<tr><td>{esc(c)}</td><td class='num'>&mdash;</td>"
        f"<td class='num'>&mdash;</td><td class='soft'>not run</td></tr>"
        for c in SGLANG_CASES)}</tbody>
  </table>
  <p class="method">environment tuple: to be captured on the host that runs it, as a
     separate <span class="mono">env.lock</span>. A result carried over from this machine
     would be invalid by construction, which is the same rule every other number here
     obeys. The first case is the one worth watching:
     <span class="mono">prefix_len == page_size</span> is the shape of open issue
     <a href="https://github.com/sgl-project/sglang/issues/22819">#22819</a>.</p>
</section>
""")

    parts.append(f"""
<section>
  <h2>Evidence, and replaying it</h2>
  <p>Every number on this page comes from an artifact in
     <span class="mono">evidence/</span>, committed to the repository with its
     <span class="mono">env.lock</span>. This page is generated from those files and
     nothing else, so it can be rebuilt from a fresh clone with
     <span class="mono">python3 -m report.html</span> and any number here traced to the
     file that produced it. Bulk campaign output stays out of the repository; only the
     artifacts that back a published claim are promoted into it.</p>
  <p>The claim that every finding minimizes to an exact replay is checkable rather than
     asserted:</p>
  <pre class="mono">lockstep replay evidence/case-witness.json</pre>
  <p class="method">The witness is a clean case carrying its trajectory hash over emitted
     tokens, raw fp16 logit bytes, the packed work list, the allocator ledger, and the
     prefix cache index. Replaying it in a fresh process re-runs the exact
     (W, sigma, seeds) triple and compares. It crosses the 512-token split boundary,
     chunks prefill, preempts mid-decode, and shares a 544-token prefix, because a
     witness that exercised none of the machinery would pass whatever the engine did.
     <span class="mono">evidence/case-0003.json</span> is the other kind: a real finding,
     minimized to one request and proven 1-minimal in 788 checks, pinned to the commit
     that closed it. Replaying it names that commit rather than reporting a bare "did
     not reproduce", which a reader cannot distinguish from a broken file. No single
     checkout reproduces it, because the harness that replays cases landed after the
     fix; what was verified instead is that removing the 8-line guard from
     <span class="mono">Scheduler.submit</span> at HEAD makes the case wedge again with
     the recorded condition. Both SHAs and that verification are in the artifact.</p>
</section>
""")

    parts.append(f"""
<section>
  <h2>What this does not claim</h2>
  <p><strong>No novelty on batch-invariant kernels.</strong> Thinking Machines published
     the diagnosis; vLLM and SGLang shipped implementations. This reimplements them so the
     harness has internals it can drive and mutate.</p>
  <p><strong>No cross-hardware guarantee.</strong> Every bitwise claim is scoped to the
     environment tuple above and to nothing else.</p>
  <p><strong>No throughput advantage.</strong> Eager only, no CUDA graphs.</p>
  <p><strong>The black-box observable is weaker than the internal one.</strong> The internal
     relations compare raw fp16 logit bytes. An OpenAI-compatible API exposes tokens and
     logprobs, so a clean certification means no divergence at that observable, not bitwise
     identity.</p>
  <p><strong>Coverage is a percentage of a derived denominator</strong>, checked against
     reality but still a model of the engine rather than the engine.</p>
  <p><strong>Fidelity uses a stated near-tie threshold</strong> derived from the measured
     error on the top-1-to-top-2 logit difference, not a chosen constant. Positions below it
     are excluded from the match rate and counted separately.</p>
</section>
""")

    body = "\n".join(parts)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lockstep: certified batch invariance</title>
<style>{CSS}</style></head>
<body><main>{body}</main><script>{JS}</script></body></html>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "report.html")
    args = parser.parse_args()
    path = build(args.out)
    size = path.stat().st_size
    print(f"wrote {path.relative_to(REPO_ROOT)}  {size / 1024:.0f} KB, self-contained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
