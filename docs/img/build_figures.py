#!/usr/bin/env python3
"""
Regenerate every figure in docs/img/.

    python3 docs/img/build_figures.py

DESIGN
------
The figures are drawn as instrument panels, because that is what this project
is: a bench that measures whether two executions stayed in step. Every figure
carries the same chrome, and each piece of it encodes something rather than
decorating.

    tick scale      down the left edge, because everything here is a
                    measurement against a declared scale
    trace glyph     top right, two traces either coincident or forked. It says
                    at a glance whether the figure is about agreement or
                    divergence
    hatch           declared but never reached. Not empty, not zero: unvisited
    source line     the artifact each figure was drawn from, so a figure is a
                    pointer to evidence rather than a claim of its own

PALETTE, named after the project, built from four given values
    2C2C2C  ground, dark theme        F3F4F4  ground, light theme
    612D53  step, plum                853953  drift, rose

    step    in step: identical, holds, killed, reached, found from inside
    drift   out of step: divergence, defect, retracted, found from outside
    slate   neither, or withdrawn
    grid    the track everything is measured against
    ink     the frame

The two given accents sit 24 degrees apart in hue, both dark magenta, which is
not enough separation to tell two states apart. They are stretched to roughly
50 degrees for use as marks (step toward violet, drift toward warm pink) while
612D53 and 853953 themselves stay as the deep variants, step_dim and drift_dim.
Even so, colour is NOT carrying the encoding on its own and must not be asked
to. Everywhere step and drift
appear together there is a redundant non-colour cue: tick against cross, solid
fill against outline, hatch for never-reached, and lane position in the thesis
panel. On dark, both accents are used as lightened tints of the given hex so
they clear contrast on 2C2C2C; the given values themselves are the solid fills
in the light theme.

NUMBERS: the N dict below is the only place figures take values from, and it
duplicates values that also live in README.md and evidence/. That is a third
copy, which is exactly row 16. Wire tests/test_figure_constants.py to assert N
against evidence/index.json before trusting a figure over an artifact.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent

W = 840
MX = 22
LX = MX + 30
RX = W - MX - 14
CW = RX - LX
TOP = 116

MONO = "ui-monospace, SFMono-Regular, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace"
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

DARK = dict(
    ground="#2C2C2C", panel="#333333", panel_alt="#292929",
    grid="#3D3D3D", grid_hi="#4F4F4F", edge="#454545",
    ink="#F3F4F4", muted="#A9AAAA", faint="#7C7D7D",
    step="#B474B4", step_hi="#D0A3D0", step_dim="#612D53", step_wash="#382C3A",
    drift="#E87D8F", drift_hi="#F2ABB6", drift_dim="#853953", drift_wash="#4A2F35",
    slate="#8A8A8A", slate_wash="#383838",
    glow="1.6",
)

LIGHT = dict(
    ground="#F3F4F4", panel="#FFFFFF", panel_alt="#EAEBEB",
    grid="#DEDFDF", grid_hi="#BFC0C0", edge="#D6D7D7",
    ink="#2C2C2C", muted="#6A6B6B", faint="#949595",
    step="#6C2D6C", step_hi="#9A4C9A", step_dim="#612D53", step_wash="#F0E9F0",
    drift="#93304A", drift_hi="#B85070", drift_dim="#853953", drift_wash="#FAECEF",
    slate="#7A7B7B", slate_wash="#E4E5E5",
    glow="0.01",
)

# ---------------------------------------------------------------- numbers ---

N = dict(
    claims=[
        ("I1", "Batch invariance", "bit-identical to canonical, whatever cohabits",
         "MR1, MR5 · fp16 logit bytes · batch 1 to 32", "holds"),
        ("I2", "Schedule invariance", "any preemption, chunk partition, eviction, cache hit",
         "MR2, MR3, MR4 · per-layer KV equality", "holds"),
        ("I3", "Replay determinism", "same (W, sigma, seeds), same trajectory hash",
         "MR6 · 8 shapes · cross-process, differing PYTHONHASHSEED", "holds"),
        ("I4", "RNG isolation", "tokens from (seed, uid, position) and own logits only",
         "MR7 · 11 perturbations of the cohabitant set", "holds"),
        ("F1", "Fidelity", "batch-1 logits against an fp64 CPU reference",
         "exact KL over 151936 tokens at 2756 positions", "7 of 7"),
    ],
    three=[
        ("INVARIANCE", "65", "65", "relation runs, bitwise identical",
         ["13 relations across 5 block sizes", "MR1 to MR8, PATH-EQ, EOS finish"], 1.0, "step"),
        ("HARNESS POWER", "10", "10", "seeded faults killed",
         ["0 equivalent, 0 not-exercised", "median time to detection 10.6 s"], 1.0, "step"),
        ("COST", "5.2", "5.7", "x vLLM batch-invariant, eager",
         ["1.05x to 1.10x vs its own fast path", "two runs, reported as a range"], None, "drift"),
    ],
    throughput=[
        ("lockstep, fast mode", 2.930, 2.823, 1.24, "slate"),
        ("lockstep, invariant", 3.226, 2.957, 1.17, "step"),
        ("vLLM default, eager", 0.470, 0.490, 1.27, "slate"),
        ("vLLM batch-invariant, eager", 0.563, 0.574, 1.15, "step"),
        ("vLLM default, CUDA graphs", 0.343, 0.341, 1.08, "slate"),
        ("vLLM batch-invariant, graphs", 0.484, 0.475, 1.27, "step"),
    ],
    kill=dict(threshold=15.0, span=20.0,
              marks=[(14.6, "drift", True), (16.6, "drift", False),
                     (10.6, "step", True), (11.5, "step", False)]),
    coverage=dict(two_total=25, two_swarm=15, two_evict=4,
                  three_total=79, three_swarm=28, three_evict=15,
                  probes_two=20, probes_three=46),
    mutation=dict(n=10, killed_bare=9,
                  survivor="reversed split-combine fold"),
    thesis=[("1", "in"), ("2", "in"), ("3", "in"), ("4", "in"), ("5", "in"),
            ("6", "in"), ("7", "in"), ("8", "in"),
            ("9", "audit"), ("10", "audit"), ("11", "audit"), ("12", "audit"),
            ("12b", "audit"), ("13", "anomaly"), ("14", "reader"), ("15", "reader"),
            ("16", "late"), ("17", "late")],
    lanes=[("in", "own machinery", "a run contradicting a declaration", 8),
           ("audit", "outside audit", "commissioned, looking for defects", 5),
           ("anomaly", "an anomaly", "predicted by neither side", 1),
           ("reader", "readers", "not looking for defects at all", 2),
           ("late", "self-audit, later", "once it knew where to look", 2)],
    certification=[("prefix_len == block_size - 1", 15, None),
                   ("prefix_len == block_size", 15, None),
                   ("prefix_len == block_size + 1", 15, None),
                   ("zero-prefix co-batched with nonzero", 16, None),
                   ("cache hit covering the full prompt", 15, None),
                   ("batch 31, shared prefix of one block", 44, "4.685e-02"),
                   ("batch 32, shared prefix of one block", 45, "3.906e-02")],
    control=[("--max-num-seqs 8", 8, "1 of 5, reproducible", True),
             ("--max-num-seqs 64", 45, "not reproducible", False),
             ("--max-num-seqs 128, the default", 45, "not reproducible, intermittently", False)],
)

SOURCE = {
    "claims": "evidence/verify-0002.json  ::  evidence/fidelity-0001.json",
    "three-numbers": "evidence/verify-0002.json  ::  fuzz-0002.json  ::  throughput-0004.json",
    "throughput": "evidence/throughput-0004.json  ::  docs/kickoff/01-PRD.md s11",
    "coverage": "evidence/fuzz-0002.json  ::  scripts/verify_no_gpu.py",
    "mutation": "evidence/fuzz-0002.json",
    "thesis": "README.md  ::  tests/test_thesis_table.py",
    "certification": "evidence/certify-pairs-{a,b,mns8}.json  ::  vllm-project/vllm#51187",
    "hero": "evidence/  ::  vllm-project/vllm#51187",
    "architecture": "docs/kickoff/02-technical-architecture.md",
    "observers": "harness/mr/  ::  harness/fuzz/golden.py",
    "limits": "README.md  ::  docs/kickoff/01-PRD.md",
    "prior-art": "docs/kickoff/02-technical-architecture.md s2",
    "evidence": "evidence/index.json",
    "gates": ".github/workflows/ci.yml  ::  Makefile",
}

# --------------------------------------------------------------- plumbing ---


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=12, fill="ink", fam=SANS, weight=400, anchor="start",
        op=None, ls=None, P=None):
    a = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'font-family="{fam}"',
         f'font-size="{size}"', f'fill="{P[fill]}"']
    if weight != 400:
        a.append(f'font-weight="{weight}"')
    if anchor != "start":
        a.append(f'text-anchor="{anchor}"')
    if op is not None:
        a.append(f'opacity="{op}"')
    if ls is not None:
        a.append(f'letter-spacing="{ls}"')
    return f'<text {" ".join(a)}>{esc(s)}</text>'


def mono(x, y, s, size=11, fill="muted", weight=400, anchor="start", ls=None,
         op=None, P=None):
    return txt(x, y, s, size, fill, MONO, weight, anchor, op, ls, P)


def rect(x, y, w, h, fill=None, stroke=None, r=0, sw=1, op=None, dash=None,
         raw_fill=None, P=None):
    a = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'width="{max(w, 0):.1f}"',
         f'height="{max(h, 0):.1f}"']
    if r:
        a.append(f'rx="{r}"')
    a.append(f'fill="{raw_fill or (P[fill] if fill else "none")}"')
    if stroke:
        a += [f'stroke="{P[stroke]}"', f'stroke-width="{sw}"']
    if dash:
        a.append(f'stroke-dasharray="{dash}"')
    if op is not None:
        a.append(f'opacity="{op}"')
    return f'<rect {" ".join(a)}/>'


def line(x1, y1, x2, y2, stroke="grid", sw=1, dash=None, op=None, P=None):
    a = [f'x1="{x1:.1f}"', f'y1="{y1:.1f}"', f'x2="{x2:.1f}"', f'y2="{y2:.1f}"',
         f'stroke="{P[stroke]}"', f'stroke-width="{sw}"']
    if dash:
        a.append(f'stroke-dasharray="{dash}"')
    if op is not None:
        a.append(f'opacity="{op}"')
    return f'<line {" ".join(a)}/>'


def tick_mark(x, y, colour, P, s=12):
    return (f'<path d="M {x-s*0.40:.1f} {y:.1f} l {s*0.28:.1f} {s*0.32:.1f} '
            f'l {s*0.54:.1f} {-s*0.64:.1f}" fill="none" stroke="{P[colour]}" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')


def x_mark(x, y, colour, P, s=11):
    h = s * 0.36
    return (f'<path d="M {x-h:.1f} {y-h:.1f} L {x+h:.1f} {y+h:.1f} '
            f'M {x+h:.1f} {y-h:.1f} L {x-h:.1f} {y+h:.1f}" stroke="{P[colour]}" '
            f'stroke-width="2" stroke-linecap="round"/>')


def chip(x, y, label, colour, P, w=None, size=10.5, h=20, glyph=None):
    tw = w or (len(label) * size * 0.66 + 22)
    out = [rect(x, y - h / 2, tw, h, f"{colour}_wash", colour, r=h / 2, sw=1, P=P)]
    if glyph == "tick":
        out.append(tick_mark(x + 14, y - 1, colour, P, 10))
    elif glyph == "x":
        out.append(x_mark(x + 14, y, colour, P, 10))
    out.append(mono(x + tw / 2 + (6 if glyph else 0), y + 3.6, label, size, colour,
                    700, "middle", P=P))
    return out, tw


def trace_glyph(x, y, forked, P):
    """Two traces, coincident or forked. The whole product in seventy pixels."""
    base = (f"M {x} {y} L {x+14} {y} L {x+22} {y-8} L {x+34} {y-8} "
            f"L {x+42} {y} L {x+70} {y}")
    out = [f'<path d="{base}" fill="none" stroke="{P["step"]}" stroke-width="2" '
           f'stroke-linejoin="round" stroke-linecap="round"/>']
    if forked:
        out.append(f'<path d="M {x+42} {y} L {x+50} {y+9} L {x+70} {y+9}" fill="none" '
                   f'stroke="{P["drift"]}" stroke-width="2" stroke-linejoin="round" '
                   f'stroke-linecap="round" stroke-dasharray="1 3.4"/>')
        out.append(f'<circle cx="{x+42}" cy="{y}" r="2.8" fill="{P["drift"]}"/>')
    else:
        out.append(f'<path d="{base}" fill="none" stroke="{P["step_hi"]}" '
                   f'stroke-width="2" stroke-dasharray="1 3.4" stroke-linecap="round"/>')
    return out


def defs(P):
    return f"""<defs>
<linearGradient id="gstep" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="{P['step']}"/><stop offset="1" stop-color="{P['step_hi']}"/>
</linearGradient>
<linearGradient id="gdrift" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="{P['drift']}"/><stop offset="1" stop-color="{P['drift_hi']}"/>
</linearGradient>
<linearGradient id="gslate" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="{P['slate']}"/><stop offset="1" stop-color="{P['muted']}"/>
</linearGradient>
<linearGradient id="grule" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="{P['grid_hi']}"/>
  <stop offset="1" stop-color="{P['grid_hi']}" stop-opacity="0"/>
</linearGradient>
<pattern id="dots" width="14" height="14" patternUnits="userSpaceOnUse">
  <circle cx="1" cy="1" r="0.8" fill="{P['grid_hi']}" opacity="0.5"/>
</pattern>
<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
  <line x1="0" y1="0" x2="0" y2="7" stroke="{P['faint']}" stroke-width="1.6" opacity="0.42"/>
</pattern>
<pattern id="hatchd" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
  <line x1="0" y1="0" x2="0" y2="7" stroke="{P['drift']}" stroke-width="1.6" opacity="0.7"/>
</pattern>
<filter id="soft" x="-40%" y="-40%" width="180%" height="180%">
  <feGaussianBlur stdDeviation="{P['glow']}" result="b"/>
  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
</defs>"""


def chrome(P, height, kicker, title, sub, source, forked, accent):
    b = [rect(0, 0, W, height, "ground", "edge", r=16, sw=1, P=P),
         rect(1, 1, W - 2, height - 2, raw_fill="url(#dots)", r=15, op=0.5, P=P)]
    b.append(rect(MX, 26, 7, 7, accent, r=1.5, P=P))
    b.append(mono(MX + 17, 33, kicker, 10, accent, 700, ls="2.2", P=P))
    rule_x = MX + 24 + len(kicker) * 8.2
    b.append(rect(rule_x, 29.2, max(RX - 86 - rule_x, 0), 1, raw_fill="url(#grule)", P=P))
    b += trace_glyph(RX - 66, 30, forked, P)
    b.append(mono(MX, 68, title, 18, "ink", 700, ls="-0.3", P=P))
    if sub:
        b.append(txt(MX, 90, sub, 12.5, "muted", P=P))

    b.append(line(MX + 6, TOP - 8, MX + 6, height - 56, "grid_hi", 1, op=0.9, P=P))
    y, i = TOP - 8, 0
    while y < height - 56:
        long = i % 4 == 0
        b.append(line(MX + 6, y, MX + (13 if long else 10), y, "grid_hi", 1,
                      op=0.95 if long else 0.45, P=P))
        y += 13
        i += 1

    fy = height - 34
    b.append(line(MX, fy, RX + 14, fy, "grid", 1, P=P))
    b.append(rect(MX, fy + 13, 5, 5, "faint", r=1, op=0.8, P=P))
    b.append(mono(MX + 14, fy + 18.5, "SOURCE", 9.5, "faint", 700, ls="1.6", P=P))
    b.append(mono(MX + 76, fy + 18.5, source, 9.5, "faint", P=P))
    b.append(mono(RX + 14, fy + 18.5, "LOCKSTEP", 9.5, "faint", 700, "end", ls="1.6", P=P))
    return b


def render(name, P, theme, height, body):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
           f'width="{W}" height="{height}" role="img" aria-labelledby="ttl dsc">\n'
           f'<title id="ttl">{esc(TITLES[name])}</title>\n'
           f'<desc id="dsc">{esc(DESCS[name])}</desc>\n'
           + defs(P) + "\n" + "\n".join(body) + "\n</svg>\n")
    (OUT / f"{name}-{theme}.svg").write_text(svg, encoding="utf-8")


def arrow(x1, y1, x2, y2, colour="grid_hi", sw=1.2, dash=None, P=None):
    out = [line(x1, y1, x2, y2, colour, sw, dash=dash, P=P)]
    if x2 >= x1:
        out.append(f'<path d="M {x2:.1f} {y2:.1f} l -6 -3.4 l 0 6.8 z" fill="{P[colour]}"/>')
    else:
        out.append(f'<path d="M {x2:.1f} {y2:.1f} l 6 -3.4 l 0 6.8 z" fill="{P[colour]}"/>')
    return out


def wrap(text, n):
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > n:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def box(x, y, w, h, label, sub=None, colour=None, P=None, size=11.5):
    out = [rect(x, y, w, h, "panel", colour or "edge", r=7, sw=1.2 if colour else 1, P=P)]
    if colour:
        out.append(rect(x, y, 3, h, colour, r=1.5, P=P))
    ty = y + h / 2 + (0 if sub is None else -4)
    out.append(mono(x + w / 2, ty + 4, label, size, "ink", 600, "middle", P=P))
    if sub:
        out.append(mono(x + w / 2, ty + 19, sub, 9, "faint", anchor="middle", P=P))
    return out


# ---------------------------------------------------------------- figures ---


def fig_claims(P):
    b, y = [], TOP
    for cid, title, what, how, status in N["claims"]:
        acc = "step" if cid != "F1" else "drift"
        b.append(rect(LX, y, CW, 62, "panel", "edge", r=7, sw=1, P=P))
        b.append(rect(LX, y + 11, 3, 40, acc, r=1.5, P=P))
        b.append(line(MX + 6, y + 31, LX, y + 31, "grid_hi", 1, op=0.8, P=P))
        b.append(f'<circle cx="{MX+6}" cy="{y+31}" r="3" fill="{P[acc]}"/>')
        b.append(rect(LX + 16, y + 20, 40, 22, f"{acc}_wash", acc, r=4, sw=1, P=P))
        b.append(mono(LX + 36, y + 35, cid, 12.5, acc, 700, "middle", P=P))
        b.append(f'<text x="{LX+72}" y="{y+29}" font-family="{SANS}" font-size="13.5" '
                 f'fill="{P["ink"]}" font-weight="600">{esc(title)}'
                 f'<tspan dx="11" font-size="12.5" font-weight="400" fill="{P["muted"]}">'
                 f'{esc(what)}</tspan></text>')
        b.append(mono(LX + 72, y + 47, how, 10.5, "faint", P=P))
        cs, _ = chip(RX - 100, y + 31, status, acc, P, w=88, glyph="tick")
        b += cs
        y += 70
    b.append(mono(LX, y + 18, "every claim is scoped to the environment tuple in env.lock. "
                              "a claim without one is invalid by construction.",
                  10.5, "faint", P=P))
    return b, y + 56


def fig_three(P):
    b = []
    colw = (CW - 2 * 18) / 3
    for i, (kick, a, bnum, label, notes, gauge, colour) in enumerate(N["three"]):
        x = LX + i * (colw + 18)
        b.append(rect(x, TOP, colw, 178, "panel", "edge", r=8, sw=1, P=P))
        b.append(rect(x, TOP, colw, 2.5, raw_fill=f"url(#g{colour})", P=P))
        b.append(mono(x + 18, TOP + 28, kick, 9.5, colour, 700, ls="2", P=P))
        if gauge:
            b.append(f'<text x="{x+18}" y="{TOP+82}" font-family="{MONO}" font-size="44" '
                     f'font-weight="700" fill="url(#g{colour})" filter="url(#soft)">{a}'
                     f'<tspan font-size="20" fill="{P["faint"]}" dx="5">/</tspan>'
                     f'<tspan font-size="20" fill="{P["muted"]}" dx="5">{bnum}</tspan></text>')
        else:
            b.append(f'<text x="{x+18}" y="{TOP+82}" font-family="{MONO}" font-size="33" '
                     f'font-weight="700" fill="url(#g{colour})" filter="url(#soft)">{a}'
                     f'<tspan font-size="17" fill="{P["faint"]}" dx="6">to</tspan>'
                     f'<tspan dx="6">{bnum}</tspan></text>')
        gx, gy, gw = x + 18, TOP + 102, colw - 36
        if gauge:
            seg = (gw - 19 * 3) / 20
            for s in range(20):
                b.append(rect(gx + s * (seg + 3), gy, seg, 7, colour, r=1.5,
                              op=1 if s / 20 < gauge else 0.16, P=P))
        else:
            b.append(rect(gx, gy + 2, gw, 3, "grid_hi", r=1.5, P=P))
            for v in (0, 2, 4, 6):
                px = gx + gw * v / 6
                b.append(line(px, gy, px, gy + 7, "grid_hi", 1, P=P))
                b.append(mono(px, gy + 20, f"{v}x", 8.5, "faint", anchor="middle", P=P))
            b.append(rect(gx + gw * 5.2 / 6, gy - 2, gw * 0.5 / 6, 11,
                          raw_fill="url(#gdrift)", r=2, P=P))
        b.append(txt(x + 18, TOP + 138, label, 12, "ink", SANS, 600, P=P))
        for j, note in enumerate(notes):
            b.append(mono(x + 18, TOP + 156 + j * 14, note, 9.5, "faint", P=P))
    return b, TOP + 178 + 46


def fig_throughput(P):
    b = []
    x0, x1 = LX + 196, RX - 172
    scale = (x1 - x0) / 3.4
    y = TOP + 6
    for gv in (0, 1, 2, 3):
        gx = x0 + gv * scale
        b.append(line(gx, y - 8, gx, y + 6 * 42 - 12, "grid", 1,
                      dash=None if gv == 0 else "2 4", P=P))
        b.append(mono(gx, y + 6 * 42 + 4, f"{gv}s", 9.5, "faint", anchor="middle", P=P))
    for label, a, bb, spread, colour in N["throughput"]:
        b.append(txt(x0 - 16, y + 16, label, 12, "ink", anchor="end", P=P))
        b.append(rect(x0, y + 3, a * scale, 10, raw_fill=f"url(#g{colour})", r=2, P=P))
        b.append(rect(x0, y + 17, bb * scale, 10, raw_fill=f"url(#g{colour})", r=2,
                      op=0.38, P=P))
        b.append(mono(x1 + 16, y + 12, f"{a:.3f}", 10.5, "ink", 600, P=P))
        b.append(mono(x1 + 16, y + 26, f"{bb:.3f}", 10.5, "muted", P=P))
        b.append(mono(x1 + 74, y + 19, f"spread {spread:.2f}x", 9.5,
                      "drift" if spread >= 1.25 else "faint", P=P))
        y += 42
    y += 20
    b.append(rect(LX, y, CW, 138, "panel_alt", "edge", r=8, sw=1, P=P))
    b.append(mono(LX + 18, y + 26, "KILL CRITERION", 9.5, "drift", 700, ls="2", P=P))
    b.append(txt(LX + 142, y + 26, "set in week one, before anything was built: below 15 "
                                   "percent of vLLM default", 11.5, "muted", P=P))
    k = N["kill"]
    ax0, ax1 = LX + 46, RX - 250
    aw = ax1 - ax0
    ay = y + 108
    b.append(rect(ax0, ay, aw, 4, "grid_hi", r=2, P=P))
    for v in (0, 5, 10, 15, 20):
        px = ax0 + aw * v / k["span"]
        b.append(line(px, ay - 4, px, ay + 8, "grid_hi", 1, P=P))
        b.append(mono(px, ay + 22, f"{v}%", 9, "faint", anchor="middle", P=P))
    tx = ax0 + aw * k["threshold"] / k["span"]
    b.append(line(tx, ay - 58, tx, ay + 8, "ink", 1.4, dash="3 3", op=0.85, P=P))
    b.append(mono(ax0 - 12, ay - 40, "A", 9, "faint", 700, "end", P=P))
    b.append(mono(ax0 - 12, ay - 18, "B", 9, "faint", 700, "end", P=P))
    for v, colour, run_a in k["marks"]:
        px = ax0 + aw * v / k["span"]
        cy = ay - (44 if run_a else 22)
        b.append(f'<circle cx="{px:.1f}" cy="{cy:.1f}" r="4.5" fill="{P[colour]}"/>')
        b.append(mono(px + 9, cy + 3.6, f"{v}", 9.5, colour, 700, P=P))
    b.append(mono(RX - 232, y + 58, "eager", 10, "drift", 700, P=P))
    b.append(mono(RX - 186, y + 58, "14.6 and 16.6, straddling", 10, "muted", P=P))
    b.append(mono(RX - 232, y + 76, "graphs", 10, "step", 700, P=P))
    b.append(mono(RX - 186, y + 76, "10.6 and 11.5, below it", 10, "muted", P=P))
    b.append(mono(RX - 232, y + 100, "the bar is finer than this", 9.5, "faint", P=P))
    b.append(mono(RX - 232, y + 114, "measurement can resolve", 9.5, "faint", P=P))
    b.append(mono(LX + 18, y + 48, "solid bar run A, faded bar run B. two runs do not "
                                   "establish a distribution, which is why both are drawn.",
                  9.5, "faint", P=P))
    return b, y + 138 + 46


def fig_coverage(P):
    c = N["coverage"]
    b = []
    y = TOP

    def block(x, title, total, swarm, evict, probes, cols):
        out = [mono(x, y, title, 11.5, "ink", 700, ls="0.6", P=P)]
        reached = swarm + evict
        out.append(mono(x + 92, y, f"{reached} of {total}", 11.5, "step", 700, P=P))
        out.append(mono(x + 168, y, f"probes reach {probes}", 9.5, "faint", P=P))
        gy = y + 16
        for i in range(total):
            cx = x + (i % cols) * 32
            cy = gy + (i // cols) * 32
            if i < swarm:
                out.append(rect(cx, cy, 26, 26, raw_fill="url(#gstep)", r=3, P=P))
            elif i < reached:
                out.append(rect(cx, cy, 26, 26, "step", r=3, op=0.42, P=P))
            elif i < probes:
                out.append(rect(cx, cy, 26, 26, None, "step", r=3, sw=1.2, op=0.65, P=P))
            else:
                out.append(rect(cx, cy, 26, 26, None, "grid_hi", r=3, sw=1, P=P))
                out.append(rect(cx, cy, 26, 26, raw_fill="url(#hatch)", r=3, op=0.85, P=P))
        return out, gy + ((total + cols - 1) // cols) * 32

    o, b2 = block(LX, "2-GRAMS", c["two_total"], c["two_swarm"], c["two_evict"],
                  c["probes_two"], 5)
    b += o
    o, b3 = block(LX + 300, "3-GRAMS", c["three_total"], c["three_swarm"],
                  c["three_evict"], c["probes_three"], 12)
    b += o
    y = max(b2, b3) + 20

    b.append(rect(LX, y, CW, 88, "panel_alt", "edge", r=8, sw=1, P=P))
    b.append(mono(LX + 18, y + 28, "DENOMINATOR", 9.5, "drift", 700, ls="2", P=P))
    b.append(mono(LX + 134, y + 29, "27 and 84", 12.5, "faint", P=P))
    b.append(line(LX + 132, y + 25, LX + 202, y + 25, "faint", 1.3, P=P))
    b.append(mono(LX + 214, y + 29, "->", 12.5, "faint", P=P))
    b.append(mono(LX + 242, y + 29, "25 and 79", 12.5, "ink", 700, P=P))
    b.append(txt(LX + 334, y + 29, "one declared transition was never reachable", 11,
                 "muted", P=P))
    b.append(txt(LX + 18, y + 54, "every percentage published before the correction was "
                                  "computed against a denominator that was too large.", 11,
                 "muted", P=P))
    b.append(txt(LX + 18, y + 72, "the correction moves them up, 55.6 to 60.0 percent on "
                                  "2-grams at identical observed counts, which is exactly "
                                  "why it is stated here.", 11, "muted", P=P))
    y += 88 + 24

    x = LX
    for label, kind in (("swarm campaign, 72 cases", "solid"),
                        ("plus eviction, 132 cases", "faded"),
                        ("targeted probe only", "outline"),
                        ("never reached", "hatch")):
        if kind == "solid":
            b.append(rect(x, y - 9, 12, 12, raw_fill="url(#gstep)", r=2, P=P))
        elif kind == "faded":
            b.append(rect(x, y - 9, 12, 12, "step", r=2, op=0.42, P=P))
        elif kind == "outline":
            b.append(rect(x, y - 9, 12, 12, None, "step", r=2, sw=1.2, op=0.65, P=P))
        else:
            b.append(rect(x, y - 9, 12, 12, None, "grid_hi", r=2, sw=1, P=P))
            b.append(rect(x, y - 9, 12, 12, raw_fill="url(#hatch)", r=2, P=P))
        b.append(mono(x + 19, y + 1, label, 10, "muted", P=P))
        x += len(label) * 6.2 + 40
    return b, y + 48


def fig_mutation(P):
    m = N["mutation"]
    b = []
    y = TOP
    tile, gap = 34, 8
    gx = RX - (m["n"] * tile + (m["n"] - 1) * gap)
    bands = [("invariance relations and F1 only",
              ["the fold reversal is not a function of the schedule,",
               "so the engine agrees with itself and both pass"], m["killed_bare"]),
             ("with golden bytes",
              ["a committed sha256 over raw fp16 logit bytes,",
               "compared exactly, from outside the process"], m["n"])]
    for title, notes, killed in bands:
        b.append(rect(LX, y, CW, 96, "panel", "edge", r=8, sw=1, P=P))
        b.append(txt(LX + 18, y + 30, title, 12.5, "ink", SANS, 600, P=P))
        for j, note in enumerate(notes):
            b.append(mono(LX + 18, y + 48 + j * 14, note, 9.5, "faint", P=P))
        cs, _ = chip(LX + 18, y + 80, f"{killed} of {m['n']} killed",
                     "step" if killed == m["n"] else "drift", P, w=116)
        b += cs
        for i in range(m["n"]):
            tx, ty = gx + i * (tile + gap), y + 22
            dead = i < killed
            if dead:
                b.append(rect(tx, ty, tile, tile, "step_wash", "step", r=7, sw=1.2, P=P))
                b.append(tick_mark(tx + tile / 2, ty + tile / 2 - 1, "step", P, 13))
            else:
                b.append(rect(tx, ty, tile, tile, "drift_wash", "drift", r=7, sw=1.4, P=P))
                b.append(rect(tx, ty, tile, tile, raw_fill="url(#hatchd)", r=7, op=0.5, P=P))
                b.append(x_mark(tx + tile / 2, ty + tile / 2, "drift", P, 13))
            b.append(mono(tx + tile / 2, ty + tile + 14, f"O{i+1}", 8.5,
                          "step" if dead else "drift", 700, "middle", P=P))
        y += 112
    sx = gx + (m["n"] - 1) * (tile + gap) + tile / 2
    b.append(line(sx, TOP + 74, sx, TOP + 134, "drift", 1.2, dash="3 3", P=P))
    b.append(mono(LX, y - 4, "the one mutant that separates the two rows:", 10.5, "muted", P=P))
    b.append(mono(LX + 252, y - 4, m["survivor"], 10.5, "drift", 700, P=P))
    b.append(mono(LX, y + 16, "every operator fires a sentinel from inside the mutant body, so "
                              "no trial here is a mutant that never ran. path executed,",
                  10, "faint", P=P))
    b.append(mono(LX, y + 31, "patch executed and fault injected are three claims, and the "
                              "counters can only tell you about the first.", 10, "faint", P=P))
    return b, y + 76


def fig_thesis(P):
    b = []
    slots = len(N["thesis"])
    gut = 250
    tx0 = LX + gut
    colw = (CW - gut) / slots
    lane_h = 42
    y = TOP + 24
    x_start, x_end = tx0 + colw * 8, tx0 + colw * 16
    b.append(f'<path d="M {x_start:.1f} {TOP+14:.1f} L {x_start:.1f} {TOP+6:.1f} '
             f'L {x_end:.1f} {TOP+6:.1f} L {x_end:.1f} {TOP+14:.1f}" fill="none" '
             f'stroke="{P["drift"]}" stroke-width="1.2" opacity="0.9"/>')
    b.append(mono((x_start + x_end) / 2, TOP - 2, "FOUND FROM OUTSIDE", 9,
                  "drift", 700, "middle", ls="1.6", P=P))

    lanes = {k: i for i, (k, _, _, _) in enumerate(N["lanes"])}
    for i, (key, name, note, count) in enumerate(N["lanes"]):
        ly = y + i * lane_h
        b.append(rect(LX, ly, CW, lane_h - 6, "panel" if i % 2 == 0 else "panel_alt",
                      "edge", r=6, sw=1, P=P))
        b.append(line(tx0 - 12, ly + 6, tx0 - 12, ly + lane_h - 12, "grid_hi", 1,
                      op=0.7, P=P))
        colour = "drift" if key in ("audit", "reader") else (
            "slate" if key == "anomaly" else "step")
        b.append(mono(LX + 14, ly + 21, str(count), 14, colour, 700, P=P))
        b.append(txt(LX + 36, ly + 17, name, 12, "ink", SANS, 600, P=P))
        b.append(mono(LX + 36, ly + 30, note, 8.5, "faint", P=P))

    for idx, (label, key) in enumerate(N["thesis"]):
        cx = tx0 + colw * idx + colw / 2
        ly = y + lanes[key] * lane_h
        colour = "drift" if key in ("audit", "reader") else (
            "slate" if key == "anomaly" else "step")
        s = 25
        bx, by = cx - s / 2, ly + (lane_h - 6) / 2 - s / 2
        if key == "reader":
            b.append(rect(bx, by, s, s, f"{colour}_wash", colour, r=6, sw=1.5, P=P))
            fill = colour
        elif key == "anomaly":
            b.append(rect(bx, by, s, s, "slate", r=6, P=P))
            fill = "ground"
        else:
            b.append(rect(bx, by, s, s, raw_fill=f"url(#g{colour})", r=6, P=P))
            fill = "ground"
        b.append(mono(cx, by + s / 2 + 3.4, label, 9.5 if len(label) < 3 else 7.5,
                      fill, 700, "middle", P=P))

    ay = y + len(N["lanes"]) * lane_h + 6
    b.append(line(tx0, ay, RX, ay, "grid", 1, P=P))
    for idx, (label, _) in enumerate(N["thesis"]):
        cx = tx0 + colw * idx + colw / 2
        b.append(line(cx, ay, cx, ay + 4, "grid_hi", 1, P=P))
        b.append(mono(cx, ay + 16, label, 8, "faint", anchor="middle", P=P))
    b.append(mono(LX, ay + 12, "finding index", 9, "faint", 700, "end", ls="1.2", P=P)
             .replace(f'x="{LX:.1f}"', f'x="{tx0-14:.1f}"'))
    b.append(mono(LX, ay + 40, "eighteen tiles, seventeen findings: 12b is a row and not a "
                               "count, because the audit named it outright and the witness "
                               "check only", 10, "faint", P=P))
    b.append(mono(LX, ay + 55, "reproduced it from a different direction. counting it as new "
                               "would inflate the one table whose only value is that it does "
                               "not.", 10, "faint", P=P))
    return b, ay + 92


def fig_certification(P):
    b = []
    y = TOP
    bx0, bw, maxb = LX + 296, 140, 48
    b.append(mono(LX, y, "BOUNDARY CASE", 9, "faint", 700, ls="1.8", P=P))
    b.append(mono(bx0, y, "CO-RESIDENT", 9, "faint", 700, ls="1.8", P=P))
    b.append(mono(bx0 + bw + 42, y, "VERDICT", 9, "faint", 700, ls="1.8", P=P))
    y += 12
    for label, batch, delta in N["certification"]:
        clean = delta is None
        b.append(rect(LX, y, CW, 34, "panel" if clean else "drift_wash",
                      "edge" if clean else "drift", r=6, sw=1, P=P))
        b.append(txt(LX + 16, y + 22, label, 12, "ink", P=P))
        b.append(rect(bx0, y + 14, bw, 6, "grid_hi", r=3, op=0.55, P=P))
        b.append(rect(bx0, y + 14, bw * batch / maxb, 6,
                      raw_fill="url(#gstep)" if clean else "url(#gdrift)", r=3, P=P))
        b.append(mono(bx0 + bw + 32, y + 22, str(batch), 11, "muted", 600, "end", P=P))
        cs, _ = chip(bx0 + bw + 42, y + 17, "clean" if clean else "diverged",
                     "step" if clean else "drift", P, w=84 if clean else 92,
                     glyph="tick" if clean else "x")
        b += cs
        if delta:
            b.append(mono(bx0 + bw + 148, y + 22, f"max delta {delta}", 10, "drift", P=P))
        y += 40
    y += 14
    b.append(rect(LX, y, CW, 152, "panel_alt", "edge", r=8, sw=1, P=P))
    b.append(mono(LX + 18, y + 28, "CONTROL", 9.5, "step", 700, ls="2", P=P))
    b.append(txt(LX + 100, y + 28, "one knob, and the only controlled single-variable result "
                                   "in this section", 11.5, "muted", P=P))
    cy = y + 46
    for flag, batch, verdict, clean in N["control"]:
        cy += 24
        b.append(mono(LX + 18, cy, flag, 11, "step" if clean else "muted",
                      700 if clean else 400, P=P))
        b.append(rect(LX + 258, cy - 8, 104 * batch / maxb, 8,
                      raw_fill="url(#gstep)" if clean else "url(#gdrift)", r=4, P=P))
        b.append(mono(LX + 374, cy, f"{batch} co-resident", 10, "faint", P=P))
        b.append(mono(LX + 484, cy, verdict, 11, "step" if clean else "drift",
                      600 if clean else 400, P=P))
    b.append(mono(LX + 18, y + 138, "intermittent at about one lifetime in three, so which "
                                    "cases trip is not the claim. token ids never move, only "
                                    "logprobs. filed as vllm#51187.", 9.5, "faint", P=P))
    return b, y + 152 + 46



def fig_hero(P):
    h = 396
    b = [rect(0, 0, W, h, "ground", "edge", r=16, sw=1, P=P),
         rect(1, 1, W - 2, h - 2, raw_fill="url(#dots)", r=15, op=0.55, P=P)]
    b.append(f'<text x="{MX+14}" y="96" font-family="{MONO}" font-size="52" '
             f'font-weight="700" letter-spacing="-1.5" fill="url(#gstep)" '
             f'filter="url(#soft)">LOCKSTEP</text>')
    b.append(txt(MX + 18, 124, "a batch-invariant inference engine, and the harness that "
                               "tries to prove it wrong", 13.5, "muted", P=P))
    b.append(mono(RX + 14, 62, "sm_89  ::  torch 2.6.0  ::  Qwen3-0.6B fp16", 9.5, "faint",
                  anchor="end", P=P))
    b.append(mono(RX + 14, 80, "every claim scoped to this tuple", 9.5, "faint",
                  anchor="end", P=P))

    ty = 222
    fork = MX + 470
    base = (f"M {MX+18} {ty} L {MX+120} {ty} L {MX+150} {ty-28} L {MX+250} {ty-28} "
            f"L {MX+280} {ty} L {fork} {ty}")
    b.append(f'<path d="{base}" fill="none" stroke="{P["step"]}" stroke-width="6" '
             f'stroke-linejoin="round" stroke-linecap="round" opacity="0.5"/>')
    b.append(f'<path d="{base}" fill="none" stroke="{P["step_hi"]}" stroke-width="1.8" '
             f'stroke-linejoin="round" stroke-linecap="round"/>')
    b.append(f'<path d="M {fork} {ty} L {RX-16} {ty}" fill="none" stroke="{P["step"]}" '
             f'stroke-width="2.4" stroke-linecap="round"/>')
    b.append(f'<path d="M {fork} {ty} L {fork+34} {ty+32} L {RX-16} {ty+32}" fill="none" '
             f'stroke="{P["drift"]}" stroke-width="2.4" stroke-linejoin="round" '
             f'stroke-linecap="round" stroke-dasharray="7 5"/>')
    b.append(f'<circle cx="{fork}" cy="{ty}" r="5.5" fill="{P["drift"]}"/>')
    b.append(line(fork, ty - 46, fork, ty - 10, "drift", 1, dash="3 3", op=0.85, P=P))
    b.append(mono(fork + 12, ty - 52, "44 co-resident, byte-identical, same order",
                  10, "drift", 700, P=P))
    b.append(mono(MX + 18, ty - 54, "TWO TRACES, EXACTLY OVERLAID", 9.5, "faint", 700,
                  ls="2.2", P=P))
    b.append(mono(MX + 18, ty + 30, "bitwise identical under any schedule, preemption "
                                    "or eviction", 10, "step", P=P))
    b.append(mono(RX - 16, ty + 54, "logprobs disagree  ->  vllm-project/vllm#51187", 10,
                  "drift", 700, "end", P=P))

    y = 300
    b.append(line(MX, y - 22, RX + 14, y - 22, "grid", 1, P=P))
    stats = [("65 / 65", "relation runs, bitwise", "step"),
             ("10 / 10", "seeded faults killed", "step"),
             ("19 / 25", "lifecycle 2-grams reached", "step"),
             ("1", "finding filed upstream", "drift")]
    cw = (RX + 14 - MX) / 4
    for i, (big, lab, colour) in enumerate(stats):
        x = MX + i * cw
        if i:
            b.append(line(x - 14, y - 8, x - 14, y + 42, "grid", 1, P=P))
        b.append(f'<text x="{x:.1f}" y="{y+22}" font-family="{MONO}" font-size="26" '
                 f'font-weight="700" fill="url(#g{colour})">{big}</text>')
        b.append(mono(x, y + 42, lab, 10, "faint", P=P))
    b.append(mono(MX, h - 22, "the engine is a fixture. the harness is the product.", 10,
                  "muted", 700, P=P))
    b.append(mono(RX + 14, h - 22, "u7k4rs6/LockStep", 10, "faint", anchor="end", P=P))
    return b, h


def fig_architecture(P):
    b = []
    y = TOP + 6
    ins = [("workload W", "the requests"), ("schedule sigma", "the interleaving"),
           ("seeds", "sampling, keyed")]
    for i, (lab, sub) in enumerate(ins):
        b += box(LX, y + i * 46, 150, 38, lab, sub, "step", P, size=10.5)
    ex = LX + 180
    b += arrow(LX + 154, y + 66, ex - 6, y + 66, "grid_hi", 1.2, P=P)
    b += box(ex, y + 22, 150, 88, "lockstep engine", "paged KV, preemption", None, P)
    rx0 = ex + 182
    b += arrow(ex + 154, y + 46, rx0 - 6, y + 46, "grid_hi", 1.2, P=P)
    b += arrow(ex + 154, y + 90, rx0 - 6, y + 90, "grid_hi", 1.2, P=P)
    b += box(rx0, y + 28, 168, 38, "run under sigma", None, "step", P, size=10.5)
    b += box(rx0, y + 72, 168, 38, "canonical, batch 1", None, "step", P, size=10.5)
    ox = rx0 + 200
    b += arrow(rx0 + 172, y + 46, ox - 6, y + 60, "grid_hi", 1.2, P=P)
    b += arrow(rx0 + 172, y + 90, ox - 6, y + 76, "grid_hi", 1.2, P=P)
    b += box(ox, y + 28, RX - ox, 82, "oracles", "bitwise, no tolerance", "drift", P)
    b.append(mono(rx0 + 84, y + 128, "the same engine re-run, not a second implementation",
                  9.5, "faint", anchor="middle", P=P))
    y += 152
    for x, w, kicker, chips, note, colour in (
            (LX, (CW - 24) / 2, "INTERNAL, WHITE BOX",
             ["ddmin", "witness.json", "report.html"],
             "sees the whole trajectory hash, so a finding minimises to a 1-minimal case "
             "and replays by hash", "step"),
            (LX + (CW - 24) / 2 + 24, (CW - 24) / 2, "EXTERNAL, BLACK BOX",
             ["certifier", "HTTP", "vLLM, SGLang"],
             "sees logprobs and nothing else, which is why a clean result means no "
             "divergence at that observable", "drift")):
        b.append(rect(x, y, w, 118, "panel_alt", "edge", r=8, sw=1, P=P))
        b.append(mono(x + 16, y + 24, kicker, 9.5, colour, 700, ls="2", P=P))
        cx = x + 16
        for j, c in enumerate(chips):
            cs, cwid = chip(cx, y + 52, c, colour, P)
            b += cs
            cx += cwid + 8
            if j < len(chips) - 1:
                b.append(mono(cx - 5, y + 56, "->", 9.5, "faint", P=P))
                cx += 12
        for j, ln in enumerate(wrap(note, 54)[:2]):
            b.append(mono(x + 16, y + 82 + j * 14, ln, 9.5, "faint", P=P))
    return b, y + 118 + 46


def fig_observers(P):
    rows = [("I1 to I4", "the engine", "itself, under a perturbed schedule",
             "any perturbation uniform across schedules", "step"),
            ("F1", "the engine", "an fp64 CPU reference",
             "anything inside the 0.5 tolerance", "step"),
            ("golden bytes", "the engine", "a committed sha256 baseline",
             "any fault present when the baseline was written", "step"),
            ("PATH-EQ", "the observed path", "the path the scheduler serves",
             "anything both implementations get wrong identically", "drift")]
    b = []
    y = TOP
    b.append(mono(LX + 14, y, "OBSERVER", 9, "faint", 700, ls="1.8", P=P))
    b.append(mono(LX + 150, y, "COMPARES", 9, "faint", 700, ls="1.8", P=P))
    b.append(mono(LX + 290, y, "AGAINST", 9, "faint", 700, ls="1.8", P=P))
    b.append(mono(LX + 480, y, "BLIND TO", 9, "faint", 700, ls="1.8", P=P))
    y += 12
    for name, cmp_, against, blind, colour in rows:
        b.append(rect(LX, y, CW, 44, "panel", "edge", r=7, sw=1, P=P))
        b.append(rect(LX, y + 10, 3, 24, colour, r=1.5, P=P))
        b.append(mono(LX + 14, y + 27, name, 11.5, colour, 700, P=P))
        b.append(txt(LX + 150, y + 27, cmp_, 11, "muted", P=P))
        b.append(txt(LX + 290, y + 27, against, 11, "muted", P=P))
        b.append(txt(LX + 480, y + 27, blind, 11, "ink", P=P))
        y += 50
    y += 8
    b.append(rect(LX, y, CW, 104, "drift_wash", "drift", r=8, sw=1, P=P))
    b.append(mono(LX + 18, y + 26, "THE ONE THEY SHARED", 9.5, "drift", 700, ls="2", P=P))
    b.append(txt(LX + 18, y + 50, "three observers with distinct blind spots still share one "
                                  "if all three read a file the engine does not serve from.",
                 11.5, "ink", P=P))
    b.append(txt(LX + 18, y + 70, "I1 to I4, F1 and golden bytes all called model.forward, "
                                  "while every served request goes through forward_batch.",
                 11, "muted", P=P))
    b.append(txt(LX + 18, y + 86, "PATH-EQ closed it: 3 prompts, 730 positions, bitwise, "
                                  "and all four committed digests unchanged by the move.",
                 11, "muted", P=P))
    return b, y + 104 + 46


def fig_limits(P):
    items = [("no novelty on the kernels", "diagnosed by Thinking Machines, shipped by "
              "vLLM and SGLang. reimplemented so the harness can mutate it"),
             ("no cross-hardware guarantee", "every bitwise claim is scoped to one "
              "environment tuple. nothing is known to hold on another GPU or torch"),
             ("no throughput superiority", "a correctness-reference engine, no CUDA graphs, "
              "no attempt to be fast"),
             ("one model", "Qwen3-0.6B fp16 only"),
             ("no copy-on-write", "prefix sharing is whole-block only, which holds only "
              "because there is no fork API. add one and COW comes back"),
             ("ten operators, not fifty", "a thin denominator, and three earlier ones were "
              "deleted with the unreachable code they targeted"),
             ("coverage is against a model", "the denominator comes from a declared "
              "transition relation, checked both ways. it is still a model"),
             ("counters prove the path ran", "not that the patch took effect, and not that "
              "the fault was injected. three claims, three checks")]
    b = []
    y = TOP
    colw = (CW - 20) / 2
    for i, (title, note) in enumerate(items):
        x = LX + (i % 2) * (colw + 20)
        yy = y + (i // 2) * 74
        b.append(rect(x, yy, colw, 64, "panel", "edge", r=7, sw=1, P=P))
        b.append(f'<circle cx="{x+18}" cy="{yy+22}" r="3.4" fill="{P["drift"]}"/>')
        b.append(txt(x + 32, yy + 26, title, 12, "ink", SANS, 600, P=P))
        for j, ln in enumerate(wrap(note, 56)[:2]):
            b.append(mono(x + 32, yy + 43 + j * 12, ln, 9, "faint", P=P))
    return b, y + 4 * 74 + 30


def fig_prior_art(P):
    rows = [("Thinking Machines", "the diagnosis, taken as given", "batch_invariant_ops"),
            ("vLLM, SGLang", "shipped deterministic modes", "the subjects certified here"),
            ("GRIEF", "greybox fuzzing of serving systems", "confirms by replay against a "
             "live server"),
            ("LLM-42", "argues invariance is over-constrained", "conceded, and the price "
             "is measured"),
            ("MarginGate", "sparse margin-triggered verification", "adjacent to F1's "
             "near-tie threshold"),
            ("swarm, ddmin, mutants", "Groce 2012, Zeller 2002, Just 2014", "the methods "
             "this harness is built from")]
    b = []
    y = TOP
    for name, what, rel in rows:
        b.append(rect(LX, y, CW, 42, "panel", "edge", r=7, sw=1, P=P))
        b.append(mono(LX + 16, y + 26, name, 11.5, "step", 700, P=P))
        b.append(txt(LX + 190, y + 26, what, 11.5, "ink", P=P))
        b.append(mono(LX + 440, y + 26, rel, 9.5, "faint", P=P))
        y += 48
    y += 10
    b.append(rect(LX, y, CW, 88, "step_wash", "step", r=8, sw=1, P=P))
    b.append(mono(LX + 18, y + 26, "WHAT IS ACTUALLY DIFFERENT", 9.5, "step", 700,
                  ls="2", P=P))
    b.append(txt(LX + 18, y + 50, "the schedule is supplied rather than observed, so "
                                  "minimisation is exact rather than confirmed.", 11.5,
                 "ink", P=P))
    b.append(txt(LX + 18, y + 70, "ddmin proves a case 1-minimal, and the replay is a hash "
                                  "comparison rather than a re-observation.", 11.5,
                 "muted", P=P))
    return b, y + 88 + 46


def fig_evidence(P):
    rows = [("verify-0002.json", "claim 1, 65 of 65 relation runs"),
            ("fuzz-0002.json", "claim 2, 10 of 10 killed over 192 cases"),
            ("throughput-0004.json", "claim 3, isolated and interleaved"),
            ("fidelity-0001.json", "F1, 7 of 7 bounds"),
            ("case-0003.json", "the eviction finding, minimised and 1-minimal"),
            ("case-witness.json", "a replay witness, runs on a fresh clone"),
            ("certify-pairs-{a,b,mns8}.json", "the three runs vllm#51187 rests on"),
            ("certify-clean-revision.json", "the same finding at a clean revision"),
            ("upstream-finding.json", "provenance for the filed issue")]
    b = []
    y = TOP
    b.append(mono(LX + 14, y, "COMMITTED UNDER evidence/", 9, "faint", 700, ls="1.8", P=P))
    b.append(mono(LX + 340, y, "BACKS", 9, "faint", 700, ls="1.8", P=P))
    y += 12
    for name, backs in rows:
        b.append(rect(LX, y, CW, 32, "panel", "edge", r=6, sw=1, P=P))
        b.append(f'<circle cx="{LX+16}" cy="{y+16}" r="3" fill="{P["step"]}"/>')
        b.append(mono(LX + 30, y + 20, name, 10.5, "ink", 600, P=P))
        b.append(line(LX + 300, y + 16, LX + 330, y + 16, "grid_hi", 1, P=P))
        b.append(txt(LX + 340, y + 20, backs, 11, "muted", P=P))
        y += 37
    y += 8
    b.append(rect(LX, y, CW, 74, "panel_alt", "edge", r=8, sw=1, P=P))
    b.append(mono(LX + 18, y + 26, "REPLAY", 9.5, "step", 700, ls="2", P=P))
    b.append(mono(LX + 90, y + 26, "uv run ./lockstep replay evidence/case-witness.json",
                  11, "ink", 600, P=P))
    b.append(txt(LX + 18, y + 50, "recomputes a committed trajectory hash in a fresh process "
                                  "and compares. no GPU, no server, seconds after a clone.",
                 11, "muted", P=P))
    b.append(txt(LX + 18, y + 66, "results/ stays gitignored: bulk campaign output, "
                                  "superseded every run, and no published number points at "
                                  "one.", 11, "muted", P=P))
    return b, y + 74 + 46


def fig_gates(P):
    auto = ["no atomics, no autotune, no torch reductions in engine/",
            "the static gate is not vacuous: lifting the exception hits once",
            "secret scan", "every source file parses",
            "coverage denominators recomputed from the transition relation",
            "artifacts carry an environment tuple and the repro rebuilds",
            "tests that do not need CUDA", "the report builds from committed evidence"]
    manual = [("make claim1", "bitwise invariance, MR1 to MR8 and PATH-EQ"),
              ("make claim2", "fuzz campaign with the coverage floor"),
              ("make claim3", "throughput regression"),
              ("make certify", "certification against a local vLLM")]
    b = []
    y = TOP
    colw = (CW - 24) / 2
    b.append(rect(LX, y, colw, 218, "panel", "edge", r=8, sw=1, P=P))
    b.append(mono(LX + 16, y + 26, "AUTOMATED, EVERY PUSH", 9.5, "step", 700, ls="2", P=P))
    for i, item in enumerate(auto):
        yy = y + 50 + i * 20
        b.append(tick_mark(LX + 22, yy - 4, "step", P, 11))
        b.append(mono(LX + 36, yy, item, 9, "muted", P=P))
    x2 = LX + colw + 24
    b.append(rect(x2, y, colw, 218, "panel", "edge", r=8, sw=1, P=P))
    b.append(mono(x2 + 16, y + 26, "MANUAL, NEEDS A GPU", 9.5, "drift", 700, ls="2", P=P))
    for i, (cmd, what) in enumerate(manual):
        yy = y + 56 + i * 38
        b.append(mono(x2 + 22, yy, cmd, 10.5, "ink", 600, P=P))
        b.append(mono(x2 + 22, yy + 15, what, 9, "faint", P=P))
    b.append(mono(x2 + 22, y + 196, "their artifacts are committed with the revision", 9,
                  "faint", P=P))
    b.append(mono(x2 + 22, y + 208, "that produced them, which is the substitute", 9,
                  "faint", P=P))
    y += 232
    b.append(rect(LX, y, CW, 80, "drift_wash", "drift", r=8, sw=1, P=P))
    b.append(mono(LX + 18, y + 26, "FOR MOST OF THIS PROJECT NONE OF THESE RAN", 9.5,
                  "drift", 700, ls="1.6", P=P))
    b.append(txt(LX + 18, y + 48, "five CI gates were declared in the architecture doc and "
                                  "no .github/ directory existed. found by a third party",
                 11, "muted", P=P))
    b.append(txt(LX + 18, y + 66, "reading the directory tree, which is row 15 of the "
                                  "thesis table.", 11, "muted", P=P))
    return b, y + 80 + 46



FIGS = [
    ("claims", fig_claims, "CLAIMS", "Five relations, and what measures each one",
     "every claim scoped to one environment tuple, bitwise unless a tolerance is stated",
     False, "step"),
    ("three-numbers", fig_three, "HEADLINE", "The three numbers",
     "invariance, harness power, and the price of the constraint", False, "step"),
    ("throughput", fig_throughput, "CLAIM 3 THROUGHPUT",
     "Cost of determinism, two runs of the settled design",
     "8 requests, 2972 tokens, median of 5 samples, every measurement in its own process",
     False, "step"),
    ("coverage", fig_coverage, "COVERAGE", "Lifecycle n-grams, against the real denominator",
     "reported by population, because a case built to reach a transition is not exploration",
     False, "step"),
    ("mutation", fig_mutation, "CLAIM 2 HARNESS POWER",
     "What the harness catches, and what took a third observer",
     "identical operator set, identical campaign, one mutant of difference", True, "drift"),
    ("thesis", fig_thesis, "THE THESIS, ON ITS AUTHOR",
     "Seventeen times this repository declared more surface than it tested",
     "placed by finding index, laned by who found it", True, "drift"),
    ("architecture", fig_architecture, "HOW IT FITS TOGETHER",
     "Workload and schedule are separate inputs",
     "which is what makes minimisation exact rather than confirmed", False, "step"),
    ("observers", fig_observers, "TEST ORACLES",
     "Four observers, and what each one cannot see",
     "metamorphic relations, a reference, a committed baseline, and a path equivalence",
     True, "drift"),
    ("limits", fig_limits, "WHAT THIS DOES NOT CLAIM",
     "The limits, stated before a reader finds them",
     "each one is a place where the declared surface stops", True, "drift"),
    ("prior-art", fig_prior_art, "PRIOR ART",
     "Checked against its source rather than cited from memory",
     "and what is actually different here", False, "step"),
    ("evidence", fig_evidence, "EVIDENCE",
     "Every published number, and the artifact behind it",
     "committed on purpose, replayable on a fresh clone", False, "step"),
    ("gates", fig_gates, "GATES",
     "What runs on every push, and what runs by hand",
     "hosted runners have no NVIDIA device, so the split is explicit", True, "drift"),
    ("certification", fig_certification, "CERTIFICATION",
     "Black-box differential testing of vLLM batch-invariant mode",
     "concurrent submission, co-residency read from the engine's own gauge, every pair "
     "of repeats compared", True, "drift"),
]

TITLES = {
    "hero": "Lockstep",
    "architecture": "How the pieces fit together",
    "observers": "Four test oracles and their blind spots",
    "limits": "What this project does not claim",
    "prior-art": "Prior art, and what differs here",
    "evidence": "Committed evidence artifacts and what each backs",
    "gates": "Which gates run automatically and which run by hand",
    "claims": "Claims I1 to I4 and F1",
    "three-numbers": "The three headline numbers",
    "throughput": "Wall time across six configurations, and the kill criterion",
    "coverage": "Lifecycle n-gram coverage against the corrected denominator",
    "mutation": "Mutation campaign, with and without golden bytes",
    "thesis": "Seventeen findings, laned by who found them",
    "certification": "vLLM certification across seven boundary cases",
}

DESCS = {
    "hero": "Lockstep: a batch-invariant inference engine and the harness that tries to "
            "prove it wrong. Two traces of one workload run coincident and then fork at 44 "
            "co-resident byte-identical requests, where logprobs disagree, which became "
            "vllm-project/vllm issue 51187. 65 of 65 relation runs bitwise, 10 of 10 seeded "
            "faults killed, 19 of 25 lifecycle 2-grams reached, one finding filed upstream.",
    "architecture": "Workload, schedule and seeds feed the engine, which runs the workload "
                    "under the given schedule and again under a canonical batch-1 schedule. "
                    "Oracles compare the two bitwise. An internal white-box loop runs through "
                    "ddmin, a witness artifact and the report; an external black-box loop "
                    "reads logprobs over HTTP from vLLM or SGLang.",
    "observers": "Four observers. I1 to I4 compare the engine against itself under a "
                 "perturbed schedule, F1 against an fp64 reference, golden bytes against a "
                 "committed baseline, and PATH-EQ compares the observed path against the "
                 "path the scheduler serves. The first three shared a blind spot because all "
                 "read model.forward while every served request goes through forward_batch.",
    "limits": "Eight limits: no novelty on the kernels, no cross-hardware guarantee, no "
              "throughput superiority, one model only, no copy-on-write, ten mutation "
              "operators rather than fifty, coverage against a derived model, and execution "
              "counters proving only that the path ran.",
    "prior-art": "Thinking Machines, vLLM and SGLang, GRIEF, LLM-42, MarginGate, and the "
                 "swarm testing, ddmin and mutation testing literature. What differs here is "
                 "that the schedule is supplied rather than observed, so minimisation is "
                 "exact rather than confirmed.",
    "evidence": "Nine committed artifacts under evidence/ and what each one backs, from the "
                "relation runs and the fuzz campaign to the three runs the upstream filing "
                "rests on. A replay command recomputes a committed trajectory hash on a "
                "fresh clone with no GPU.",
    "gates": "Eight checks automated on every push, four manual gates that need a GPU, and "
             "the note that for most of this project none of the declared CI gates ran "
             "because no .github directory existed.",
    "claims": "Five claims with what verifies each. I1 batch invariance, I2 schedule "
              "invariance, I3 replay determinism and I4 RNG isolation all hold. F1 fidelity "
              "passes 7 of 7 bounds against an fp64 CPU reference.",
    "three-numbers": "65 of 65 relation runs bitwise identical, 10 of 10 seeded faults killed "
                     "with none equivalent, and lockstep invariant running at 5.2 to 5.7 times "
                     "the wall time of vLLM batch-invariant in eager mode.",
    "throughput": "Wall time for six configurations, run A solid and run B faded: lockstep fast "
                  "and invariant, and vLLM default and batch-invariant, each eager and graphed. "
                  "Below, the 15 percent kill criterion, with the eager comparison at 14.6 and "
                  "16.6 percent straddling the bar and the graphed comparison at 10.6 and 11.5 "
                  "percent consistently below it.",
    "coverage": "19 of 25 two-grams and 43 of 79 three-grams reached, split between the swarm "
                "campaign and the eviction campaign, with probe-only cells outlined and "
                "never-reached cells hatched. The denominator was corrected from 27 and 84.",
    "mutation": "Ten mutation operators. Nine killed by the invariance relations and F1 alone, "
                "with the reversed split-combine fold surviving. Ten killed once golden bytes "
                "are added as an observer outside the process.",
    "thesis": "Eighteen tiles for seventeen findings, laned by who found them: eight from this "
              "project's own machinery, five from an outside audit, one from an anomaly, two "
              "from readers not looking for defects, and two from this project auditing itself "
              "later. Findings 9 to 15 came from outside the repository.",
    "certification": "Seven boundary cases against vLLM batch-invariant mode: five clean, and "
                     "two diverged at 44 and 45 co-resident with logprob deltas of 4.685e-02 "
                     "and 3.906e-02 and no token divergence. The max-num-seqs control restores "
                     "reproducibility at 8.",
}


def main():
    for theme, P in (("dark", DARK), ("light", LIGHT)):
        body, height = fig_hero(P)
        render("hero", P, theme, int(round(height)), body)
    print(f"{'hero':16s} 396px")
    for name, fn, kick, title, sub, forked, accent in FIGS:
        for theme, P in (("dark", DARK), ("light", LIGHT)):
            body, height = fn(P)
            height = int(round(height))
            render(name, P, theme, height,
                   chrome(P, height, kick, title, sub, SOURCE[name], forked, accent) + body)
        print(f"{name:16s} {height}px")


if __name__ == "__main__":
    main()
