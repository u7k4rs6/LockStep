# Seven panels, where each one goes

Copy `docs/img/` into the repo. Regenerate with `python3 docs/img/build_figures.py`.
Open `preview.html` to see all seven in both themes with a toggle.

Every block below goes **above** the table it summarises, and that table moves
into a `<details>` underneath. An SVG number is not greppable, not diffable in a
PR, and not readable by a screen reader past the `<desc>` each panel carries.

---

**1. Under `## The three numbers`**

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/three-numbers-dark.svg">
  <img alt="65 of 65 relation runs bitwise identical, 10 of 10 seeded faults killed with none equivalent, and lockstep invariant at 5.2 to 5.7 times the wall time of vLLM batch-invariant in eager mode." src="docs/img/three-numbers-light.svg" width="840">
</picture>
```

**2. Under `## Claims`, after the `env.lock` qualification, above the table**

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/claims-dark.svg">
  <img alt="Five claims with what verifies each. I1 batch invariance, I2 schedule invariance, I3 replay determinism and I4 RNG isolation all hold. F1 fidelity passes 7 of 7 bounds against an fp64 CPU reference." src="docs/img/claims-light.svg" width="840">
</picture>
```

**3. Above the six-configuration wall-time table**

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/throughput-dark.svg">
  <img alt="Wall time for six configurations, run A solid and run B faded, and the 15 percent kill criterion drawn as a scale: the eager comparison at 14.6 and 16.6 percent straddles the bar, the graphed comparison at 10.6 and 11.5 percent sits below it." src="docs/img/throughput-light.svg" width="840">
</picture>
```

The kill criterion is inside this panel deliberately. It currently sits two
`<details>` deep, so the numbers a reader actually looks at say nothing about it.

**4. Above the `Claim 2, before and after the third observer was added` table**

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/mutation-dark.svg">
  <img alt="Ten mutation operators. Nine killed by the invariance relations and F1 alone, with the reversed split-combine fold surviving as O10. Ten killed once golden bytes are added as an observer outside the process." src="docs/img/mutation-light.svg" width="840">
</picture>
```

Worth hoisting this and its table out of the kill-criterion `<details>` they
live inside. Claim 2 is a headline number filed under a collapsed heading about
claim 3.

**5. Under `## Coverage, with the denominator it is actually against`**

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/coverage-dark.svg">
  <img alt="19 of 25 two-grams and 43 of 79 three-grams reached, split between the swarm campaign and the eviction campaign, with probe-only cells outlined and never-reached cells hatched. The denominator was corrected from 27 and 84." src="docs/img/coverage-light.svg" width="840">
</picture>
```

**6. Under `## The thesis, demonstrated on its author`, after the opening paragraph**

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/thesis-dark.svg">
  <img alt="Eighteen tiles for seventeen findings, placed by finding index and laned by who found them: eight from this project's own machinery, five from an outside audit, one from an anomaly, two from readers not looking for defects, and two from this project auditing itself later. Findings 9 to 15 came from outside the repository." src="docs/img/thesis-light.svg" width="840">
</picture>
```

**7. Under `## Certification`, above `### The first result was withdrawn, and why`**

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/certification-dark.svg">
  <img alt="Seven boundary cases against vLLM batch-invariant mode: five clean, and two diverged at 44 and 45 co-resident with logprob deltas of 4.685e-02 and 3.906e-02 and no token divergence. The max-num-seqs control restores reproducibility at 8." src="docs/img/certification-light.svg" width="840">
</picture>
```

---

## The design, and why each part of it is there

Drawn as instrument panels, because that is what the project is: a bench that
measures whether two executions stayed in step. Nothing in the chrome is
decoration.

| element | what it encodes |
|---|---|
| tick scale, left edge | everything here is a measurement against a declared scale |
| trace glyph, top right | two traces coincident or forked, so you can tell at a glance whether the panel is about agreement or divergence |
| hatch | declared but never reached. Not empty, not zero: unvisited |
| solid vs outlined | found from inside the project vs found from outside it |
| source line, footer | the artifact the panel was drawn from, so a figure points at evidence rather than making a claim of its own |

**Palette, from the four values given.** `2C2C2C` is the dark ground, `F3F4F4`
the light one and the ink on dark. `612D53` is `step`, in step: identical,
holds, killed, reached, found from inside. `853953` is `drift`, out of step:
divergence, defect, retracted, found from outside.

One thing had to give. Those two accents are 24 degrees apart in hue and both
are dark magenta, which is not enough separation to tell two states apart at
tile size. They are stretched to roughly 50 degrees for use as marks, step
toward violet and drift toward warm pink, while `612D53` and `853953`
themselves stay as the deep variants used for washes and rules.

| token | dark | light |
|---|---|---|
| ground | `#2C2C2C` | `#F3F4F4` |
| ink | `#F3F4F4` | `#2C2C2C` |
| step | `#B474B4` to `#D0A3D0` | `#6C2D6C` to `#9A4C9A` |
| step, deep | `#612D53` | `#612D53` |
| drift | `#E87D8F` to `#F2ABB6` | `#93304A` to `#B85070` |
| drift, deep | `#853953` | `#853953` |
| slate | `#8A8A8A` | `#7A7B7B` |
| grid | `#3D3D3D` / `#4F4F4F` | `#DEDFDF` / `#BFC0C0` |

Two magentas will not survive being asked to carry meaning on their own, so
they are not asked to. Everywhere step and drift appear together there is a
redundant non-colour cue: tick against cross, solid fill against outline, hatch
for never-reached, and lane position in the thesis panel. Turn the figures
greyscale and every one of them still reads.

Type is mono-forward: mono for every number, label, eyebrow and footer, sans
only for prose. That is the subject's own vernacular, and it means no web font
is needed, which matters because GitHub blocks external font loads inside SVG.

## Before you trust a panel over an artifact

`build_figures.py` holds an `N` dict that duplicates values already in
`README.md` and in `evidence/`. That is a third copy of every published number,
and row 16 is what happens when two copies disagree quietly. A
`tests/test_figure_constants.py` asserting `N` against `evidence/index.json`
turns these into a checked surface rather than a wider declared one. Until it
exists, the artifacts are authoritative and the panels are illustration.

Two figures reach past `evidence/` for their numbers and should be wired
carefully: `thesis` derives its counts from the README table, which
`tests/test_thesis_table.py` already parses, so point both at the same list.
`throughput` takes the kill criterion from `docs/kickoff/01-PRD.md` section 11.
