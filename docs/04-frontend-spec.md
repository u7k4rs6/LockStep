# Lockstep: Frontend Spec

Version 0.1, August 2026. Read alongside `01-PRD.md`.

## 0. What "frontend" means here

Lockstep has no web application. It has three user-facing surfaces, and they matter more than usual because the PRD's primary user reads the artifact for 60 seconds and forms a judgment.

| Surface | Audience | Job |
|---|---|---|
| **CLI** | The developer, daily, plus anyone reproducing a result | Run campaigns, print findings, emit minimized repros |
| **HTML report** | Hiring managers, engineers evaluating the certifier | Make the three headline numbers legible and hard to dismiss, in one scroll |
| **README claims table** | Everyone, first contact | State falsifiable claims with their environment scope |

No JavaScript framework. No build step. The report is a single self-contained HTML file emitted by a Python generator, with inline CSS and, at most, a few dozen lines of vanilla JS. It must open from `file://` with no server and no network, because the person evaluating it will double-click it.

## 1. CLI

The CLI is the product for anyone who actually uses this. Design it first.

### 1.1 Command surface

```
lockstep run          Run a workload under a named policy
lockstep verify       Run the metamorphic suite, print pass/fail per relation
lockstep fuzz         Run a fuzz campaign, print coverage and findings
lockstep replay       Re-execute a recorded (workload, schedule, seed) triple
lockstep minimize     Reduce a failing case to a minimal repro
lockstep mutate       Run the mutation campaign, print the score table
lockstep certify      Point the black-box suite at an endpoint
lockstep bench        Produce the throughput table
lockstep report       Generate the HTML report from result artifacts
```

Every command that produces a claim writes a JSON artifact to `results/` with `env.lock` embedded. `report` consumes only those artifacts, never live state, so the report is always reproducible from committed data.

### 1.2 Output rules

- **Default output is quiet and factual.** One line per meaningful event, not a progress narrative
- **Failures lead with the repro command**, not with a stack trace. The first thing printed on a divergence is the exact `lockstep replay` invocation that reproduces it
- **Never print a percentage without its denominator.** `47/50 mutants detected` beats `94%`
- **Never print a bitwise claim without the environment tag.** Every summary line ends with a short `env` fingerprint
- **Progress is a single updating line**, not a scrolling wall. Fuzz campaigns run long; a wall of output makes the interesting line scroll away
- Color is used for exactly three states and nothing else: pass, divergence, and boundary-condition-hit. Respect `NO_COLOR`

### 1.3 The divergence report

This is the most important thing the CLI ever prints. It is what gets pasted into a GitHub issue.

```
DIVERGENCE  req=r02  position=131  first differing byte=0x1a2e

  expected (canonical)   logits[131] sha256:8f3a…c1
  observed (fuzzed)      logits[131] sha256:2b7d…09

  trigger    cache_hit(len=64) with block_size=64
  schedule   12 events, minimized from 847
  env        sm_89 / cu12.4 / triton 3.x / torch 2.x

  reproduce
    lockstep replay results/2026-08-14/case-0031.json
```

Constraints: fits in 80 columns, fits on one screen, contains no information the reader must scroll to find, and is directly pasteable into an issue without editing.

## 2. HTML report

### 2.1 Design brief

Subject: bitwise equality across thousands of adversarially scheduled executions. The world this comes from is instrumentation and plotter output, not dashboards. The single job of the page is to make a skeptical reader believe three numbers.

Deliberately avoided, because they are the current defaults rather than choices: warm cream with a high-contrast serif and terracotta accent; near-black with one acid accent; broadsheet hairlines with dense newspaper columns. The last is the most tempting here and the most templated.

### 2.2 Tokens

**Color.** Cold plotter paper, ink, and two signal colors. Signal colors are used only where they carry meaning, never decoratively.

```
--paper       #EDF0F3   cold gray-blue substrate, plotter stock
--paper-deep  #DDE3E9   recessed panels, table zebra
--ink         #14181F   primary text
--ink-soft    #5A6472   labels, captions, secondary data
--signal      #B54A00   boundary condition hit. Burnt amber
--divergence  #C11B2F   bitwise inequality. Used sparingly, high impact
--lock        #1E5A52   invariance held. Deep teal, deliberately quiet
```

Rationale for the quiet green: the passing state is the common state and must not shout, so the eye lands on amber and red. If everything passes, the page should read as calm and dense rather than celebratory.

**Type.**

- Display: **Archivo** at an expanded width axis, heavy weight, tight tracking. Wide engineering signage, not editorial serif
- Body: **Instrument Sans**
- Data and code: **IBM Plex Mono**, which carries the systems vernacular honestly and is what hashes and schedules must be set in

Scale: 64 / 40 / 24 / 17 / 13 px. Display used at exactly two sizes and nowhere else. All numerals tabular.

**Layout.** Single column, max width 960px, generous vertical rhythm. Three claim blocks stacked, each: claim sentence in display, the number, then the measurement method in mono, then the evidence. Method sits above evidence deliberately, because the method is what makes the number credible.

### 2.3 Signature element: the invariance strip

The hero is not a big number with a gradient. It is a dense field of ticks, one per executed schedule, laid out as a continuous band that wraps across the full page width. Each tick is a single vertical rule, roughly 2px wide.

- Invariance held: `--lock`, low contrast, nearly texture
- Boundary predicate hit: `--signal`, slightly taller tick
- Divergence: `--divergence`, full height, breaking the band's top and bottom edges

At thousands of runs this reads as a barcode or a seismograph trace where nothing happened, which is exactly the claim. If a divergence exists, it is the only thing on the page that breaks the band, and it is unmissable. Hovering a tick reveals the schedule summary; clicking copies the `lockstep replay` command.

This is the one place boldness is spent. Everything else on the page stays quiet and tabular.

### 2.4 Sections, in order

1. **Claim strip.** One sentence stating what Lockstep is, one stating what it does not claim. The environment tuple, in mono, immediately visible without scrolling
2. **The invariance strip** with its three counts
3. **Claim 1, invariance under adversarial scheduling.** Coverage table: lifecycle n-gram percentage with denominator, boundary predicate checklist as a literal checklist with hit values, preemption depth histogram
4. **Claim 2, harness power.** Mutation table: one row per mutant, columns for operator, detected by which relation, time to detection. Survivors at the bottom with their written classification inline, not hidden behind a link
5. **Claim 3, cost of determinism.** Five-bar comparison, ratios only, with the workload trace named and linked
6. **Certification findings.** One card per external divergence with its minimized repro in mono and the upstream issue link
7. **What this does not claim.** Not a footnote. A full section with the same type treatment as the others

### 2.5 Copy rules

- Name things by what the reader controls or observes, never by internal module names
- No adjectives on numbers. "47 of 50 detected" not "an impressive 47 of 50"
- Every claim sentence is falsifiable as written. If a sentence cannot be proven wrong, rewrite it
- Empty states are directions, not apologies. If no external divergence was found, the section says what was searched and how thoroughly, with the coverage numbers, and states plainly that the deterministic modes held under those conditions. A negative result presented confidently reads as competence; the same result presented apologetically reads as failure

### 2.6 Quality floor, unannounced

Responsive to 375px, where the invariance strip wraps to more rows rather than shrinking ticks below 2px. Visible keyboard focus. `prefers-reduced-motion` respected, which mostly means the strip does not animate in. Prints cleanly to PDF, because someone will attach it to an email. Passes contrast on all text. No external font requests at runtime: subset the three faces and inline them as base64, so the file works offline from `file://`.

## 3. README claims table

The first thing anyone sees. Structure, in order:

1. **One sentence** describing what this is, including the word "certified" and the names vLLM and SGLang
2. **The differentiator sentence**, verbatim from the PRD: GRIEF fuzzes live servers with wall-clock traces, so repro is probabilistic; Lockstep's execution is a pure function of (workload, schedule, seeds), so every finding minimizes to an exact replay
3. **The claims table.** One row per invariant (I1 to I4, F1). Columns: ID, statement, how verified, current status, environment scope
4. **The three headline numbers**, each linking to the script that reproduces it
5. **What this does not claim**, before any installation instructions. Placing it high is a credibility move, not modesty
6. **Prior art**, cited generously and specifically. Thinking Machines, vLLM, SGLang, GRIEF, LLM-42. Named with links, not a bibliography dump
7. Install and quickstart, ending with a single command that reproduces claim 1 from scratch

No badges beyond CI status. No logo. No architecture diagram above the claims table; if a diagram appears at all it sits below, because the claims are the product and the architecture is an implementation detail.
