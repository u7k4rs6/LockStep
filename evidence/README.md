# Why these files are committed

`evidence/` holds the artifacts behind every number in the top-level README, one
per claim, each carrying its `env.lock` and the engine revision that produced it.

**This is not build output, and committing it was a fix rather than an
oversight.** The divergence report tells a reader to run `lockstep replay
evidence/case-0003.json`. For most of this project that command named a file
under a gitignored directory and an executable nobody had written, so from a
fresh clone the reproduce line pointed at nothing. The provenance argument this
repository makes depends on `lockstep replay evidence/case-witness.json` working
immediately after `git clone`, which it cannot if the artifact is generated
rather than committed.

The general rule that generated files do not belong in version control is
correct, and it applies here to `results/`, which stays gitignored. That is the
bulk output: hundreds of files, superseded on every run, no published claim
pointing at any of them. Promotion from `results/` to `evidence/` is deliberate,
one artifact at a time, through `python3 -m report.publish`, which refuses
anything without an environment tuple.

Total size is around 330 KB of JSON.

## The report's fonts are vendored for the same kind of reason

`report/fonts/` holds four woff2 subsets, about 32 KB, inlined as base64 when the
report is built. A CDN would be smaller in the repository and wrong for the
artifact: `results/report.html` is deliberately one self-contained file that
opens from `file://` with no network, so it renders identically on a machine that
has never reached Google Fonts. The OFL license texts are committed beside the
binaries, as that license requires. `fonttools` is not a dependency; the
regeneration recipe is in `report/fonts.py` and runs in a throwaway environment.

## What is here

Run `python3 scripts/verify_no_gpu.py` to check all of it without a GPU: every
file parses, carries an environment tuple, and hashes to a printed digest, and
`case-0003.json` rebuilds into a repro whose arithmetic is visible without
running a kernel.
