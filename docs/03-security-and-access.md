# Lockstep: Security and Access

Version 0.1, August 2026. Read alongside `01-PRD.md` and `02-technical-architecture.md`.

This is a single-developer research artifact with no multi-tenant deployment, no user accounts, and no persistent user data. Most of what would fill a security doc in a product context does not apply. What does apply, and is treated seriously here, is the threat model of a project whose entire value proposition is *trustworthy measurement*, plus the disclosure ethics of pointing an adversarial harness at other people's open-source projects.

## 1. Trust boundaries

| Boundary | Description | Posture |
|---|---|---|
| Local engine and harness | Runs on the developer's machine, no network listener | Trusted. No hardening required |
| Model weights | Downloaded from Hugging Face | Pin by revision SHA. Verify checksums. Never load `.bin` pickles when a safetensors revision exists |
| Rented GPU (A100/H100) | Short-lived third-party VM used for the final table | Untrusted host. No credentials of any kind uploaded. Assume the disk and RAM are readable by the provider |
| External engine endpoints (vLLM, SGLang) under certification | Local processes the developer starts, or a rented instance | Treated as the system under test, not as a trusted party |
| Published artifacts (repo, report, filed issues) | Public | Everything here is deliberate disclosure. Review before push |

## 2. What the certifier is and is not allowed to do

The black-box certifier sends request traces to an inference endpoint and compares outputs. That is functionally a load and concurrency test. Rules:

1. **Only against endpoints the developer controls.** Localhost, or a rented instance the developer started. Never against a hosted API, a shared cluster, a university machine, or anyone's demo deployment. Not even a "quick check".
2. **No third-party API keys in any campaign.** The certifier must refuse to run if `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or similar are present in the environment, as a hard guard against a misconfigured base URL sending a fuzz campaign to a paid hosted endpoint. Fail closed with a clear message.
3. **Rate and concurrency caps** are configuration, defaulting low, with a documented ceiling. The point is to find determinism divergences at boundary conditions, not to stress a service to failure.
4. **No workload that resembles a denial-of-service pattern** even locally, because the repo is public and someone will copy the script. Memory-pressure campaigns are in-process only, driven through the policy seam, not through request floods.

The certifier config file carries a required `i_control_this_endpoint: true` field with no default. This is a speed bump against accident, not a security control, and is documented as such.

## 3. Responsible disclosure for upstream findings

The PRD's Tier 1 success criterion is an upstream filing. That interaction has to be handled well, because it is also the most visible thing a hiring manager will find.

**Classification first.** For each divergence, decide before filing:

- **Correctness bug**: deterministic mode produces different outputs for the same logical request. Not a security issue. File a normal public GitHub issue.
- **Cross-request influence**: request A's content changes request B's output beyond documented numerical nondeterminism. This is a tenant-isolation concern. Treat as potentially security-relevant.
- **KV cache state corruption**: one request reads or is influenced by another request's KV. This is a security issue. Do not file publicly first.

**Process:**

1. Minimize to the smallest repro (harness ddmin output, under 15 lines).
2. Search existing issues first. The SGLang block-boundary corruption issue already exists; a duplicate filing reads as not having done the reading. If a finding matches an open issue, comment with the additional minimized case rather than opening a new one.
3. For correctness bugs: public issue, following the project's template, with the environment tuple from `env.lock`, the repro, expected versus observed, and the specific commit tested.
4. For anything in the cross-request or corruption categories: use the project's private security reporting channel (GitHub Security Advisories for vLLM and SGLang), wait for maintainer guidance, and do not publish the repro in the Lockstep repo or in any writeup until they say it is fine or a fix ships. Standard 90-day fallback, but with genuine flexibility since these are volunteer-maintained OSS projects.
5. Never frame a finding as an exploit or a vulnerability unless it actually is one. Overclaiming here damages credibility permanently and is the single easiest own-goal in this project.

**Tone.** These maintainers shipped the feature being tested. The framing is "your deterministic mode is good enough that I had to go to boundary conditions to break it, here is exactly where", not "I broke your engine". Cite their work in the README before filing anything.

## 4. Repo hygiene

- No secrets in the repo. Pre-commit hook running `gitleaks` or equivalent
- `.env` gitignored; a committed `.env.example` documents required variables
- Rented-instance SSH keys are per-session, generated fresh, and revoked after. Never a long-lived personal key on a rented box
- Model weights and fp64 reference tensors are gitignored, with a download script and checksums committed instead
- Fuzz corpora and crash artifacts: committed only after review, because a minimized repro is a prompt plus a schedule, and prompts from a public dataset are fine while anything typed by hand should be checked for accidental personal content
- The `env.lock` emitted into artifacts records versions and GPU model. Confirm it does not include hostname, username, or absolute home paths. Strip them in the emitter

## 5. Supply chain

- `uv` with a committed lockfile. No unpinned installs
- Direct dependency set kept small and named in the architecture doc: torch, triton, transformers (loading and tokenizer only), numpy, pytest, hypothesis. Every addition beyond this is a deliberate decision
- Triton kernel configs live in a committed registry. A change to that registry is a claims-affecting change and requires re-running the invariance suite before publishing any number
- Do not vendor code from vLLM, SGLang, or `batch_invariant_ops`. Cite and reimplement. If any snippet is adapted, mark it inline with the source and the upstream license, and check license compatibility (vLLM and SGLang are Apache-2.0; attribution and a NOTICE entry are required)

## 6. License and attribution

- Lockstep ships under Apache-2.0, which is compatible with the ecosystem it certifies and permissive enough that maintainers can borrow the harness ideas
- `NOTICE` credits Thinking Machines for the batch-invariance diagnosis, vLLM and SGLang for the shipped implementations, and any adapted code
- Model weights carry Qwen's license; the README states the model is downloaded by the user and not redistributed

## 7. Integrity of published claims

This is the real security property of the project. The threat is not an attacker; it is the developer accidentally publishing a number that does not hold.

Controls:

1. Every published number is produced by a committed script in `bench/` and reproducible from a committed workload trace
2. Every artifact embeds `env.lock`. A claim without an environment tuple is invalid by construction
3. CI fails if the kernel config registry changes without a corresponding claims-table review commit
4. Mutation scores publish the survivor census, including proven-equivalents with their written arguments. A bare percentage is not publishable
5. Statistical claims publish alpha, power, effect size, and sample count
6. The README carries a "what this does not claim" section: no novelty on batch-invariant kernels, no cross-hardware guarantees, no throughput superiority, no claim about untested models or configurations

## 8. Personal data

None collected, none stored, none processed. Prompts used in campaigns come from public datasets or are synthetically generated. No telemetry in the engine, harness, or report generator, and none is to be added.

## 9. Access model

Single developer, single machine, plus a public GitHub repo. No roles, no auth, no session management, no API surface exposed to a network. If the project later grows a hosted demo, this document needs a full rewrite before that ships; a certifier endpoint reachable from the internet is a materially different security posture and nothing in this document covers it.
