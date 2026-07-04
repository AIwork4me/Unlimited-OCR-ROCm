# Design: Delivery Discipline & Versioning Pipeline (一版一测一存一推送) + Repo Hygiene

- **Date:** 2026-07-04
- **Status:** Approved (brainstorming complete) — pending implementation plan
- **Owner:** aiwork4me
- **Related:** parent `2026-06-25-unlimited-ocr-rocm-top-tier-design.md` (Phase 1 — Evidence Engine); addendum `2026-07-03-phase1-repetition-fix-and-versioning-addendum.md` (§3 sketched the versioning contract; this spec makes it concrete and adds the "一推送" layer); three-backend `2026-07-03-rocm-three-backend-eval-design.md` (each backend feeds this pipeline).
- **Scope:** First of the user's sequenced workstreams: **#2 delivery discipline + repo hygiene → #1 three-backend → #3 speed / #4 deployment.** This spec covers #2 only.

---

## 1. Why this spec

The project's two hardest-won goals — *eval-data backed* and *accuracy parity with baidu/Unlimited-OCR* — are already met on the PyTorch backend (OmniDocBench v1.6 Overall **91.95** ≈ PARITY 92.04; CDM 0.957; manifest committed). What is **not** yet met is the **discipline that keeps those claims trustworthy as the project grows**: a contract that every reported metric traces to a unique, persisted, gated version, and that every change to `main` is tested before it lands.

The 2026-07-03 addendum §3 sketched this contract ("一测一版一存": one eval → one version → one saved artifact) but left the mechanism unimplemented. Two things changed since:

1. The user elevated the contract to **"一版一测一存一推送"** — adding an explicit *push* discipline: a tested version is not done until it is on GitHub, not just on a local laptop.
2. The user named **clean repo hygiene** as a first-class requirement: no unnecessary file is ever pushed.

This spec designs the concrete pipeline, gate, CI, hygiene, and auth work that realizes that contract, so that every subsequent workstream (three-backend, speed, deployment) ships safely under it.

### Host reality (binding constraint)

The 4-GPU AMD host is the only place the real OmniDocBench eval can run (~4 h, gfx1100 ×4). GitHub Actions CI runs on CPU-only `ubuntu-latest` with **no AMD GPU**. Therefore "测" is split into two layers: a fast CI layer (lint + unit tests + manifest-schema) and a heavy local layer (full eval + regression gate). Everything in this design respects that split.

---

## 2. Goals & Non-Goals

**Goals**
- Realize **一版一测一存一推送** as a single, reproducible, gated pipeline: one command on the AMD box takes a change from eval → manifest → gate → tag → Release.
- Protect the accuracy-parity thesis with a **strict regression gate** that blocks any accuracy regression from shipping (overridable only with a recorded reason).
- Keep `main` protected: every commit lands via PR with green CI.
- Keep the repo lean forever via forward-looking `.gitignore` hygiene.
- Rotate the compromised credential before any push.

**Non-Goals (YAGNI red lines)**
- ❌ Self-hosted AMD runner / fully-automated CI eval (Approach B — deferred as a future enhancement).
- ❌ SGLang/vLLM enablement (workstream #1 — next spec).
- ❌ Speed optimization (workstream #3).
- ❌ The §2 targeted text-repetition fix (separate; its looping-page count plugs *into* this gate but the fix itself is out of scope here).
- ❌ PyPI release automation changes (`release.yml` `v*` → PyPI already works; release assets are attached by the local `gh`, not by CI).

---

## 3. Architecture: two layers + one baseline

```
                    ┌─ Layer 0: Repo hygiene (baseline) ──────────────────────┐
                    │  Forward-looking .gitignore + pre-commit large-file guard │
                    │  → repo stays lean; big artifacts are never tracked       │
                    └──────────────────────────────────────────────────────────┘
 code increment ──→ Layer 1: PR / CI fast-check (per change, fast)  ──→ main
                   feat/* branch → push → open PR
                   CI (cpu): ruff + pytest + ★manifest-schema validation
                   green → squash-merge; main is branch-protected
                                 │
                                 ▼  (periodic, heavy)
                   Layer 2: eval / evidence (local 4-GPU, ~4 h)
                   make eval-release:
                     direct-path eval → manifest → strict gate (vs last manifest)
                       → pass/override → eval/<tag> tag
                       → git push --tags → gh release create (predictions.zip)
```

The eval must run locally (CI has no GPU), so the entire Layer-2 chain runs on the host and `gh release create` uploads assets directly — no CI relay, no extra infrastructure.

---

## 4. The contract — "一版一测一存一推送" decoded

| Glyph | Meaning in this project | Where it lives |
|---|---|---|
| **测** (test) | Layer-1 fast-check (every PR) + Layer-2 full-eval regression gate (every release) | CI; local `gate.py` |
| **版** (version) | A git tag = the unique identifiable version | `eval/<ver>-<date>` or `v*` |
| **存** (persist) | Manifest committed in-repo (reviewable evidence trail) + predictions as a Release asset | `eval/results/*.yaml`; Release URL |
| **推送** (push) | `git push --tags` + `gh release create` — the result lands on GitHub, not just locally | local orchestrator |

---

## 5. Components (new / changed)

| Type | File | Purpose |
|---|---|---|
| new | `src/rocm_ocr/gate.py` | Regression gate: load previous manifest, compare, apply thresholds, return PASS/BLOCK/OVERRIDE + deltas |
| new | `src/rocm_ocr/release.py` | Orchestrator: eval → manifest → gate → PR → merge → tag → `gh release create` |
| new | `eval/results/manifest.schema.json` | JSON Schema for manifests; CI validates every committed manifest against it |
| changed | `Makefile` | Add `eval-direct` (direct `model.infer` path) + `eval-release`; fix the stale `eval` comment (it points at the broken SGLang-client path) |
| changed | `.github/workflows/ci.yml` | Add a manifest-schema validation job |
| changed | `.gitignore` | Forward-looking hardening (three-backend / deploy artifacts) |
| unchanged | `.github/workflows/release.yml` | `v*` → PyPI is unchanged; `eval/*` does not match `v*` so it never fires — the local `gh` already created the Release |
| setting | GitHub `main` branch protection | Require PR + required status checks (one-time, ADMIN) |
| prerequisite | Credential rotation | Revoke the leaked full-scope classic PAT → fine-grained repo-scoped token + `gh auth login` |

**Reuse:** `eval_manifest.py` already provides `build_manifest / manifest_filename / write_manifest / capture_git / capture_env / hardware_fingerprint`; the real eval entry is `scripts/run_omnidocbench_direct.py` (4-GPU sharded via `run_omnidocbench_4gpu.sh`). The gate compares against the latest existing manifest (`eval/results/pytorch-v1.6__4f8c5eb7ea__2026-07-03.yaml`).

---

## 6. The `make eval-release` pipeline (data flow)

The orchestrator (`src/rocm_ocr/release.py`) runs:

1. **Preflight** — assert a clean git tree (so the manifest's SHA matches what was evaluated), dataset present, scorer venv present (with the CJK toolchain), model cached, `gh` authed, ROCm sees GPUs. Fail fast with a clear message otherwise.
2. **Eval** — `run_omnidocbench_direct.py` (4-GPU, gundam mode, pinned config: `no_repeat_ngram_size=35`, `ngram_window=128`, `max_length=32768`, model `revision=` pinned). Writes `{stem}.md` to a versioned `predictions/<version>/`.
3. **Score** — OmniDocBench scorer (its own py3.11 venv) → per-module metrics.
4. **Build manifest** — `eval_manifest.build_manifest()` auto-fills git/model/dataset/env/metrics/timing; add `backend`, `predictions_ref` (the URL derivable from the tag — no back-fill needed), and `compared_against` (previous manifest's tag).
5. **Gate** — `gate.evaluate(curr, prev)` → PASS / BLOCK / OVERRIDE (§7).
6. **Tag + push (manifest via PR)** — create branch `eval/<tag>` → commit the manifest → `gh pr create` → CI validates manifest schema → on green, `gh pr merge --squash` → tag the merge commit `eval/<tag>` → `git push --tags`.
7. **Release** — `gh release create eval/<tag> predictions.zip --notes-file <summary>` (the `predictions_ref` URL becomes live).
8. **Report** — print metrics, gate verdict, tag, Release URL.

**Manifest goes via PR** (not direct push): `main` branch protection has zero exceptions, the manifest-schema check is meaningful pre-merge, and any override reason is visible in the PR. The cost is a few minutes of CI after a 4-hour eval — negligible.

---

## 7. Gate rules (the "测" that protects parity)

`gate.evaluate(curr, prev)` checks each item; only a *worsening* counts as a regression (an improvement is always pass):

| Check | Direction | Default threshold | Fails when |
|---|---|---|---|
| Overall | higher better | Δ ≤ 0.3 | regression > 0.3 |
| text Edit_dist | lower better | Δ ≤ +0.005 | beyond tolerance |
| table TEDS / TEDS-structure | higher better | Δ ≥ −0.005 | beyond tolerance |
| reading Edit_dist | lower better | Δ ≤ +0.005 | beyond tolerance |
| formula Edit_dist | lower better | Δ ≤ +0.005 | beyond tolerance |
| formula CDM | higher better | Δ ≥ −0.005 | beyond tolerance |
| looping pages | fewer better | no increase | count increased |

**Speed (tok/s)** is recorded with its Δ and any regression is noted, but it **never blocks** (advisory only — hard speed thresholds belong to workstream #3).

**Verdict:**
- All pass → **PASS** → proceed.
- Any fail, no override → **BLOCK** → abort before push; print which metrics regressed + deltas; suggest fix or `--allow-regression "reason"`.
- Any fail + override → **OVERRIDE** → proceed; `{reason, regressed_metrics, deltas, by, timestamp}` is written to `manifest.gate.override` and surfaced in the Release notes + PR body. Overrides are auditable, never silent.

Thresholds live as constants in `gate.py` (tunable + versioned).

**Edge cases:**
- **No previous manifest** for a backend×dataset pair → gate is a no-op (baseline; recorded "no comparison").
- **New backend** (SGLang/vLLM first run) → the gate compares **within backend and dataset** (PyTorch↔PyTorch), **never across backends** (different backends legitimately differ). This is why the manifest carries a `backend` field.

---

## 8. Tag / version taxonomy

- `eval/<backend>-<dataset>-<shortsha>-<YYYYMMDD>` — e.g. `eval/pytorch-v1.6-4f8c5eb7ea-20260704` (matches `manifest_filename`). Does **not** trigger `release.yml` (the `v*` glob won't match).
- `v<semver>` — e.g. `v1.3.0`, formal PyPI releases; triggers `release.yml` → PyPI. The release script may attach the matching manifest + predictions to this Release as well.
- **Frequency:** every full eval → one `eval/*` tag. Evals are ~4 h and infrequent, so tag noise is acceptable. **Not every eval becomes a PyPI release** (avoids release noise).

---

## 9. CI additions (`ci.yml`)

Add a **standalone job** (single py3.12, no torch, only `jsonschema` + `pyyaml` — fast) on PR and push to `main`:
- Validate every `eval/results/*.yaml` against `manifest.schema.json`.
- Assert each committed manifest's `gate.verdict ∈ {PASS, OVERRIDE}` — a **BLOCK** verdict must never be committed (the script aborts before commit; CI double-checks).
- Assert `predictions_ref` is a well-formed derivable URL/tag.

This job is the "green" that the manifest PR (§6 step 6) waits on. The existing lint + pytest matrix job is unchanged.

---

## 10. Repo hygiene

Audit (2026-07-04): the repo is already lean — **680 KB / 86 files**, largest is `assets/Unlimited-OCR.png` (106 KB); no logs, predictions, venvs, wheels, or data are tracked. The existing `.gitignore` already covers bytecode, venvs, build/dist, `*.whl`, `eval_predictions*/`, `log/`, `*.log`, secrets, `.superpowers/`, `hf_cache/`.

**Forward-looking hardening** — add patterns for the upcoming workstreams so their artifacts can never be pushed even if a build happens inside the repo:

```
# three-backend / deploy artifacts (workstreams #1/#3/#4)
predictions/          releases/         *.zip
sglang-src/           sglang-*-venv/    vllm-*-venv/    *-rocm-venv/
docker-out/           .rocm_cache/      build_logs/     *.deb
.claude/              # local agent state (same class as .superpowers/)
```

`pre-commit`'s `check-added-large-files` (500 KB threshold) is already configured and retained as the catch-all guard.

---

## 11. Auth rotation (prerequisite for any push)

Per the host runbook, the classic PAT pasted on 2026-07-03 is full-scope and compromised. **Before the first push:**
1. Revoke the old token.
2. Create a **fine-grained PAT** scoped to `AIwork4me/Unlimited-OCR-ROCm` only, permissions `contents: write` + `pull-requests: write`, 90-day expiry.
3. `gh auth login` interactively (the token never enters plaintext in scripts or commits).
4. Verify with `gh auth status`.

Recorded in `docs/RELEASE.md` and the runbook. This is an acceptance gate, not a code deliverable.

---

## 12. Testing strategy (without a 4-hour eval each time)

- **Unit tests (CI, seconds, CPU):**
  - `tests/test_gate.py` — synthetic curr/prev manifest pairs → assert PASS/BLOCK/OVERRIDE verdicts, correct Δ computation, and every threshold boundary.
  - `tests/test_release.py` — mock `subprocess` / `gh` → assert the eval→manifest→gate→PR→merge→tag→release call order, and that BLOCK-without-override **aborts before any push**.
  - `tests/test_manifest_schema.py` — existing + fixture manifests against the schema.
- **Smoke integration (local, ~2 min, real GPU):** `make eval-release --limit 4 --smoke` runs the whole chain on 4 pages → asserts a smoke manifest is produced and the gate runs, but creates **no formal tag and no formal Release** (smoke mode short-circuits §6 steps 6–7 or tags `-smoke`). Catches integration bugs (scorer/gh wiring) cheaply.
- **Full eval (local, ~4 h):** the real `make eval-release` — run once to produce the authoritative manifest + Release. This is the **acceptance run**, not a regression test.

CI runs unit + schema tests on every PR; smoke + full eval run locally.

---

## 13. Implementation sequencing (bottom-up; each step is its own PR — dogfooding the model)

1. **Auth rotation** (prerequisite) — rotate token, `gh auth login`, verify. Unblocks all pushes.
2. **`.gitignore` hardening + `manifest.schema.json` + CI schema job** — small, isolated; the foundation.
3. **`gate.py` + `test_gate.py`** — pure logic; the core of "测".
4. **`release.py` orchestrator + `test_release.py` (mocked) + `Makefile` targets** — wires eval→manifest→gate→PR→tag→release; mocked tests.
5. **`--limit N` smoke integration** — 4 real pages; debug the full chain cheaply.
6. **`main` branch protection** — require PR + the new schema/gate status checks.
7. **First full `make eval-release`** — authoritative PyTorch v1.6 run → manifest PR → `eval/*` tag → Release with `predictions.zip`. End-to-end acceptance + contract validation.

---

## 14. Acceptance criteria (Done)

- From a clean clone, after token rotation, one command (`make eval-release`) on this host reproduces the v1.6 eval, emits a manifest, runs the gate, opens a manifest PR (schema-validated in CI), tags `eval/*`, and creates a Release with `predictions.zip`.
- The gate strict-blocks any accuracy regression beyond thresholds; override is possible only with a recorded reason; speed is recorded but never blocks.
- CI validates manifest schema on every PR; `main` is branch-protected (PR + required status checks).
- `.gitignore` covers current + anticipated artifacts; tracked repo size stays < 1 MB; the pre-commit large-file guard is active.
- Unit + schema tests are green in CI; smoke integration is verified locally on 4 pages.
- The compromised PAT is rotated; auth is via a fine-grained repo-scoped token + `gh`.

---

## 15. Decisions log

- **2026-07-04** — First workstream to design to completion = **#2 delivery discipline + repo hygiene**; sequence is #2 → #1 → #3/#4.
- **2026-07-04** — Branch/push model = **feature-branch + PR**; CI on PR; squash-merge; `main` protected.
- **2026-07-04** — Eval gate = **strict block + override**; accuracy regression blocks tag/push; override requires a recorded reason; speed is advisory.
- **2026-07-04** — Artifact storage = **GitHub Release assets** (repo stays lean; manifest's `predictions_ref` points at the asset URL).
- **2026-07-04** — Pipeline mechanism = **local `make eval-release` orchestration** (Approach A); self-hosted runner deferred.
- **2026-07-04** — Manifest commit = **via PR** (schema-validated pre-merge; `main` protection has no exceptions).
- **2026-07-04** — Gate thresholds: Overall 0.3; per-module 0.005; looping pages must not increase.
- **2026-07-04** — Tag taxonomy: `eval/<backend>-<dataset>-<shortsha>-<date>` (no PyPI) vs `v<semver>` (PyPI).

---

## 16. Open items (resolved during planning)

- Exact `gate.py` threshold constants + whether they live in code or schema (default: code constants).
- The smoke-mode tag/release short-circuit (`-smoke` suffix vs skip) — decide at implementation.
- Whether `release.py` attaches manifest + predictions to `v*` formal releases too, or only `eval/*` (default: both).

---

## 17. Relationship to prior specs

This spec is a child of `2026-06-25-unlimited-ocr-rocm-top-tier-design.md` (Phase 1 — Evidence Engine) and **realizes** §3 of the `2026-07-03` addendum (which sketched the versioning contract but did not implement it), adding the explicit **"一推送"** push discipline and the **repo-hygiene** requirement. It changes nothing about Phases 2–3. The three-backend spec (`2026-07-03-rocm-three-backend-eval-design.md`) feeds this pipeline: each backend (PyTorch/SGLang/vLLM) produces a manifest that this gate compares within its own backend. Implementation proceeds one PR per sequencing step; per repo convention, the spec itself is committed as an isolated commit.
