# Design: Phase 1 Addendum — Text-Repetition Fix & 一测一版一存 Versioning

- **Date:** 2026-07-03
- **Status:** Approved (brainstorming complete) — pending implementation plan
- **Owner:** aiwork4me
- **Parent spec:** [`2026-06-25-unlimited-ocr-rocm-top-tier-design.md`](2026-06-25-unlimited-ocr-rocm-top-tier-design.md) (Phase 1 — Evidence Engine)
- **Scope:** Addendum. Extends the parent's Phase 1 with two net-new deliverables that surfaced after the parent was approved, and resolves six concrete gaps found while standing the project up on a fresh AMD host.

---

## 1. Why this addendum

The parent spec (2026-06-25) approved Phase 1 "Evidence Engine" and remains the governing design. Roughly 80% of the eval harness and the real PARITY numbers already exist (`src/rocm_ocr/omnidocbench.py`, `docs/PARITY.md` with v1.5 + v1.6 measured at Overall **92.04** gundam). Two things changed after 2026-06-25:

1. **Issue #55 (2026-06-30)** showed the ~1-point accuracy gap (92.04 vs the 93.92 self-report) — previously treated in `PARITY.md` as *inherent* looping pages — is **fixable**. Fixing it becomes a first-class Phase 1 deliverable (the headline accuracy + speed win).
2. The project requires **"一测一版一存"** (one eval → one version → one saved artifact): every reported metric must trace to a uniquely identifiable, persisted state. `release.yml` today only does tag→PyPI; the eval→version→persist contract is missing.

This addendum scopes only the net-new work. It does **not** re-open the parent's Phase 1 design.

### Goals
- Eliminate text-repetition looping on the ~14/1651 affected v1.6 pages **without altering output on the 99% healthy pages** — preserving the "zero accuracy loss vs NVIDIA reference" thesis.
- Make every evaluation a single reproducible action that produces a persisted, traceable version, gated against regressions.
- Stand the project up on a fresh AMD host and close six concrete gaps so the eval is reproducible from a clean clone.

### Non-Goals (YAGNI red lines)
- ❌ Re-deriving the parent spec's Phase 1 design.
- ❌ README rewrite, demo, community flywheel (parent Phase 1; out of scope here).
- ❌ SGLang-on-ROCm enablement (Phase 2 upstream deliverable).
- ❌ Competing on raw OCR-accuracy SOTA (the upstream model's job, not ours).

---

## 2. Text-repetition fix (issue #55)

**Deciding constraint.** The project's credibility rests on "AMD output ≡ NVIDIA output." Any anti-repetition mechanism with non-zero effect on healthy pages breaks that claim. This is the criterion that picks the approach.

**Root cause (issue #55).** `SlidingWindowNoRepeatNgramProcessor(no_repeat_ngram_size=35, ngram_window=128)` bans only 35-grams seen in the last 128 tokens. Repeating units shorter than 35 tokens (e.g., a 10-token Chinese phrase) escape detection because the 35-gram window boundaries never align with the repeating-unit boundary. Affects ~14/1651 v1.6 pages in `base` mode and ~4/1651 in `gundam` mode; produces 8K–80K chars of pure repetition, 100–738 s/page, and OOM risk. Backend-agnostic — identical on CUDA and ROCm.

**Chosen approach — detection + targeted intervention (default-on).**

A lightweight streaming repetition detector observes the generated token stream:
- Tracks short-ngram (≈3–12 token) recurrence frequency and/or recent-token entropy within a sliding window.
- Fires **only** when a short repeating unit is detected looping (same k-gram recurring > N times within the window, or entropy collapse) — i.e., the model has entered a degenerate loop.
- On firing, applies a **targeted intervention at that point only**: terminate generation (emit the correct prefix up to loop onset), or — if measurement shows truncation loses too much post-loop content — a one-shot strong penalty to break the loop and continue.

**Why this approach.** On the ~99% healthy pages the detector never fires, so output is byte-identical → the parity thesis is preserved exactly. On the ~1% looping pages the output was already 100% garbage (pure repetition) and ruinously slow; truncating at loop onset yields the correct pre-loop prefix (large edit-distance improvement) and removes the 100–738 s/page slowdown and OOM risk entirely — a strict net win.

**Implementation surface.** New module under `src/rocm_ocr/` (e.g., `repetition_guard.py`) consumed by `infer.py` / `infer_async.py` behind the existing infer path. Configurable thresholds; defaults tuned by ablation on the ~14 looping pages (variables: detect threshold N, window size, truncate-vs-break-and-continue).

**Regression guard.**
- **Byte-parity test:** run a fixed healthy-page corpus through infer with the guard on vs off; assert identical output (the guard must be a no-op on healthy pages).
- **Looping fixture:** the ~14 known looping pages become a regression fixture — assert looping eliminated, output sane, per-page time bounded.
- **Full OmniDocBench v1.6 re-run:** confirms Overall rises toward 93.92 with no module regression on healthy content.

**Optional escape hatch.** An opt-in `--repetition-penalty` flag (gentle `repetition_penalty ≈ 1.05`) for users who accept a small determinism trade-off for more aggressive loop prevention. Off by default to preserve byte-parity; documented in `docs/TUNING.md`.

---

## 3. 一测一版一存 — versioning & persistence

**Contract decoded.** "一测一版一存" = traceability: every reported metric ↔ exactly one identifiable version ↔ persisted artifacts. No ephemeral or untraceable numbers.

**The versioned artifact — eval manifest.** A structured YAML committed to the repo at `eval/results/<version>__<shortsha>__<date>.yaml` capturing:
- `git`: commit SHA, dirty flag, branch, tag.
- `model`: id (`baidu/Unlimited-OCR`), weights revision/hash, dtype (BF16), image_mode (gundam/base), repetition-guard settings.
- `dataset`: OmniDocBench version (v1.5 / v1.6), data branch + commit, page count.
- `env`: ROCm version, torch version + wheel tag, GPU (gfx1100 ×N), seed.
- `metrics`: per-module (text EditDist, table TEDS / TEDS-structure, reading EditDist, formula EditDist + CDM) and composite Overall, for each of v1.5 and v1.6.
- `timing`: tok/s, total wall time, per-page time stats, looping-pages count before/after the fix.
- `predictions_ref`: pointer to the raw predictions artifact (Release asset) for re-scoring.
- `timestamp`, `run_by`, `hardware_fingerprint`.

**Persistence.**
- **Manifest:** in-repo (small, reviewable, diffable — itself the public "evidence trail" the Evidence Engine thesis wants).
- **Raw predictions** (~hundreds of MB for 1651 pages × 2 versions): attached as a **GitHub Release asset** under the tag (not committed; keeps the repo lean). Per-module metric JSON optionally via Git LFS if asset-size caps bite.

**Flow — one command (`make eval-release`).**
1. Run full eval (v1.5 + v1.6) → predictions + metrics.
2. Generate manifest (auto-filled from git + env).
3. **Regression gate:** block if composite Overall regresses > 0.3, any module regresses beyond tolerance, or looping-pages count increases vs the last manifest. Otherwise continue.
4. Bump version (or use provided); commit manifest; create an annotated git tag whose message carries the metrics summary.
5. Push tag → existing `release.yml` publishes to PyPI and (extended) attaches manifest + predictions to the Release.

**Tag scheme.** Distinguish `eval/<ver>-<date>` (evaluation snapshots; do **not** trigger PyPI) from `v*` (formal releases; trigger `release.yml` → PyPI). Not every evaluation must become a PyPI release — this avoids release noise while still giving every eval a persisted tag + manifest.

**Credentials.** The push/tag step is the only place a GitHub credential is required. The token pasted into chat on 2026-07-03 is compromised and **must be rotated** before any push; auth via `gh auth login` (interactive) so the token never re-enters plaintext. `gh` is not yet installed on the host.

---

## 4. Six gap resolutions

| # | Gap | Resolution |
|---|-----|-----------|
| 1 | `configs/unlimited_rocm.yaml` referenced by `PARITY.md` but absent (no `configs/` dir) | Add it **in-repo**, modeled on OmniDocBench `configs/end2end.yaml`, wired to the `rocm_ocr.omnidocbench` adapter and the predictions dir. Makes the scoring step reproducible from a clean clone. |
| 2 | SGLang contradiction: `BENCHMARK.md` claims "SGLang (Triton), 56 tok/s"; `PARITY.md` says SGLang serving is broken on ROCm for this model and the direct path is used | Phase 1 standardizes on the **direct `model.infer` path** (the source of the real 92.04). **Correct `BENCHMARK.md`** to stop over-claiming SGLang (annotate as a Phase 2 target). Actual SGLang-on-ROCm enablement is a Phase 2 upstream deliverable. |
| 3 | ROCm wheel version undecided (`setup_rocm.sh` defaults to 6.2; host is 7.2.1; `BENCHMARK.md` claims torch 2.12.1+rocm7.2) | Prefer a **rocm7.x wheel to match the host** (validate the claimed torch 2.12.1+rocm7.2); confirm via a GPU matmul smoke test in SP0. If unavailable/failing, fall back to the rocm6.2 wheel + `HSA_OVERRIDE_GFX_VERSION=11.0.0`. **Pin and document** the winning combo. |
| 4 | v1.5 CDM still broken (pending in `PARITY.md`) | Port the v1.6-verified fix (ImageMagick `magick` / IM6 symlink + TeX Live + Ghostscript) to the **v1.5 scorer** path (older code); verify CDM > 0 on v1.5. |
| 5 | Host cannot reach huggingface.co (curl timeouts) | Use **ModelScope** for model weights download (repo already ships `model_scope_demo/`). Keep HF as the documented default path; ModelScope as the host/fallback path. |
| 6 | Host is clean (no torch/transformers/sglang/etc.) | SP0 runs `scripts/setup_rocm.sh` (with the wheel from #3) + ModelScope model + a single-page infer smoke test + installs `gh` and completes secure auth. |

---

## 5. Execution order & acceptance

**Order (dependency-driven):**
1. **SP0 — host bring-up** (gaps 3/5/6 + `gh` install + token rotation). Exit: one-page GPU infer correct; matmul / `rocm-smi` healthy; `gh` authed.
2. **Reproduce baseline** on this host via the existing harness → reproduce v1.6 Overall ≈ **92.04** (gaps 1, 4, 5). Also emits the **first eval manifest**, validating the 一测一版一存 pipeline itself.
3. **Repetition fix** (§2): implement detection + targeted intervention; ablate on the ~14 looping pages. Exit: looping pages = 0; healthy-page output byte-identical (regression test); Overall rises toward 93.92.
4. **Versioning** (§3): extend `release.yml`, add `make eval-release` + gate + in-repo manifest + Release-asset predictions.
5. *(Optional)* full v1.5 run + post-fix re-measurement.

**Acceptance (Done) for this addendum:**
- One command on this host reproduces the v1.6 eval and emits a manifest; Overall ≥ 92.04 and looping pages = 0.
- Repetition guard on by default, zero effect on healthy pages (regression test + full-corpus diff prove it); opt-in penalty flag available.
- 一测一版一存 pipeline live: eval → manifest → gate → tag → Release; regressions blocked.
- `configs/unlimited_rocm.yaml` in repo; `BENCHMARK.md` SGLang wording corrected; ROCm wheel pinned + documented.

---

## 6. Decisions log
- **2026-07-03** — Repetition fix = **detection + targeted intervention** (default-on); opt-in `repetition_penalty` as escape hatch. Chosen to preserve byte-parity on healthy pages.
- **2026-07-03** — Versioning = **one-command + strict gate**; manifest in-repo, predictions as Release asset; `eval/*` vs `v*` tag split.
- **2026-07-03** — SGLang deferred to Phase 2; Phase 1 = direct `model.infer` path.
- **2026-07-03** — ROCm wheel pinned empirically in SP0 (prefer rocm7.x; fall back rocm6.2 + `HSA_OVERRIDE_GFX_VERSION=11.0.0`).
- **2026-07-03** — Model source = ModelScope on this host (HF unreachable); HF remains the documented default.
- **2026-07-03** — GitHub token (pasted into chat) treated as compromised → rotate before any push; auth via `gh`.

---

## 7. Open items (resolved during planning / SP0)
- ROCm wheel pin (smoke-test-determined).
- Detection thresholds + truncate-vs-break-and-continue (ablation on the ~14 looping pages).
- Predictions storage final choice (Release asset default; LFS if asset-size caps bite).
- Inspect `src/rocm_ocr/omnidocbench.py` and `tests/test_omnidocbench.py` during planning to reuse existing metric emission rather than duplicate it.

---

## 8. Relationship to the parent spec

This addendum is a child of `2026-06-25-unlimited-ocr-rocm-top-tier-design.md`. It adds deliverables to that spec's Phase 1 (Evidence Engine) and changes nothing about Phases 2–3. Where the parent says "tag releases" and "CI runs an OmniDocBench sample every release" as anti-bitrot, this addendum formalizes the stronger **一测一版一存** contract (every eval → manifest → gated tag → Release asset). Implementation proceeds on the `phase1/repetition-fix-and-versioning` branch; per the parent's commit strategy, the spec is committed as an isolated commit.
