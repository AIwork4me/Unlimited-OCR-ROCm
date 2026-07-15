# Design — README/BENCHMARK textfix refresh + issue #66 engagement record

- **Date:** 2026-07-15
- **Status:** Approved (direct execution — no separate writing-plans cycle)
- **Branch:** `docs/readme-textfix-refresh-issue66`

## Context

PR #62 (merged, HEAD `3cbb6b6`) shipped the text-EditDist root-cause investigation
+ the NonText-marker strip fix + the `eval/results/pytorch-v1.6-textfix__*.yaml`
manifest (Overall **92.4506**, round-3-first; re-verified **bit-for-bit** on
2026-07-15 by re-running the full scorer). But the user-facing docs were **not**
refreshed:

- `README.md`, `README_CN.md`, `docs/BENCHMARK.md` still cite Overall **92.431**
  (the v1.3.0 number) and raw `overall_notebook` **92.436**.
- Worse, `README.md:66` / `README_CN.md:66` decompose the gap with the
  **superseded + partially-disproven** heuristic from
  `moderate-tail-attribution-2026-07-11.md` (~35% inline-math + 25/25/15).
  PR #62's evidence-based attribution (`docs/parity/text-editdist-rootcause-2026-07-12.md`)
  is the current truth: ~65% content divergence + ~14% math-style residual
  (CDM 0.958 ⇒ math correct) + ~8% dense over-gen ≈ **~80% inherent**. The 35%
  inline-math figure is explicitly disproven (actual math_residual = 13.8%).

Separately, this session posted our reply to `baidu/Unlimited-OCR` issue #66
(comment 4976675225), which is currently unrecorded in our fork.

## Changes

### 1. `README.md` + `README_CN.md` — headline + decomposition refresh

- Headline **92.431 → 92.451** (round-3-first); raw `overall_notebook`
  **92.436 → 92.4506**. Touch points: README.md {8, 44, 58, 62, 64, 80, 83};
  README_CN.md {8, 44, 58, 62, 64, 80, 83}.
- Metric-table row → `Overall 92.451 · TextEdit 0.087 · CDM 95.831 ·
  TEDS 90.221 · TEDS_s 93.38 · Read-order 0.1435` (TextEdit/CDM unchanged;
  values taken from the textfix manifest).
- `"+0.465 vs 91.97"` → `"+0.481"` (92.451 − 91.97).
- Line 66 decomposition paragraph: **replace** the 35/25/25/15 heuristic with
  the evidence-based attribution — **~80% inherent** (~65% recognition
  divergence + ~14% math-style residual [CDM 0.958 ⇒ math correct] + ~8% dense
  over-gen), ~6% fixable (NonText strip applied, +0.0196 Overall; looping retry
  reverted); block-level 77.9% of 11,219 blocks <0.05, 66.1% exact; re-link to
  `text-editdist-rootcause-2026-07-12.md`; note the one open variable
  (checkpoint revision) and link the new upstream engagement doc.

### 2. `docs/BENCHMARK.md` — headline + raw (speed block stays honest)

- Line 15 headline **92.431 → 92.451**; line 17 → 92.451; line 19 raw
  **92.436 → 92.4506**.
- Speed code-block (line 55) **keeps** `Overall = 92.431` — that is the actual
  timed v1.3.0 leaderboard run (`wall_s = 7840`); the textfix re-score did not
  re-measure speed (`wall_s: 0` in its manifest). Add a one-line note that
  accuracy has since bumped to 92.451 via the NonText strip (PR #62,
  postprocess-only, no speed impact) and link the textfix manifest. Keeping the
  timed run's real number is the honest choice over retro-editing it.

### 3. New `docs/upstream/issue-66-text-gap-engagement-2026-07-15.md`

Follows the existing `docs/upstream/` tradition (baidu-amd-rocm-issue.md,
sglang-rocm-enablement.md). Records: issue #66 is our methodology request
(text EditDist 0.087 vs ~0.042); the 2026-07-13 reply from `kushdab` is
**non-official** (`author_association=NONE`, community member — no maintainer has
replied), ruling out the `<image>` prompt format; our independent byte-identity
verification (rendered chat template ≡ README `model.infer` example); our
2026-07-15 reply (comment 4976675225) sharing the localization; the open
question to `@MurphyYin` (is `84757cb0` the 93.92 checkpoint? any eval-config
delta?); status = awaiting maintainer; checkpoint revision is the sole open
variable (HF API unreachable from this env, can't self-check for a newer one).

### 4. Housekeeping

- Remove junk untracked files: `5`, `7` (old plan-doc fragments), `err.txt`
  (stale `gh` error).
- Remove the stray symlink `vllm-venv -> /root/vllm-venv`; add `*-venv` to
  `.gitignore` to prevent recurrence.
- **Leave** the legit untracked docs (`HANDOFF-vllm-rocm-2026-07-10.md`,
  amd-doc-parsing plan/spec, `RECOVERY.md`) untouched — out of scope / may belong
  elsewhere.

## Verification

- The three edited docs: headline reads **92.451**, raw **92.4506**; the only
  remaining `92.431` is BENCHMARK's speed-block timed-run reference (with the
  clarifying note) — intentional.
- Decomposition paragraphs cite the evidence-based split and link
  `text-editdist-rootcause-2026-07-12.md` (not the superseded moderate-tail doc).
- Metric-table row matches the textfix manifest exactly.
- New upstream doc renders; `git status` shows only intended changes.
- `gh` push succeeds; PR opened; CI green (docs-only, no code or schema change).

## Non-goals

`ROADMAP.md` (out of scope); a 2026-07-15 re-verification append to the parity
doc (not selected — reproduce commands already exist there); the legit untracked
docs; any code/test change.

## Commit strategy

Two commits on `docs/readme-textfix-refresh-issue66`: (1) this spec; (2) the
implementation (doc edits + new upstream doc + housekeeping). Then push + open a
PR against `main` (never push directly to main).
