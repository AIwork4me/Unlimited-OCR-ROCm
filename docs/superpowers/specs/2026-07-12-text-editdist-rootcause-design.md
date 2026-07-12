# Design — Text EditDist root-cause confirmation + targeted fixes

- **Date:** 2026-07-12
- **Author:** brainstorming session (continues the v1.3.0 PyTorch fast-path work)
- **Status:** Approved (brainstorm) → pending implementation plan
- **Relates:** builds on `docs/parity/moderate-tail-attribution-2026-07-11.md` (whose heuristic "35% inline-math style" is superseded by this evidence-based finding).

---

## 1. Problem & motivation

Text EditDist is the dominant chunk of the Overall gap: ours **0.087** vs Baidu self-report **0.042** on OmniDocBench v1.6 (Overall 92.431 vs 93.92; ~1.49 gap, ~entirely text). The prior `moderate-tail-attribution-2026-07-11.md` categorized the gap heuristically (≈35% inline-math style, 25% recognition, 25% dense-layout, 15% format). **This design supersedes that with instrumented, official-scorer-口径 evidence**, then fixes the fixable causes.

### Evidence gathered this session (official scorer + per-page diffs)

- **Official text normalization is aggressive:** `normalized_text = clean_string(textblock2unicode(text))`. `clean_string` strips ALL whitespace/punctuation/markdown (`re.sub(r'[^\w一-鿿]', '', …)` — keeps only alphanumeric + Chinese); `textblock2unicode` converts inline LaTeX (`\(...\)`, `$...$`) to unicode. **Format/markdown differences are normalized away → the residual gap is genuine CONTENT difference**, not formatting. (This already weakens the old "inline-math style" attribution — the scorer converts most math to unicode.)
- **Distribution is heavy right-tail** (n=1557 text pages): median **0.020**, mean **0.087**. 50.2% excellent (<0.02), 12.7% good, 18.6% tail (0.10–0.30), 6.1% bad (0.30–0.60), 2.1% failure (≥0.60). **The mean is pulled up by ~8% bad+failure + ~19% tail.**
- **Worst-page root causes (confirmed with GT-vs-pred diffs):**
  - **Looping/repetition** — e.g. `yanbaoppt 4570` (EditDist 0.998): "他每日四场，他每日四场…". The 2 known looping pages (`yanbaoppt 4570`, `dianzishu 149`) are in the worst-12; the fast path does NOT apply the two-pass retry (a documented Task-12 follow-up).
  - **Over-generation** — newspaper pages: GT ~22–32 K chars, **pred ~102 K** (3–5×); two newspapers both ~102 K (suspicious — possibly tile-repetition or a length cap pattern). Pending Phase-1 confirmation of repetitive-vs-dense.
  - **Truncation / wrong-region** — `jiaocai 1349`: GT 1888 chars, **pred 58** (only the vertical "密封线" seal-line border, one char per line).
  - **Proportional-length divergence** — `dianzishu 80`: GT 175 ≈ pred 200, yet EditDist 0.93 (content/matching divergence).

**Headline shift:** the fixable leverage is the failure tail (looping + over-gen + truncation), NOT inline-math style. Phase 1 quantifies each; Phase 2 fixes the fixable.

---

## 2. Goals & success criteria

- **G1 (confirm).** Produce an evidence-based, official-scorer-口径 attribution of every text-EditDist root cause: per cause, page count, % of pages, and `Σedit_num / Σupper_len` mass contribution (how much of the length-weighted mean each cause explains).
- **G2 (fix).** For each cause confirmed fixable, apply a targeted fix that **recovers correct OCR content** (not metric-gaming). Each fix gated: re-score full set → text EditDist drops, the good pages (63%, edit<0.05) do NOT regress, Overall does not drop beyond the noise floor.
- **G3 (honesty).** Causes confirmed inherent (content divergence, dense over-gen) are documented as such — NOT forced/gamed.
- **G4 (ship).** Updated manifest (new text EditDist + Overall, leaderboard round-3-first 口径) + `docs/parity/text-editdist-rootcause-2026-07-12.md` with the attribution + per-cause fix outcomes.

**Non-goals:** metric-gaming (artificial truncation/format-matching to chase GT length); retraining; vLLM/SGLang (still parked/blocked); re-deriving the speed work.

### Locked decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Approach | **Full systematic — confirm every cause first (instrumented, official口径), then fix each.** |
| Levers | **Widest: decoding params + content-recovery post-process + inference-path deep-dive** (max_length / crop / tile-repetition). |
| Honesty boundary | Fixes must recover correct content (dedup looping, retry, fix tile-repetition, fix truncation path). **Never** artificial truncation/matching to game EditDist. |
| Confirmation口径 | Use the **official scorer's own** quick_match + `normalized_text` (via a dump hook) — not a re-implemented matcher. |
| Global `no_repeat_ngram_size=5` | **NEVER** (crashed Overall to 64.56). Repetition control is per-page-type / targeted only. |

---

## 3. Phase 1 — Root-cause confirmation (instrumented)

### 3.1 Dump hook (local OmniDocBench checkout, env-gated, no metric change)

Add an env-gated dump at the end of `OmniDocBench/src/metrics/cal_metric.py::call_Edit_dist.evaluate` (the local checkout at `/root/ocr-eval/OmniDocBench`, NOT our repo): when `OMNIDOCBENCH_DUMP_TEXT=1`, write the per-sample `{image_name, norm_gt, norm_pred, Edit_num, upper_len, category_type}` to `./result/<save_name>_text_pairs.json`. Default behavior (no env var) is unchanged — the EditDist computation is not touched. The hook is a measurement tool; it is documented and revertible.

Run the scorer with `OMNIDOCBENCH_DUMP_TEXT=1` on the **existing** v1.3.0 predictions (`/root/eval_predictions_fast`) — no regeneration, one ~20-min scored run with **workers=4** (workers=13 deadlocked before).

### 3.2 Categorization (`scripts/analysis/text_editdist_rootcause.py`)

For each dumped page compute: `gt_len`, `pred_len`, `edit_ratio = edit/upper`, `zlib_ratio = len(zlib.compress(norm_pred))/len(norm_pred)`, LaTeX-token residual count. Classify (thresholds are starting points, tuned against the real distribution from 3.1):

| Category | Rule | Nature |
|---|---|---|
| good | edit_ratio < 0.05 | — |
| **looping** | zlib_ratio < 0.05 and pred_len > 3000 | fixable (retry) |
| **over-gen repetitive** | pred_len > 2×gt_len and zlib_ratio < 0.20 | fixable (path/control) |
| over-gen dense | pred_len > 2×gt_len and zlib_ratio > 0.30 | inherent |
| **truncation** | pred_len < 0.4×gt_len and gt_len > 300 | investigate path |
| math residual | post-clean norm contains LaTeX tokens (`\`, `^`, `_`) asymmetric gt↔pred | partial (align converter) |
| content divergence | otherwise, edit_ratio ≥ 0.05 | inherent |

### 3.3 Quantification

Per category: page count, % of pages, `Σedit_num / Σupper_len` (mass contribution to the length-weighted mean). Output the attribution table to `docs/parity/text-editdist-rootcause-2026-07-12.md` with example pages per category (evidence). This directly answers "each root cause accounts for how many points."

Unit tests (`tests/test_text_editdist_rootcause.py`): synthetic `{norm_gt, norm_pred}` fixtures per category assert the classifier.

---

## 4. Phase 2 — Per-cause gated fixes

Each fix is independent and gated: regenerate affected pages + re-score the full set → text EditDist drops, good pages unchanged, Overall not beyond noise. A cause confirmed inherent is documented, not forced.

- **A. Looping → two-pass retry in the fast path.** After generation, `is_looping_output(text)` per page (zlib ratio); for hits, re-run that page single-pass with issue-#55 settings (`no_repeat_ngram_size=5, ngram_window=256, repetition_penalty=1.05`). This is the documented Task-12 follow-up; safe per the 2026-07-06 report (98.6% pages byte-identical). ~2–32 pages affected.
- **B. Over-generation (repetitive) → inference-path first.** Phase-1 confirms repetitive (zlib<0.20) vs dense. If repetitive: inspect the gundam tiling in `build_page_inputs`/`model.infer` for **duplicate-crop feeding** (a bug) → fix in `batching.py`/`engine.py`. If no tile bug (pure model over-gen): targeted per-page retry with stronger repetition params. **Never global ngram=5.**
- **C. Truncation → inference-path.** For truncated pages: is `max_length=32768` hit? Is decode stopping early (EOS)? Is a crop boundary dropping a region? Fix the path if so; else document as inherent.
- **D. Math residual → align with converter.** Analyze which LaTeX constructs `textblock2unicode` handles asymmetrically; align our pred's math emission. Expected small.
- **E. Content divergence + dense over-gen → inherent.** Document; do not force.

---

## 5. Phase 3 — Final eval + ship

Apply all confirmed-viable fixes; re-run the full 1,651-page eval; re-score (leaderboard round-3-first 口径); write a new manifest (new text EditDist + Overall). Update `docs/parity/text-editdist-rootcause-2026-07-12.md` with the attribution + per-cause fix outcomes + new numbers + what remains inherent.

---

## 6. Cross-cutting

- **Honesty (hard):** every fix recovers correct content. No artificial truncation/format-matching. Each category has dump evidence.
- **Gates:** each fix — full re-score; text EditDist must drop; good pages (63%) must not regress; Overall must not drop beyond noise. Full-eval scorer uses **workers=4**.
- **Cost:** Phase 1 ≈ 1 score (~20 min, reuse predictions). Phase 2 ≈ ≤4 cycles (regenerate affected + re-score, ~30–40 min each). Phase 3 ≈ 1 full eval (~50 min). Total ~3–4 h GPU.
- **Risks:** (a) dump hook touches the local scorer — env-gated, no metric change, revertible; (b) over-gen/truncation may be model-inherent → documented, not forced; (c) scorer deadlock → workers=4 mitigation.
- **Testing:** categorization unit tests; each fix gated by full re-eval.

---

## 7. Definition of done

- Evidence-based root-cause attribution table (official口径) committed.
- Each fixable cause fixed + gated (text EditDist down, no good-page regression).
- Inherent causes honestly documented.
- New manifest (Overall + text EditDist, leaderboard round-3-first) + parity doc.
