# Parsing / Matching Audit (WS-A, Task A2)

> Question: across the worst/tail/good pages, is OmniDocBench-scorer
> **mis-parsing** our `.md` or **mis-matching** pred↔GT blocks (artificially
> inflating Text EditDist), or is the text genuinely different?
> Decision: does the cheap no-GPU fix path (Task D2) activate?

## Data source

Used the scorer's own **saved match-result JSON** directly (no scorer code
executed for the verdicts):

- `/workspace/OmniDocBench/result/eval_predictions_v16_quick_match_text_block_result.json`
  — flat list of 11,517 match rows across 1,557 pages, each with
  `gt`/`pred`/`norm_gt`/`norm_pred`/`edit`/`gt_category_type`/`pred_category_type`/
  `image_name`. This is the scorer's FINAL output (parse → match →
  cross-category adaptation), so it already reflects any segmentation /
  matching / category-leak pathology.
- `/workspace/OmniDocBench/result/eval_predictions_v16_quick_match_text_block_per_page_edit.json`
  — `{image_name: editdist}`, 1,557 entries, used only to pick pages by bin.

This is the cleanest source (the brief's Step 1 preferred path). The
`scripts/analysis/inspect_match.py` tool reproduces every per-page dump below
via `--from-per-page-edit`; `--live` re-runs `md_tex_filter` +
`match_gt2pred_simple` for the divergence check on `docstructbench pdf_57`.

## Page selection (per brief Step 2)

Bin distribution over 1,557 pages: `==0` 419 · `(0,0.05)` 538 · `[0.05,0.1)` 158 ·
`[0.1,0.5)` 386 · `>=0.5` 56.

- **5 worst** (edit ≥ 0.99)
- **3 tail** (0.1 ≤ edit < 0.5)
- **3 good** (edit < 0.05)

## Per-page table

| Page (image_name) | bin | GT-text / PRED-text rows | scorer observation | verdict |
|---|---|---|---|---|
| `docstructbench_enbook-zlib-o.O-17208435.pdf_57.jpg` | worst (1.0) | 3 / 0 saved (202 live) | saved result: 3 GT rows all `pred_cat=''`, `norm_pred=''`. **Live re-parse: `md_tex_filter` returns 202 `text_all` one-liners; fresh `match_gt2pred_simple` yields edit≈0.97 matched.** The eval pipeline's post-match adaptation collapsed the 202 fine-grained lines away from the 3 GT blocks. Even live, edit≈0.97 — content is genuinely misaligned at block granularity. | **matching-artifact + genuine-content-diff** (the 1.0 in the saved JSON is partly a segmentation/over-segmentation artifact; the underlying content is still ~0.97 off) |
| `jiaocai_f32828acecb4282c87eaa554d2e1db74e418cd6845843012463a3324028bdd9d.pdf_9.jpg` | worst (1.0) | 22 / 0 saved | saved: all 22 rows `pred_cat=''`. **No matching `.md` file exists in `/workspace/eval_predictions_v16/`** (only unrelated `jiaocai_*` / `jiaocaineedrop_*` files). Page scored against a missing/empty prediction. | **matching-artifact** (prediction missing at eval time → 22 GT blocks scored against empty pred → edit 1.0) |
| `yanbaopptmerge_yanbaoPPT_4570.jpg` | worst (1.0) | 31 / 2 saved | `.md` contains a real prediction but it is **hallucinated looping garbage** ("陈忠实…负责首钢首钢厂第二车间主任…" repeating; "甲级公司" repeating). 29/31 GT rows unmatched; the 2 matched rows pair GT with the wrong hallucinated block. | **genuine-content-diff** (model degenerate output / decoding loop) |
| `newspaper_0b1bb8d03b4287eb95f67b68c2cf9f92_1.jpg` | worst (1.0) | 24 / 24 saved | `.md` is mostly **`[Non-Text]` tokens** (80+ of them) for a page whose GT is a long list of names. 21/24 rows have `pred='NonText'`. Model returned Non-Text for genuine text. | **genuine-content-diff** (model failed to OCR text; emitted `[Non-Text]`) |
| `docstructbench_dianzishu_zhongwenzaixian-o.O-61518266.pdf_149.jpg` | worst (0.997) | 6 / 2 matched + 4 unmatched | row 0 perfect (edit 0); row 1 GT-text matched against a **table** pred (`tabletrtdcolspan3…`); 4 GT rows `pred_cat=''`. Table content leaking into text scoring + unmatched headings. | **cross-category-formula-as-text + matching-artifact** |
| `yanbaor2_b997efc056ce194205f46dbd1c669eb32da025c4a3055c88d5f19ce434040b7b.pdf_46.jpg` | tail (0.499) | 2 / 2 matched | both rows cleanly matched, but pred has **extra header/footer text** ("KPMG", "Source Pulse of Fintech 2018…") prepended — genuine text the model added. | **genuine-content-diff** |
| `book_en_搬书匠-3299-Swift…page_111.png` | tail (0.491) | 6 / 6 matched | row 0 edit 0.216 (extra "Standing on the Shoulders of Giants" watermark); rows are code-vs-text and contain real substitutions ("Initializingwithstartingvalues" vs "Initialization…"). | **genuine-content-diff** (watermark + code/text boundary) |
| `notes_9e951846094758afac08c620144e3a76_13.jpg` | tail (0.487) | 1 / 1 matched | single block, pred has OCR substitutions and extra commentary ("看到平子再着到相似比…"). | **genuine-content-diff** |
| `PPT_1001115_eng_page_003.png` | good (0.006) | 2 / 2 matched | both near-perfect; `title` row edit 0, `text_block` edit 0.015. | **segmentation-ok** |
| `PPT_1001115_eng_page_011.png` | good (0.013) | 2 / 2 matched | title edit 0, text edit 0.027. | **segmentation-ok** |
| `PPT_1001115_eng_page_015.png` | good (0.032) | 2 / 2 matched | title edit 0.024, text edit 0.039. | **segmentation-ok** |

## Verdict tally (11 pages)

- `segmentation-ok`: 3 (all "good" bin)
- `genuine-content-diff`: 4 (yanbaopptmerge looping, newspaper `[Non-Text]`,
  yanbaor2 extra headers, Swift watermark/code, notes substitutions)
- `matching-artifact` (incl. mixed): 3 (docstructbench pdf_57 over-segmentation
  collapse, jiaocai missing `.md`, docstructbench dianzishu partial)
- `cross-category-formula-as-text`: 1 (docstructbench dianzishu — table content
  scored as text)

## Whole-dataset materiality (computed from the result JSON)

- Total match rows: 11,517 across 1,557 pages.
- Rows with `pred_cat==''` and empty `norm_pred` (unmatched GT): **485 (4.2%)**
  across **87 pages (5.6%)**, covering **62,240 / 2,295,589 GT chars (2.7%)**.
- Rows whose `pred` contains non-text markers (`[Non-Text]`, `<td>`,
  `colspan`, …): **207 (1.8%)** across **151 pages**.
- Among the 11,032 properly matched rows (both cats non-empty): **mean edit
  0.0873, median 0.0000** — i.e. the matched-pair distribution is healthy; the
  gap lives in the tail + unmatched rows.
- Counterfactual: dropping the 485 unmatched rows from each page's per-row
  average moves the mean from **0.0944 → 0.1086** (it rises, because those
  rows carried edit=1.0 and removing them concentrates on still-imperfect
  matched pairs). **Zero pages** flip from `>=0.5` to `<0.1` after the drop.

## Summary position — does Task D2 activate?

**No.** Parsing/matching mismatch is **not a material contributor** to the
0.0944 mean Text EditDist and is not a cheap no-GPU fix.

The evidence: (1) among properly matched pairs the median edit is 0.0 and the
mean is only 0.087 — the scorer is parsing and matching our predictions
correctly in the vast majority of cases; (2) the pathologies that do exist
(unmatched GT on 87 pages, `[Non-Text]`/table leak on 151 pages) are
**symptoms of bad model output**, not of the scorer mis-reading good output —
the `.md` files genuinely contain looping hallucinations (`yanbaopptmerge`),
`[Non-Text]` spam (`newspaper`), missing files (`jiaocai`), and over-segmented
plain text with no markdown structure (`docstructbench pdf_57`); (3) the one
true scorer-side artifact (the over-segmentation collapse on `docstructbench
pdf_57`, where saved edit 1.0 becomes ~0.97 on a live re-parse) is a ~0.03
delta confined to a handful of pages and the underlying content is still
genuinely misaligned; and (4) removing all unmatched rows from the per-page
average does not bring a single worst page into the good bin.

The gap is therefore driven by **genuine content differences produced by the
model** — degenerate/looping decode, `[Non-Text]` emission on dense pages,
hallucinated headers, and over-segmented plain-text output. The fix path is
on the model/serving side (decode parameters, looping mitigation, non-text
suppression) — i.e. the WS-B/C GPU track — **not** a scorer-side patch. Task
D2 (cheap no-GPU scorer/post-processing fix) should **not** activate on the
strength of this audit; A3/A4 may revisit if the looping-tail decomposition
surfaces a distinct, scorer-addressable sub-population.

## Reproducibility

```bash
# Reproduce every per-page dump above:
/workspace/OmniDocBench/.venv/bin/python \
  scripts/analysis/inspect_match.py --from-per-page-edit --max-rows 6

# Divergence check (saved result vs live re-parse) for the over-segmentation case:
/workspace/OmniDocBench/.venv/bin/python \
  scripts/analysis/inspect_match.py \
  docstructbench_enbook-zlib-o.O-17208435.pdf_57.jpg --live
```
