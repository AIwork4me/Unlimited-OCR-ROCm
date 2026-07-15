# Issue #66 — text-EditDist gap engagement with baidu/Unlimited-OCR

- **This record:** 2026-07-15 (engagement spans 2026-07-12 → 2026-07-15)
- **Upstream issue:** [baidu/Unlimited-OCR#66](https://github.com/baidu/Unlimited-OCR/issues/66)
- **Our reply:** [comment 4976675225](https://github.com/baidu/Unlimited-OCR/issues/66#issuecomment-4976675225) (account `AIwork4me`, 2026-07-15)
- **Status:** awaiting maintainer (`@MurphyYin`)

## Context

We filed issue #66 asking the maintainers to localize our text-EditDist gap on
OmniDocBench v1.6: ours **0.087** (page-mean; Overall **92.451**) vs the ~0.042
implied by Baidu's self-reported ~93.92. CDM/TEDS are at parity; the gap is
~entirely Text EditDist. Our own root-cause investigation is in
[`../parity/text-editdist-rootcause-2026-07-12.md`](../parity/text-editdist-rootcause-2026-07-12.md)
(PR #62): ~80% inherent — 65% recognition divergence + 14% math-style residual
(CDM 0.958 ⇒ math correct) + 8% dense over-generation.

## The 2026-07-13 reply is NOT official

The only comment on #66 is from [`kushdab`](https://github.com/kushdab),
`author_association=NONE` — a **community member, not a baidu maintainer**. No
maintainer (including `@MurphyYin`, first author Youyang Yin) has replied.
Treat kushdab's analysis as useful community input, not authoritative
confirmation.

kushdab's substance: checked the README on `main`, confirmed the `model.infer`
example uses `prompt='<image>document parsing.'` (image token before text), so
our prompt format is byte-identical to the documented example and can be ruled
out. They flagged the inference path (`model.infer` vs the served SGLang/vLLM
endpoint in `infer.py`) and the text-metric definition as open variables, and
recommended pinging `@MurphyYin`.

## Our independent verification (2026-07-15)

- **Prompt format — ruled out, re-verified.** Rendered our actual chat template
  against our request payload → it produces `<image>document parsing.`, byte-for-byte
  identical to the README `model.infer` example. (Nuance: the README's *served*
  SGLang `generate()` example sends the text `"document parsing."` without the
  literal `<image>` — the server injects the image token — but the two paths
  converge at the rendered model input.)
- **Decoding, inference path, and metric methodology — all ruled out** (see the
  parity doc). HF `model.infer`, the vLLM served path, and our PyTorch fast path
  all give text EditDist ≈ 0.087.

## Our reply (comment 4976675225)

We shared the evidence-based localization (the attribution table + block-level
stats, reproducible via `python scripts/analysis/text_editdist_rootcause.py`),
noted the `[Non-Text]` marker strip fix (Overall 92.431 → 92.451, re-verified
bit-for-bit on 2026-07-15 by re-running the full scorer) and the reverted
per-page looping retry, and framed the conclusion as ~80–87% inherent —
**conditional on our pinned checkpoint**.

## The one open variable (maintainer-only)

**Checkpoint revision.** We pin weights revision `84757cb0`. We could **not**
confirm whether that is the exact checkpoint behind Baidu's reported ~93.92, nor
whether there is an eval-config delta (image preprocessing, `image_mode`,
block-splitting) beyond the public README:

- The HuggingFace API was unreachable from our environment (MITM proxy), so we
  can't self-check for a newer revision.
- We have no NVIDIA hardware here, so a CUDA-vs-ROCm numerics isolation is not
  possible locally.

The reply asks `@MurphyYin` to confirm the checkpoint and any config delta.
**Until answered, "≈80% inherent" is conditional on `84757cb0`** — a checkpoint
or config delta alone could account for the residual 0.087-vs-0.042.

## Related

- [`../parity/text-editdist-rootcause-2026-07-12.md`](../parity/text-editdist-rootcause-2026-07-12.md) — the evidence-based attribution.
- [`baidu-amd-rocm-issue.md`](baidu-amd-rocm-issue.md) — the other open baidu upstream thread (AMD/ROCm support).
