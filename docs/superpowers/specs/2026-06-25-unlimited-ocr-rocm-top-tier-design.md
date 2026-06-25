# Design: Taking Unlimited-OCR-ROCm to Top-Tier Open Source

- **Date:** 2026-06-25
- **Status:** Approved (brainstorming phase complete) — pending implementation plan
- **Owner:** aiwork4me
- **North star:** Become the **de facto standard** for running Baidu Unlimited-OCR (and long-horizon document parsing) on AMD Radeon / ROCm — the path teams depend on in production.

---

## 1. Context & Current State

**What the project is.** `unlimited-ocr-rocm` ports Baidu's [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) — a state-of-the-art one-shot long-horizon document parser (3B params, BF16, 32k context) — to AMD ROCm GPUs via SGLang. One `pip install` / one CLI command to run.

**Engineering maturity (already solid).** ~1300 LOC source, ~850 LOC tests (35+ tests), ruff/mypy/pre-commit, CI scaffolding, Dockerfile + docker-compose, EN/CN README and BLOG, ARCHITECTURE/BENCHMARK/TUNING docs, issue/PR templates, CODEOWNERS, Star History. Version 1.2.0, classified "Production/Stable".

**Genuine technical moat (the biggest asset).**
- **R-SWA constant KV cache** — VRAM grows only ~0.2 GB from 1 → 50 pages; a 16 GB consumer Radeon runs an entire book.
- **DPI=150 ≡ DPI=300 finding** — identical output, 38% faster, 2 GB less VRAM.
- **56 tok/s** on AMD Radeon PRO W7900; structured Markdown (tables, formulas, bounding boxes) preserved.

**Observed tensions that work against "top-tier":**
1. README reads as a **sales funnel** — "AMD Radeon Cloud register" CTA dominates; the "See It in Action" demo is commented out ("coming soon"). AI developers distrust funnel-shaped projects.
2. Benchmarks are **single-GPU self-reported**, with no comparison to alternatives, no third-party reproducibility, and a "100% accuracy" claim with no defined metric or eval set.
3. **Zero community contributions** (all commits by AIwork4me); no Discussions enabled.

**Landscape findings that shape this design:**
- **SGLang already has official ROCm support**, but only validated on datacenter **Instinct MI300X/MI355X**. The **consumer Radeon (RDNA3: W7900/RX7900) tier is uncovered upstream** — the real contribution gap.
- **Baidu Unlimited-OCR is NVIDIA-only.** There is no AMD path upstream at all; the entire portability is this project's original contribution.
- The document-parsing field is **horizontally crowded on accuracy** (minerU, Docling, olmOCR, Marker, Surya, GLM-OCR, dots.mocr). We do **not** compete there.
- **Standard benchmarks already exist** (OmniDocBench v1.5 and v1.6). We do not invent a benchmark — we run the existing standards on AMD and prove parity.

## 2. Goals & Non-Goals

**Goals**
- Make the project's technical claims **verifiable and reproducible** (standard-benchmark parity).
- Get **referenced from upstream** as THE AMD path (SGLang AMD docs, Baidu README, HF card, ROCm pages).
- Make the project **trivially drop-in** to real production stacks (thin, high-leverage integrations).
- Reform the README to **lead with credibility**, not a commercial funnel.
- Start a **contribution flywheel** (currently zero community).

**Non-Goals (YAGNI red lines)**
- ❌ Build a multi-model "document-AI platform."
- ❌ Compete on raw OCR-accuracy SOTA (that is the upstream model's job, not ours).
- ❌ Broad integration sprawl (only the 3 integrations in Phase 3).

## 3. Core Thesis & Positioning

**One-line identity:**
> Unlimited-OCR-ROCm is the canonical, production-grade way to run Baidu's Unlimited-OCR (and long-horizon document parsing) on AMD Radeon / ROCm — from datacenter MI300X down to 16 GB consumer cards.

**The moat — three walls competitors do not have:**

| Wall | Evidence |
|---|---|
| **Uniqueness** | Baidu Unlimited-OCR is NVIDIA-only; we are its only AMD path. |
| **Uncovered tier** | SGLang's AMD support validates MI300X/MI355X only; consumer Radeon (W7900/RX7900) is ours. |
| **Citable IP** | R-SWA constant KV cache → 16 GB runs a whole book; DPI=150≡300 → 38% faster, 2 GB less VRAM. |

**North star, operationalized.** "De facto standard" is achieved when **any** of these is true:
1. SGLang's AMD docs and/or Baidu's README list this project as the AMD path.
2. OmniDocBench-on-AMD parity is a cited external reference.
3. Developers default to `pip install unlimited-ocr-rocm` when they need document parsing on Radeon.

## 4. Strategy Overview

Sequenced execution: **🅰️ Evidence Engine → 🅱️ Upstream Siege → 🅲️ Thin Integrations**, with relationship-warming (Phase 2) and the independent API work (Phase 3) allowed to overlap Phase 1. Solo-but-focused bandwidth; no parallelization that isn't independent.

Rationale: the scarcest resource is **upstream maintainers' trust and attention**. We earn it with evidence (Phase 1), then spend it on upstream (Phase 2). Sprawling integrations before securing the canonical path is how solo OSS projects burn out.

## 5. Phase 1 — Evidence Engine (the credibility wedge)

**Goal:** Replace "trust me, 100%" with a reproducible, standard-benchmark parity proof, and make the README lead with it.

**Deliverables**
- **Eval harness** — `scripts/eval_omnidocbench.py` running **OmniDocBench v1.5 AND v1.6** on AMD Radeon (W7900), producing metrics on the same definitions as the official leaderboard, one-command reproducible (pinned weights / ROCm version / seed). v1.5 = Unlimited-OCR's eval baseline; v1.6 = forward compatibility. Both covered to withstand scrutiny.
- **Parity report** — `docs/PARITY.md`: AMD W7900 vs NVIDIA reference, per-metric parity (byte-for-byte where applicable). Includes a crowded-field positioning table (where Unlimited-OCR-on-AMD sits vs minerU/Docling/olmOCR on the same benchmark) — used **only as a positioning anchor, never to pick a fight**.
- **Real demo** — before/after (input PDF page → structured Markdown with tables/formulas) as README images/GIF, replacing the commented-out "coming soon".
- **README rewrite** — identity line → parity number (+ repro link) → demo GIF → R-SWA/DPI technical story. AMD Radeon Cloud CTA **demoted** from hero to a small footer "no hardware? try free".

**Done criteria.** A skeptical AI developer lands on the README and within 30 seconds sees: (1) a standard-benchmark number with a reproducible link, (2) a real demo, (3) a credible technical story — with zero marketing-funnel friction.

## 6. Phase 2 — Upstream Siege (lock the canonical path)

**Goal:** Be referenced from upstream as THE AMD path. **Phase 1's parity report is the gate** — we bring evidence to upstream conversations, not promises. Relationship-warming (issues, AMD-contact alignment, scoping the SGLang gap) runs in parallel during Phase 1, but formal PRs wait for the report.

**Deliverables (ordered by leverage)**
- **SGLang contribution** — extend `docs/platforms/amd_gpu.md` (and kernels/recipes where needed) to cover **consumer RDNA3 (W7900/RX7900)** plus an **Unlimited-OCR recipe**, co-authored with AMD contacts, pointing back to this repo as the reference. Scope an AMD CI runner for the consumer tier with the AMD contact. This fills the real gap (SGLang only validates MI300X).
- **Baidu repo link** — open an issue/PR on `baidu/Unlimited-OCR` to add an "AMD/ROCm" section linking here (they are NVIDIA-only). Even a doc-level link is a large "standard" signal. Back it with the Phase-1 parity report.
- **HF model card / AMD ROCm page** — get listed on Baidu's HF model card ("Run on AMD") and AMD's ROCm community-models page.

**Done criteria.** At least one upstream (SGLang AMD docs **or** Baidu README) links here as the AMD path. The standard is now **institutionalized, not self-declared.**

## 7. Phase 3 — Thin Integrations (prove "production dependency")

**Goal:** Make the project trivially drop-in to existing stacks so adoption/dependency signals emerge. **Exactly 3 deliverables — everything else is cut.**

- **OpenAI-compatible endpoint** — reuse SGLang's native OpenAI mode + an Unlimited-OCR preset, **not** a from-scratch server (YAGNI). Any OpenAI client / curl / langchain plugs in.
- **One-click try** — an HF Space or `docker compose up` hosted/self-hosted demo. This is the "try without AMD hardware" path — **free and OSS**, replacing the heavy Radeon Cloud funnel.
- **One RAG example** — a canonical langchain **or** llamaindex (pick one) example using this project as the document loader. Proof of "real-pipeline dependency".

**Done criteria.** A developer goes `pip install` → OpenAI-compatible API → into their existing RAG stack in under 10 minutes, with a free one-click try available.

## 8. Cross-Cutting: README Reform & Community Flywheel

**README reform (living document, updated every phase):**
- Credibility (benchmark number + repro link) leads; AMD Radeon Cloud CTA is a footer "no hardware? try free".
- Real demo (GIF/screenshots) replaces "coming soon".
- Every phase's output flows back into the README.

**Community flywheel (active — currently zero community):**
- **Open Discussions** (Q&A + Show & Tell) for engagement beyond issues.
- **Community benchmark table** — contributors submit their own AMD card (RX 7900 XTX / 7800 XT / MI50 …) OmniDocBench results → a "does it run on my card?" matrix. Turns users into contributors and reinforces the consumer-Radeon positioning. ★
- **Good-first-issues** carved from real Phase 1–3 work (add a v1.6 metric, add a langchain example, translate TUNING.md).
- **Public `ROADMAP.md`** (this phased plan, polished) — shows momentum, attracts aligned contributors.
- **Release cadence** — keep CHANGELOG current, tag releases so the project feels alive.

## 9. Success Metrics

| Type | Metric |
|---|---|
| **Lagging (the standard itself)** | # upstream references pointing here (SGLang AMD docs / Baidu README / HF card / ROCm page); OmniDocBench-on-AMD cited externally. |
| **Leading (adoption signals)** | PyPI download trend; star velocity; # community-contributed benchmark results; external "how I use it" posts; non-AIwork4me good-first-issue PRs. |
| **Health (anti-metrics)** | Issue/PR response time; AMD CI green; parity holds across releases (regression guard). |

## 10. 90-Day Milestones (solo, focused, with overlap)

- **Day 1–30 (Phase 1):** v1.5 + v1.6 eval harness + parity report committed; README rewritten with real demo; cloud CTA demoted.
- **Day 20–60 (Phase 2 warming → PR):** SGLang AMD docs PR opened (consumer Radeon + recipe); Baidu repo issue/PR for AMD link; AMD CI runner scoped with contact.
- **Day 45–90 (Phase 3):** OpenAI-compatible preset + one-click demo + one RAG example shipped; Discussions open; first good-first-issues; first external benchmark contribution.

## 11. Risks & Mitigations

1. **Upstream PR latency / rejection** (SGLang/Baidu maintainers slow or decline) → lead with undeniable parity proof + AMD co-authorship; fallback: become the de-facto reference via community adoption + HF/ROCm listings even without a merge.
2. **Solo burnout / maintenance sprawl** → ruthless YAGNI (3 integrations only); the eval harness doubles as a regression test so parity does not bit-rot; CI automation.
3. **"Just a wrapper" dismissal** → the R-SWA + DPI findings and the consumer-Radeon gap are genuine IP; surface them as first-class content (blog posts, parity report), not buried.
4. **Commercial-cloud tension erodes trust** → README reform; the cloud is one of several "try" options, never the hero.
5. **Parity drift** (ROCm/SGLang updates break byte-identical output) → CI runs an OmniDocBench sample every release; pin versions; maintain a supported-matrix.

## 12. Open Questions / Decisions Log

- **OmniDocBench versions:** support **both v1.5 and v1.6** (decided 2026-06-25).
- **Phase 3 RAG framework:** langchain **or** llamaindex — to be decided at implementation time (default: langchain, larger reach).
- **Spec language:** English (matches `docs/` convention and upstream-facing work); CN translation available on request.
- **Commit strategy:** spec to be committed as an isolated commit (not bundled with the pre-existing dirty working tree); branch off `main` per repo workflow.
- **Implementation-plan scope:** `writing-plans` will plan **Phase 1 first** (eval harness + parity report + README rewrite + community-setup). Phases 2 and 3 get **separate plans at their respective gates** — do not try to plan all 90 days in one document.

## 13. References

- [SGLang `docs/platforms/amd_gpu.md`](https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md)
- [AMD official SGLang setup blog](https://rocm.blogs.amd.com/artificial-intelligence/sglang/README.html)
- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)
- [Unlimited-OCR review (GPU requirements)](https://pasqualepillitteri.it/en/news/6063/unlimited-ocr-baidu-long-pdfs)
- [Euler AI doc-parser benchmark (minerU/Docling/olmOCR)](https://www.eulerai.au/blog/doc-parser-benchmark)
- [r/LocalLLaMA — best OSS OCR 2026](https://www.reddit.com/r/LocalLLaMA/comments/1sk6kst/what_is_the_best_open_source_ocr_in_2026/)
