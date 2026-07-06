# README Redesign — Top-Tier Open Source Standard

**Date:** 2026-07-06 | **Status:** design

## Context

Unlimited-OCR-ROCm has built substantial infrastructure since initial release — OmniDocBench evaluation pipeline (v1.6 Overall 92.04), gate-based regression prevention, manifest traceability, full release automation, CI/CD, bilingual docs, and 20+ documentation files. The README has not been updated to reflect any of this. It still carries the inaccurate "zero accuracy loss" claim and hides the evaluation data that gives the project its strongest credibility.

## Goal

Redesign both `README.md` and `README_CN.md` to meet top-tier open source project standards — positioned to persuade AI developers to adopt Unlimited-OCR on AMD ROCm GPUs with trust built on real evaluation data.

## Design Decisions

### Approach: Hub Model (~350–400 lines)

The README is a credibility builder + activation gateway, not an encyclopedia. Key evidence is surfaced inline; depth is linked to `/docs/`. A Hub model balances three impressions equally: production rigor, ease of use, and AMD authority.

### Accuracy framing: Honest disclosure + root cause

Old claim: "Zero accuracy loss vs. the NVIDIA reference" — inaccurate per measured data.

New framing: OmniDocBench v1.6 Overall **92.04** on AMD ROCm (gfx1100, gundam mode), vs Baidu原始论文 (~93.92, [arxiv:2606.23050](https://arxiv.org/abs/2606.23050)). Gap of ~1.88pt attributed to ~14 inherent looping pages (~1%) and inline-math LaTeX formatting style differences — not recognition errors (formula CDM 95.7% ≈ paper 95.8%). Gate PASS. Manifest committed. Ready for reproduction.

### Evaluation frontloaded

OmniDocBench score, per-module breakdown, and gate verdict appear BEFORE Quick Start — answering the visitor's real question ("is this port reliable?") before giving them installation commands.

### Branding

All references to "AMD Cloud" use the full name **AMD Radeon Cloud** with URL `https://radeon.anruicloud.com/`. Both README versions include ModelScope and HuggingFace entry points.

### No competitor comparison

Deferred to a future document-parsing zone. This project focuses on persuading AI developers to use Unlimited-OCR on ROCm/AMD GPUs.

## README Structure

### Section 0: Badge Wall (7 badges)

PyPI version | ROCm 6.0+ | Python 3.10+ | CI status | License MIT | Downloads | Stars

Add: CI status, Downloads, Stars.

### Section 1: Hero

```
<h1 align="center">Unlimited-OCR-ROCm</h1>
```

One-liner: "将百度的长文档 OCR SOTA 模型带到 AMD GPU 上 — 评测数据支撑，精度可复现。"

Three-column key metrics card: OmniDocBench 92.04 / gate PASS ✓ | TODO: 速度评测待完成 | 16 GB VRAM R-SWA constant

Retain `assets/Unlimited-OCR.png` diagram.

### Section 2: OmniDocBench Evaluation (NEW, frontloaded)

| | Overall | Text | Table TEDS | Formula CDM | Reading |
|---|---|---|---|---|---|
| AMD ROCm | 92.04 | 90.6% | 89.8% | 95.7% | 85.5% |
| Baidu原始论文* | ~93.92 | — | — | 95.8% | — |

\*Baidu self-report from [arxiv:2606.23050](https://arxiv.org/abs/2606.23050). Our AMD measured score ~1.88pt below, with known root causes: ~14 inherent looping pages (~1%) + inline-math LaTeX formatting style differences (not recognition errors; formula CDM 95.7% ≈ paper 95.8%).

Links: → Full parity report → Reproduction recipe → Manifest file

Remove: the current DPI self-comparison accuracy table (misleading — it's Levenshtein vs DPI=300 self-reference, not OmniDocBench). Replace with link to `docs/BENCHMARK.md`.

### Section 3: Why Unlimited-OCR-ROCm

Three differentiators:
- **评测可信** — OmniDocBench standard, gate prevents regression, manifest traceable. Not present in original Baidu repo.
- **AMD 原生** — ROCm one-command launch, 16 GB consumer Radeon runs a whole book, no NVIDIA GPU needed.
- **结构化输出** — Markdown with tables, formulas, bounding boxes. Same API as original.

### Section 4: Quick Start

Keep existing 3-command local. Add Docker entry (project already has Dockerfile + docker-compose). Add AMD Radeon Cloud link for no-GPU access. Include HuggingFace and ModelScope entry points.

### Section 5: Performance Tuning

Keep existing 3-scenario tuning commands. Compact. Link to `docs/TUNING.md`.

### Section 6: R-SWA Architecture

Keep existing "Why VRAM stays constant" section — traditional vs R-SWA comparison + multi-page VRAM table. This is an effective visual explanation.

### Section 7: Evaluation Infrastructure (NEW)

```
eval/ → omnidocbench predictions → gate gatekeeper → manifest.yaml → release
                  ↓ BLOCK on regression
```

- **Manifest** — Every eval result has a traceable YAML snapshot (git commit, model revision, env, metrics)
- **Gate** — Strict regression gate (Overall drop >0.3 or any module >0.005 → BLOCK)
- **Release** — eval → manifest → gate → PR → merge → tag → PyPI publish, fully automated

This is the project's strongest differentiator vs the original Baidu repo.

### Section 8: Usage Cheatsheet / Config / Async Engine

Keep existing, slightly compacted.

### Section 9: Project Structure

Compacted version — remove per-module file listing, keep directory-level summary.

### Section 10: Troubleshooting

Keep existing collapsible FAQ.

### Section 11: Roadmap + Community

Keep Roadmap link + community links. Trim to essentials.

### Section 12: Acknowledgement + License

Keep existing. Full name "AMD Radeon Cloud" with correct URL.

## README_CN.md

Structure synchronized with English README. Adjust tone for Chinese audience. Both versions include HuggingFace + ModelScope entry points.

## Key Changes vs Current README

| Current Issue | Resolution |
|---|---|
| "Zero accuracy loss" claim inaccurate | Honest disclosure: 92.04 vs ~93.92 (Baidu paper), with root cause analysis |
| Evaluation data invisible | OmniDocBench score + per-module table frontloaded |
| "Benchmark Snapshot" uses self-comparison | Removed; replaced with link to BENCHMARK.md |
| Evaluation infrastructure absent | New standalone "Evaluation Infrastructure" section |
| Badges sparse | Add CI, Downloads, Stars |
| No Docker entry | Add Docker Quick Start |
| Project structure verbose | Compact to directory-level |
| "AMD Cloud" everywhere | Full name: AMD Radeon Cloud (`https://radeon.anruicloud.com/`) |

## Edge Cases

- **User skims only badges + hero + quick start**: They get the core value prop (92.04 + gate PASS + 16GB) and can install in 3 commands. Covered.
- **User wants deep evaluation data**: Linked to `docs/PARITY.md` with full breakdown and root cause analysis.
- **User has no AMD GPU**: Directed to AMD Radeon Cloud (free trial), HuggingFace, or ModelScope.
- **User questions the 1.88pt gap**: README gives honest attribution (looping pages + LaTeX formatting). Detailed breakdown in PARITY.md.
- **User is comparing to original Baidu repo**: Evaluation infrastructure section shows what their repo lacks (manifest, gate, release pipeline).
