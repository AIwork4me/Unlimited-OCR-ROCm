# Roadmap

Making Baidu [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) the default way to parse long-horizon documents on AMD GPUs.

_Last updated: 2026-07-03_ · [Design spec](docs/superpowers/specs/2026-06-25-unlimited-ocr-rocm-top-tier-design.md) · [Session progress](docs/PROGRESS_2026-07-03.md)

## North star

Become the **de-facto standard** for running Baidu Unlimited-OCR (and long-horizon document parsing) on AMD Radeon / ROCm — from datacenter MI300X down to 16 GB consumer Radeon cards.

## Status

| Phase | Name | Status |
|-------|------|--------|
| Phase 1 | Evidence Engine | ✅ PyTorch v1.6 baseline done (Overall 91.95, CDM 0.957 — confirms 92.04) |
| Phase 2 | Upstream Siege | ⏳ Planned (SGLang/vLLM on ROCm blocked on driver version — see below) |
| Phase 3 | Thin Integrations | ⏳ Planned |

### Three-backend status (2026-07-03)

| Backend | ROCm status | Note |
|---|---|---|
| **PyTorch** (transformers direct) | ✅ Working | Overall 91.95, full v1.6 eval, manifest committed |
| **SGLang** | ⚠️ Blocked | Source build: `sgl-kernel` compiled ✓ for gfx1100, v0.5.9 has the OCR model (`deepseek_ocr.py`); but `[all_hip]` deps (torchao 0.9.0) need newer torch than 2.5.1 → conflict |
| **vLLM** | ❌ Blocked | Only `vllm 0.24.0+rocm723` wheel exists (pins torch 2.11 / ROCm 7.2.3 > this host's 7.2.1 driver → won't init GPU) |

**Root cause**: this host's ROCm 7.2.1 driver + torch 2.5.1 (the issue #55 baseline) is older than the current vLLM/SGLang ROCm stack. Unblock options: (1) upgrade the driver to 7.2.3 (cleanest), (2) salvage SGLang with a ≤7.2.1-driver newer torch, (3) accept PyTorch-only. See [PROGRESS_2026-07-03.md](docs/PROGRESS_2026-07-03.md).

### §2 text-repetition status

Issue #55's looping pages (~3 in gundam, ~1% drag) are NOT fixed. The comment's `ngram_size=5` fix is **catastrophic globally** (Overall 91.95→64.56) — reverted. A **targeted** (per-page runaway detection + truncation) fix is needed. See `src/rocm_ocr/repetition_fix.py` (kept with a WARNING) + [PARITY.md](docs/PARITY.md).

## Phase 1 — Evidence Engine

The credibility wedge: prove parity on the OmniDocBench standard and reform public docs to lead with evidence.

- OmniDocBench v1.5 + v1.6 accuracy parity vs the NVIDIA reference.
- Credibility-first README — real numbers, real hardware, zero accuracy loss up front.
- Community flywheel: [community benchmarks](docs/COMMUNITY_BENCHMARKS.md), good-first-issues, and GitHub Discussions (coming).

## Phase 2 — Upstream Siege

Lock the canonical path: be referenced from upstream as _the_ AMD path.

- Make consumer Radeon (RDNA3) first-class in SGLang's AMD docs.
- Get linked from Baidu's Unlimited-OCR repo as the AMD path.

## Phase 3 — Thin Integrations

Prove "production dependency" with thin, high-leverage integrations.

- OpenAI-compatible endpoint.
- One-click hosted demo.
- One RAG-framework example.

## How to help

- Share your numbers on real AMD hardware → [Community benchmarks](docs/COMMUNITY_BENCHMARKS.md).
- Pick up a `good first issue` (being carved from Phase 1–3 work).
- Join **GitHub Discussions** (opening soon) for Q&A, tuning tips, and roadmap input.
