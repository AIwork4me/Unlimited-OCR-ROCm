# Roadmap

Making Baidu [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) the default way to parse long-horizon documents on AMD GPUs.

_Last updated: 2026-07-09_ · [Design spec](docs/superpowers/specs/2026-06-25-unlimited-ocr-rocm-top-tier-design.md) · [Session progress](docs/PROGRESS_2026-07-03.md)

## North star

Become the **de-facto standard** for running Baidu Unlimited-OCR (and long-horizon document parsing) on AMD Radeon / ROCm — from datacenter MI300X down to 16 GB consumer Radeon cards.

## Status

| Phase | Name | Status |
|-------|------|--------|
| Phase 1 | Evidence Engine | ✅ PyTorch v1.6 baseline done (Overall 91.95, CDM 0.957 — confirms 92.04) |
| Phase 2 | Upstream Siege | 🟡 In progress (SGLang workaround shipped + upstreamed [#30599](https://github.com/sgl-project/sglang/issues/30599); vLLM next) |
| Phase 3 | Thin Integrations | ⏳ Planned |

### Three-backend status (updated 2026-07-09)

| Backend | ROCm status | Note |
|---|---|---|
| **PyTorch** (transformers direct) | ✅ Working | Overall 91.97, full v1.6 eval, manifest committed — production backend + parity reference |
| **SGLang** | 🟡 Workaround shipped, parked | Serves end-to-end on gfx1100 via a torch-native MoE workaround; correct but doesn't reach parity (inherent bf16 divergence). Crash root cause = gfx942-only `sgl_kernel` (not triton) — verified [sglang#30245](https://github.com/sgl-project/sglang/issues/30245); consumer-RDNA support tracked [sglang#30599](https://github.com/sgl-project/sglang/issues/30599). Status: [sglang-radeon-rdna-status-2026-07-09.md](docs/upstream/sglang-radeon-rdna-status-2026-07-09.md) |
| **vLLM** | 🔜 Next | vLLM's ROCm stack officially lists gfx1100/1101/1200/1201; evaluate as the next backend (spec: [three-backend design](docs/superpowers/specs/2026-07-06-three-backend-sglang-vllm-parity-design.md)) |

**Next:** vLLM backend eval — its ROCm wheel covers consumer RDNA, so it's the most promising path to a second working serving backend (and a SGLang↔vLLM↔PyTorch A/B).

### §2 text-repetition status

Issue #55's looping pages (~3 in gundam, ~1% drag) are NOT fixed. The comment's `ngram_size=5` fix is **catastrophic globally** (Overall 91.95→64.56) — reverted. A **targeted** (per-page runaway detection + truncation) fix is needed. See `src/rocm_ocr/repetition_fix.py` (kept with a WARNING) + [PARITY.md](docs/PARITY.md).

## Phase 1 — Evidence Engine

The credibility wedge: prove parity on the OmniDocBench standard and reform public docs to lead with evidence.

- OmniDocBench v1.5 + v1.6 accuracy parity vs the NVIDIA reference.
- Credibility-first README — real numbers, real hardware, zero accuracy loss up front.
- Community flywheel: [community benchmarks](docs/COMMUNITY_BENCHMARKS.md), good-first-issues, and GitHub Discussions (coming).

## Phase 2 — Upstream Siege

Lock the canonical path: be referenced from upstream as _the_ AMD path.

- Make consumer Radeon (RDNA3/RDNA4) first-class in SGLang's AMD docs — umbrella feature-request filed [sglang#30599](https://github.com/sgl-project/sglang/issues/30599); crash root cause verified + upstreamed [sglang#30245](https://github.com/sgl-project/sglang/issues/30245).
- **vLLM backend eval** (next active workstream): vLLM ROCm lists gfx1100/1201 — the path to a second serving backend.
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
