# Roadmap

Making Baidu [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) the default way to parse long-horizon documents on AMD GPUs.

_Last updated: 2026-06-25_ · [Design spec](docs/superpowers/specs/2026-06-25-unlimited-ocr-rocm-top-tier-design.md)

## North star

Become the **de-facto standard** for running Baidu Unlimited-OCR (and long-horizon document parsing) on AMD Radeon / ROCm — from datacenter MI300X down to 16 GB consumer Radeon cards.

## Status

| Phase | Name | Status |
|-------|------|--------|
| Phase 1 | Evidence Engine | 🚧 In progress |
| Phase 2 | Upstream Siege | ⏳ Planned |
| Phase 3 | Thin Integrations | ⏳ Planned |

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
