# Community Benchmarks

> Real-world results from the community on real AMD hardware — Radeon PRO, consumer RX, Instinct. [Add yours](#how-to-submit).

Throughput and VRAM are easy to reproduce; the **OmniDocBench Overall** accuracy score is the headline parity number vs the NVIDIA reference.

| GPU | VRAM | ROCm | OmniDocBench Overall | tok/s | VRAM peak | settings | by |
|-----|------|------|----------------------|-------|-----------|----------|----|
| AMD Radeon PRO W7900 | 48 GB | 7.2 | (pending — run `make eval`) | 56 | 7.3 GB | gundam, DPI 150 | @aiwork4me |

## How to submit

1. **Throughput & VRAM** — run `make benchmark` on your AMD GPU.
2. **OmniDocBench Overall** — run `make eval` (the eval harness lands in v1.3 / this Phase 1). Until then, leave the column as `(pending — run make eval)`.
3. **Open a PR** adding a row, including your **GPU model** and **ROCm version**.

### Coverage we want

Consumer and older Radeon coverage is thin. We especially want rows for:

- RX 7900 XTX (RDNA3, 24 GB)
- RX 7800 XT (RDNA3, 16 GB)
- MI50 / Radeon VII (Vega 20)

…plus any Instinct (MI250 / MI300X) you have access to.
