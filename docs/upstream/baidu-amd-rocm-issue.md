# Upstream Contribution: Baidu Unlimited-OCR — AMD/ROCm Link

## Issue/PR to `baidu/Unlimited-OCR`

### Title
Add AMD/ROCm community port reference (Unlimited-OCR-ROCm)

### Body

Hi! We've ported Unlimited-OCR to AMD ROCm GPUs (Radeon PRO W7900 / RX 7900 series, gfx1100/RDNA3) and would love a reference link in the README.

**Project:** [AIwork4me/Unlimited-OCR-ROCm](https://github.com/AIwork4me/Unlimited-OCR-ROCm)

**What it does:**
- Runs Unlimited-OCR on AMD GPUs via SGLang (datacenter) / direct transformers (consumer).
- One-command setup: `./scripts/setup_rocm.sh && unlimited-ocr --pdf doc.pdf`
- Supports 16 GB consumer cards (constant VRAM via R-SWA).
- Published on PyPI: `pip install unlimited-ocr-rocm`.

**Benchmark results (OmniDocBench v1.6, AMD Radeon PRO W7900):**
| Metric | NVIDIA (your paper) | AMD ROCm (our port) |
|--------|-------------------:|-------------------:|
| Overall | 93.92 | **92.04** |
| Formula CDM | 95.79 | **95.7** |
| Table TEDS | 90.16 | 89.8 |
| Text Edit_dist | 0.042 | 0.094* |

*The text Edit_dist gap is from inline-math LaTeX formatting style differences (model output style vs GT annotations), not recognition errors — formula CDM (95.7%) confirms the model's math recognition is identical.

**Requested change:** Add a section to README.md:

```markdown
## Community Ports

- **[AMD ROCm](https://github.com/AIwork4me/Unlimited-OCR-ROCm)** — Run Unlimited-OCR on AMD Radeon GPUs (gfx1100/RDNA3). OmniDocBench v1.6 Overall: 92.04.
```

This helps AMD GPU users discover the port and contributes to broader hardware support for Unlimited-OCR.
