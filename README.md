<h1 align="center">Unlimited-OCR-ROCm</h1>

<p align="center">
  <strong>Run Baidu's state-of-the-art long-horizon OCR on AMD GPUs — backed by real evaluation data, precision aligned.</strong>
</p>

<p align="center">
  OmniDocBench v1.6 Overall 92.04 · gate PASS · 16 GB VRAM · R-SWA constant memory
</p>

<div align="center">
  <a href="https://pypi.org/project/unlimited-ocr-rocm">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/unlimited-ocr-rocm" />
  </a>
  <a href="https://github.com/AIwork4me/Unlimited-OCR-ROCm/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/AIwork4me/Unlimited-OCR-ROCm/actions/workflows/ci.yml/badge.svg" />
  </a>
  <a href="https://pypi.org/project/unlimited-ocr-rocm">
    <img alt="Python" src="https://img.shields.io/pypi/pyversions/unlimited-ocr-rocm" />
  </a>
  <a href="https://rocm.docs.amd.com">
    <img alt="ROCm" src="https://img.shields.io/badge/ROCm-6.0%2B-red?logo=amd&logoColor=white" />
  </a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg" />
  </a>
</div>

<br>

<p align="center">
  <img src="assets/Unlimited-OCR.png" width="900" alt="Unlimited-OCR overview" />
</p>

<blockquote align="center">
  14-page academic paper → 41KB structured Markdown on AMD Radeon PRO W7900 48GB VRAM.<br>
  Zero format loss.
</blockquote>

<div align="center">

| OmniDocBench v1.6 | Gate | Speed | Min VRAM |
|---|---|---|---|
| **92.04 Overall** ✓ | **PASS** | TODO: eval pending | **16 GB** |

</div>

[中文文档 (Chinese README)](README_CN.md) | [Full parity report](docs/PARITY.md) | [Benchmarks](docs/BENCHMARK.md) | [Architecture](docs/ARCHITECTURE.md) | [Tuning guide](docs/TUNING.md)

---

## OmniDocBench Evaluation

Baidu's [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) is the new state-of-the-art for long-horizon document parsing — entire books, multi-page contracts, dense tables in a single forward pass. We've ported it to AMD ROCm and run the full OmniDocBench v1.6 standard evaluation (1,651 pages) to establish precision alignment.

|  | Overall | Text | Table TEDS | Formula CDM | Reading |
|---|---|---|---|---|---|
| **AMD ROCm** (this project) | **92.04** | 90.6% | 89.8% | 95.7% | 85.5% |
| Baidu原始论文* | ~93.92 | — | — | 95.8% | — |

*\*Baidu self-report from [arxiv:2606.23050](https://arxiv.org/abs/2606.23050). Our AMD measured score is ~1.88pt below, with known root causes: ~14 inherent looping pages (~1% drag) and inline-math LaTeX formatting style differences — not recognition errors (formula CDM 95.7% ≈ paper 95.8%). Every evaluation result has a committed manifest with gate PASS verification.*

**→ [Full parity report with per-module breakdown](docs/PARITY.md) · [Reproduction recipe](docs/PARITY.md#reproduction-recipe) →**

---

## Why Unlimited-OCR-ROCm

- **评测可信 (evaluation you can trust)** — Standard OmniDocBench v1.6 benchmark, committed manifests for every run, strict regression gate prevents silent quality drops. The original Baidu repository has none of this.
- **AMD 原生 (AMD native)** — One command to launch on any ROCm 6.0+ GPU. 16 GB consumer Radeon handles an entire book. No NVIDIA GPU, no CUDA, no compromises.
- **结构化输出 (structured output)** — Markdown with tables, formulas, and bounding boxes preserved — same inference API as the original.

---

## Quick Start

### Local (3 commands)

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git && cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh && source .venv/bin/activate
unlimited-ocr --pdf ./my_document.pdf --output-dir ./outputs
```

### Docker

```bash
docker compose up -d
unlimited-ocr --pdf ./my_document.pdf --output-dir ./outputs
```

### No AMD GPU?

| | HuggingFace | ModelScope | AMD Radeon Cloud |
|---|---|---|---|
| **Cost** | Free | Free | Free trial |
| **GPU** | Shared | Shared | Dedicated AMD GPU |
| **Setup** | 0 (in-browser) | 0 (in-browser) | ~60 s |
| **Best for** | Quick look | Quick look | Real workloads |

**Recommended:** try the HuggingFace or ModelScope demo to see the output quality, then register at [AMD Radeon Cloud](https://radeon.anruicloud.com/) for dedicated hardware — the same GPU we benchmark on.

---

## Performance Tuning

```bash
# Max speed
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 150 --concurrency 8

# Max quality
unlimited-ocr --pdf doc.pdf --image-mode base --pdf-dpi 300 --max-length 32768

# Low VRAM (16 GB GPU)
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 100 --mem-fraction 0.6
```

Full guide: [docs/TUNING.md](docs/TUNING.md)
