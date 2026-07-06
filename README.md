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

| | Overall ↑ | TextEdit ↓ | FormulaCDM ↑ | TableTEDS ↑ | TableTEDS_s ↑ | Read-orderEdit ↓ |
|---|---|---|---|---|---|---|---|
| **AMD ROCm** (this project) | **92.04** | 0.094 | 95.7 | 89.8 | 93.1 | 0.145 |
| Baidu原始论文* | 93.92 | 0.042 | 95.79 | 90.16 | 93.32 | 0.129 |

*\*Baidu self-report from [arxiv:2606.23050](https://arxiv.org/abs/2606.23050). Our AMD measured score is ~1.88pt below, with known root causes: ~14 inherent looping pages (~1% drag) and inline-math LaTeX formatting style differences — not recognition errors (FormulaCDM 95.7 ≈ paper 95.79). Every evaluation result has a committed manifest with gate PASS verification.*

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

---

## Why the VRAM Stays Constant

Traditional attention: KV cache grows with every token → O(n²) memory.

**R-SWA (Reference Sliding Window Attention):** The model only keeps visual tokens (~256) + last 128 output tokens in cache:

```
Traditional:  KV[t1, t2, ..., t1000]   ← 1000× growth → OOM
R-SWA:        KV[visual~256] + KV[last_128]  ← CONSTANT
```

**Verified by OCRing the same academic paper at increasing page counts:**

| Pages | VRAM |
|-------|------|
| 1 | 7.3 GB |
| 5 | 7.4 GB |
| 10 | 7.4 GB |
| 25 | 7.5 GB |
| 50 | 7.5 GB |

VRAM grows only +0.2 GB from 1 to 50 pages. A **16 GB** consumer Radeon handles an entire book. [How it works →](docs/ARCHITECTURE.md)

---

## Evaluation Infrastructure

**Unlimited-OCR-ROCm is the only Unlimited-OCR distribution with a complete, automated evaluation pipeline.** The original Baidu repository has none of this.

```
eval/ → omnidocbench predictions → gate gatekeeper → manifest.yaml → release
                  ↓ BLOCK on regression
```

- **Manifest** — Every evaluation result produces a traceable YAML snapshot: git commit, model revision, environment, per-module metrics. Stored under `eval/results/` with JSON Schema validation enforced in CI.
- **Gate** — Strict regression gatekeeper. Overall score drop >0.3 or any module drop >0.005 → **BLOCK**. No blind merges ever.
- **Release** — Full automated pipeline: eval → manifest → gate → PR → merge → git tag → PyPI publish. Every release has a committed eval manifest.

See the [release runbook](docs/RELEASE.md) for the full workflow.

---

## Usage Cheatsheet

```
unlimited-ocr --image-dir ./images | --pdf ./doc.pdf \
              [--output-dir ./out] [--image-mode gundam|base] \
              [--gpu 0] [--concurrency 8] [--pdf-dpi 200] \
              [--page-size 16] [--torch-compile] \
              [--async] [--quiet] [--version] [--config .unlimited-ocr.yaml]
```

### Async Engine

For high-concurrency batch workloads, `--async` uses aiohttp + asyncio for lower overhead:

```bash
unlimited-ocr --pdf ./large_doc.pdf --async --concurrency 16
```

The sync engine (default) uses requests + ThreadPoolExecutor. Choose sync for simplicity, async for scale.

### Configuration File (YAML)

```yaml
# .unlimited-ocr.yaml
output_dir: ./outputs
image_mode: base
pdf_dpi: 150
concurrency: 8
quiet: false
```

Place it in your project root or any parent directory — auto-discovered. Or `--config ./my-config.yaml`.

---

## Project Structure

```
Unlimited-OCR-ROCm/
├── src/rocm_ocr/        # Python package (CLI, inference, evaluation pipeline, GPU tools)
├── docs/                # Architecture, benchmarks, parity, tuning, release runbook
├── eval/                # Evaluation manifests + JSON Schema (CI-enforced)
├── scripts/             # Setup, multi-GPU eval runners, benchmarks
├── tests/               # Unit tests (conftest fixtures)
├── examples/            # transformers_infer.py, SGLang server/client
├── Makefile             # make install, make test, make benchmark, make eval-release
├── Dockerfile           # ROCm 6.0+ Docker image
├── docker-compose.yml   # Docker Compose orchestration
└── pyproject.toml       # PEP 621 package metadata
```

---

## Troubleshooting

<details>
<summary><b>SGLang: "No HIP GPUs available"</b></summary>

```bash
rocm-smi --showproductname
export HIP_VISIBLE_DEVICES=0
```
</details>

<details>
<summary><b>OOM (out of memory)</b></summary>

Reduce `--mem-fraction` or `--pdf-dpi`. See [docs/TUNING.md](docs/TUNING.md) Scenario 3.
</details>

<details>
<summary><b>torch.cuda.is_available() → False</b></summary>

```bash
pip uninstall torch torchvision torchaudio -y
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
```
</details>

---

## Roadmap

**Phase 1 — Evidence Engine:** OmniDocBench parity + credibility-first docs ✅  
**Phase 2 — Upstream Integration:** SGLang/vLLM on ROCm, consumer Radeon first-class in AMD docs ⏳  
**Phase 3 — Thin Integrations:** OpenAI-compatible endpoint, one-click hosted demo, RAG framework example ⏳

→ [Full roadmap](ROADMAP.md)

---

## Community

- [🐛 Report a bug](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new?template=bug_report.md)
- [💡 Request a feature](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new?template=feature_request.md)
- [📊 Share your benchmark](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues?q=label%3A%22help+wanted%22)
- [🌍 Help translate](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues?q=label%3A%22good+first+issue%22)

→ [Community benchmarks](docs/COMMUNITY_BENCHMARKS.md)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIwork4me/Unlimited-OCR-ROCm&type=Date)](https://star-history.com/#AIwork4me/Unlimited-OCR-ROCm&Date)

---

## Acknowledgement

Built on [Baidu Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), [SGLang](https://github.com/sgl-project/sglang), and [AMD ROCm](https://rocm.docs.amd.com).

Special thanks to AMD for compute support. Try it on [AMD Radeon Cloud](https://radeon.anruicloud.com/).

---

MIT License. [LICENSE](LICENSE) · [Contributing](CONTRIBUTING.md)
