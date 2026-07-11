<h1 align="center">Unlimited-OCR-ROCm</h1>

<p align="center">
  <strong>Run Baidu's state-of-the-art long-horizon OCR on AMD GPUs — backed by real evaluation data, precision aligned.</strong>
</p>

<p align="center">
  OmniDocBench v1.6 Overall 92.436 (PyTorch fast path) · gate PASS · 1.88× lossless speedup · 16 GB VRAM · R-SWA constant memory
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
| **92.436 Overall** ✓ _(PyTorch fast path)_ | **PASS** | **1.88× lossless** · ~0.21 pp/s (4-GPU) | **16 GB** |

</div>

[中文文档 (Chinese README)](README_CN.md) | [Full parity report](docs/PARITY.md) | [Benchmarks](docs/BENCHMARK.md) | [Architecture](docs/ARCHITECTURE.md) | [Tuning guide](docs/TUNING.md)

---

## OmniDocBench Evaluation

Baidu's [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) is the new state-of-the-art for long-horizon document parsing — entire books, multi-page contracts, dense tables in a single forward pass. We've ported it to AMD ROCm and run the full OmniDocBench v1.6 standard evaluation (1,651 pages) to establish precision alignment.

| Model | Overall ↑ | TextEdit ↓ | FormulaCDM ↑ | TableTEDS ↑ | TableTEDS_s ↑ | Read-orderEdit ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **PyTorch fast path** (this project, AMD ROCm gfx1100) | **92.436** | 0.0868 | 95.83 | 90.16 | 93.30 | 0.1442 |
| _prior baseline_ (PyTorch, torch 2.5.1+rocm6.2) | _91.97_ | _0.0939_ | _95.72_ | _89.58_ | _92.83_ | _0.1449_ |
| Baidu paper\* | 93.92 | 0.042 | 95.79 | 90.16 | 93.32 | 0.129 |

*\*Baidu self-report from [arxiv:2606.23050](https://arxiv.org/abs/2606.23050) — **not on the OmniDocBench leaderboard and not independently reproduced** by anyone. Our **92.436** (PyTorch fast path, pinned weights `84757cb0`, torch 2.10+rocm7.0) is a controlled, reproducible measurement (committed manifest, gate PASS), up **+0.465** vs the prior 91.97 baseline with all modules ≥ baseline. The gain is env+weights + the decode_bpe postprocess fix, not batching luck — the Task-8 identity gate confirmed fast ≈ direct (Δ=0.0 exact post-fix) on the same env.*

**Honest parity framing.** The ~1.48pt gap to Baidu's 93.92 is **~entirely Text EditDist**, and a lossless per-page decomposition ([`docs/parity/moderate-tail-attribution-2026-07-11.md`](docs/parity/moderate-tail-attribution-2026-07-11.md)) shows the realistic lossless ceiling is **~92.5–93.0** — our 92.436 is within ~0.06–0.56 of it (essentially at ceiling). The gap decomposes ~35% inline-math LaTeX style (the model emits semantically-correct `\(...\)`, `\sin` where GT uses `$...$`, `\operatorname{s i n}`; CDM 0.958 confirms the math is right — char-level EditDist penalizes delimiter/spacing) + ~25% genuine recognition limits + ~25% dense-layout divergence (book indexes/newspapers) + ~15% format/spacing. Only ~+0.5 pts is closable losslessly. 37% of pages hold 93.2% of the EditDist mass; 62.8% of pages are "good" (EditDist <0.05). Full diagnosis: [docs/PARITY.md](docs/PARITY.md).

**Speed.** The fast path (bucketed batching) is **1.88× lossless** vs the direct per-page path on a controlled 30-page gate (same env, same scorer, Overall Δ=0.0 exact post-fix), and the full 1,651-page run does **~0.21 pages/s aggregate** on 4× gfx1100 (wall ~7,840 s). [Benchmarks →](docs/BENCHMARK.md)*

**→ [Full parity report with per-module breakdown](docs/PARITY.md) · [Reproduction recipe](docs/PARITY.md#reproduction-recipe) →**

---

## Backend status (two paths, honestly)

This project runs Unlimited-OCR via **two backends** on AMD ROCm gfx1100:

| Backend | Status | OmniDocBench |
|---|---|---|
| **PyTorch fast path** (bucketed batching, `model.infer`) | ✅ Verified aligned reference (the 92.436 above) + 1.88× lossless speedup | **Overall 92.436**, gate PASS — 1,651 pages, committed manifest |
| **vLLM / ROCm** serving | ⏳ Numerics-blocked preview | Catastrophic on ~10% of pages (first-token EOS) |

The **PyTorch fast path** (bucketed batching) is the verified aligned reference: the Task-8 identity gate confirmed it matches the direct per-page path **exactly** (post-`decode_bpe`-fix Δ=0.0; the earlier apparent 4/30-page single-accented-char divergence was the `decode_bpe` postprocess bug, now fixed — the only residual byte-differences are trailing newlines with zero EditDist impact), so it is both accurate (92.436) **and** 1.88× faster. The **vLLM/ROCm serving backend** regresses to first-token EOS on ~10% of pages (on a 150-page representative sample, same scorer: vLLM Overall **22.3** vs the PyTorch backend **66.4**). The cause is **forward-pass numerics** — bf16 + optimized MoE/attention kernels (TRITON/ROCM_ATTN) vs PyTorch eager — on borderline pages. It is **not** R-SWA (ruled out by a direct ablation: forcing full causal attention in PyTorch does **not** reproduce the EOS), and **not** any config / decoding-contract / processor bug (all ruled out). The 92.436 figure advertised in this README is the **PyTorch** fast path; the vLLM/ROCm serving path has no passing score yet.

**⏳ vLLM + ROCm backend → waiting for the official vLLM v0.25.0+ release.** Definitive re-verification (serving unlimited-ocr on a real vLLM/ROCm build and re-scoring) is deferred until vLLM publishes an official **v0.25.0+ ROCm wheel** (the first stable release with core-side R-SWA + the Triton backend). The serving scripts are staged and ready for that day: [`scripts/rswa_spike/`](scripts/rswa_spike/). Full investigation + the re-verification trigger: [`docs/parity/rswa-spike-verdict-2026-07-11.md`](docs/parity/rswa-spike-verdict-2026-07-11.md).

---

## Why Unlimited-OCR-ROCm

- **Evaluation you can trust** — Standard OmniDocBench v1.6 benchmark, committed manifests for every run, strict regression gate prevents silent quality drops. The original Baidu repository has none of this.
- **AMD native** — One command to launch on any ROCm 6.0+ GPU. 16 GB consumer Radeon handles an entire book. No NVIDIA GPU, no CUDA, no compromises.
- **Structured output** — Markdown with tables, formulas, and bounding boxes preserved — same inference API as the original.

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
