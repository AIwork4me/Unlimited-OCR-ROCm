<h1 align="center">Unlimited-OCR-ROCm</h1>

<div align="center">
  <h4>Powered by AMD · Try it on <a href="https://radeon.anruicloud.com/">AMD Radeon Cloud</a></h4>
</div>

<div align="center">
  <a href="https://github.com/AIwork4me/Unlimited-OCR-ROCm">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" />
  </a>
  <a href="https://pypi.org/project/unlimited-ocr-rocm">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/unlimited-ocr-rocm?logo=pypi&logoColor=white" />
  </a>
  <a href="https://github.com/AIwork4me/Unlimited-OCR-ROCm/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg" />
  </a>
  <a href="https://www.python.org/downloads/">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  </a>
  <a href="https://rocm.docs.amd.com">
    <img alt="ROCm" src="https://img.shields.io/badge/ROCm-6.0%2B-red?logo=amd&logoColor=white" />
  </a>
  <a href="docs/BENCHMARK.md">
    <img alt="Benchmark" src="https://img.shields.io/badge/benchmark-56_tok%2Fs-00b894" />
  </a>
  <a href="docs/BENCHMARK.md">
    <img alt="VRAM" src="https://img.shields.io/badge/VRAM-7.3_GB-0984e3" />
  </a>
  <a href="docs/BENCHMARK.md">
    <img alt="Accuracy" src="https://img.shields.io/badge/accuracy-100%25-6c5ce7" />
  </a>
</div>

<br>

<p align="center">
  <strong>Run <a href="https://github.com/baidu/Unlimited-OCR">Baidu Unlimited-OCR</a> on AMD GPUs. One command. 56 tok/s. Zero accuracy loss. Only 16GB VRAM required.</strong>
</p>

<p align="center">
  <img src="assets/Unlimited-OCR.png" width="900" alt="Unlimited-OCR overview" />
</p>

<p align="center">
  <sub>Above: a 14-page academic paper parsed into structured Markdown on an AMD GPU — 41KB of clean output, identical to CPU/NVIDIA results.</sub>
</p>

---

[中文文档 (Chinese README)](README_CN.md) | [Benchmarks](docs/BENCHMARK.md) | [Tuning Guide](docs/TUNING.md) | [Architecture](docs/ARCHITECTURE.md)

---

## Who Is This For?

| You are… | You want… | This project gives you… |
|-----------|-----------|------------------------|
| **AMD GPU owner** (Instinct / Radeon) | To run the best OCR model without buying NVIDIA hardware | One-command deployment, auto ROCm detection |
| **AI startup / researcher** | Batch process thousands of PDFs at minimal cost | **56 tok/s** per GPU, **7.3 GB** VRAM peak |
| **Document pipeline engineer** | Production-grade OCR with structured output (Markdown + bounding boxes) | OpenAI-compatible API, Docker, SGLang serving |
| **ML tinkerer** | Understand WHY the model behaves as it does | Full root cause analysis in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |

### How It Compares

| Tool | AMD GPU | Structured Output | Long-Horizon | Cost |
|------|---------|-------------------|-------------|------|
| **Unlimited-OCR-ROCm** | ✅ Native ROCm | ✅ Markdown + bboxes | ✅ 32K context, R-SWA | Free (MIT) |
| Original Unlimited-OCR | ❌ NVIDIA only | ✅ | ✅ | Free (MIT) |

---

## Quick Start (3 Commands)

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git && cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh && source .venv/bin/activate
unlimited-ocr --pdf ./my_document.pdf --output-dir ./outputs
```

You'll get one `.md` file per page in `./outputs/`.

---

## Installation

**Prerequisites:** AMD GPU + ROCm 6.0+. Python 3.10–3.12.

```bash
# One-click (recommended)
./scripts/setup_rocm.sh --rocm-version 6.2 --python 3.12
source .venv/bin/activate

# Or: pip
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
pip install "sglang[all]>=0.4.0"
pip install -e .

# Or: Docker
ROCm_VERSION=6.2 docker compose build
docker compose run --rm unlimited-ocr --pdf /workspace/inputs/doc.pdf -o /workspace/outputs
```

## Two Inference Methods

| | Transformers | SGLang (Production) |
|------|-------------|---------------------|
| **Best for** | Quick tests, single images | Batch processing, serving |
| **Example** | `python examples/transformers_infer.py --pdf doc.pdf` | `bash examples/sglang_server.sh` + `python examples/sglang_client.py --pdf doc.pdf` |
| **Guide** | [examples/README.md](examples/README.md) | [examples/README.md](examples/README.md) |

---

## Benchmark Snapshot

> Full data: [docs/BENCHMARK.md](docs/BENCHMARK.md) | Benchmarked on AMD Radeon Graphics, ROCm 7.2, warm runs.

| DPI | tok/s | VRAM | Accuracy vs DPI=300 |
|-----|-------|------|---------------------|
| 100 | 54 | 7.3 GB | **100%** |
| 150 | 53 | 7.3 GB | **100%** |
| 200 | 54 | 7.3 GB | **100%** |
| 300 | 33 | 9.2 GB | reference |

**Key finding:** DPI=150 gives **identical text** to DPI=300, at 38% higher speed and 2 GB less VRAM. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the root cause analysis.

> **Note on SGLang benchmarks:** The benchmarks above use the HuggingFace Transformers backend — verified on real AMD hardware. SGLang adds paged attention, continuous batching, and optional `torch.compile` for additional throughput gains. We welcome community SGLang benchmark submissions!

---

## Performance Tuning

See [docs/TUNING.md](docs/TUNING.md) for scenario-based tuning guides.

```bash
# Max speed
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 150 --concurrency 8

# Max quality
unlimited-ocr --pdf doc.pdf --image-mode base --pdf-dpi 300 --max-length 32768

# Low VRAM (16 GB GPU)
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 100 --max-length 4096 --mem-fraction 0.6
```

---

## Usage Cheatsheet

```
unlimited-ocr --image-dir ./images | --pdf ./doc.pdf \
              [--output-dir ./out] [--image-mode gundam|base] \
              [--gpu 0] [--concurrency 8] [--pdf-dpi 200] \
              [--page-size 16] [--torch-compile] [--quiet] [--version]
```

---

## Project Structure

```
Unlimited-OCR-ROCm/
├── src/rocm_ocr/        # Python package (CLI, GPU detect, infer, server)
├── examples/            # transformers_infer.py, sglang_server.sh, sglang_client.py
├── docs/                # BENCHMARK.md, TUNING.md, ARCHITECTURE.md
├── scripts/             # setup_rocm.sh, full_benchmark.py, accuracy_benchmark.py
├── tests/               # Unit tests
├── Makefile             # make install, make test, make benchmark
├── Dockerfile           # ROCm 6.0+ Docker image
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

- [ ] SGLang benchmark on Instinct MI300X
- [ ] vLLM backend support
- [ ] Web UI (drag-and-drop OCR)
- [ ] Radeon RX 7000 optimization guide

---

## Acknowledgement

Built on [Baidu Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), [SGLang](https://github.com/sgl-project/sglang), and [AMD ROCm](https://rocm.docs.amd.com).

---

MIT License. [LICENSE](LICENSE) · [Contributing](CONTRIBUTING.md) · [Report Issue](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new/choose)
