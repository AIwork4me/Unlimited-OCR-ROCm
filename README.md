<h1 align="center">Unlimited-OCR-ROCm</h1>

<p align="center">
  <strong>State-of-the-Art OCR on AMD GPUs — Try it yourself on <a href="https://radeon.anruicloud.com/">AMD Radeon Cloud</a></strong>
</p>

<p align="center">
  We brought Baidu Unlimited-OCR to AMD ROCm. Same accuracy. Less VRAM.
  And you can <strong>try it on</strong> real AMD hardware right now.
</p>

<div align="center">
  <a href="https://radeon.anruicloud.com/">
    <img src="https://img.shields.io/badge/Try_on-AMD_Radeon_Cloud-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="Try on AMD Radeon Cloud" />
  </a>
  <a href="https://pypi.org/project/unlimited-ocr-rocm">
    <img src="https://img.shields.io/badge/pip_install-unlimited--ocr--rocm-3776AB?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI" />
  </a>
</div>

<br>

<div align="center">
  <a href="https://pypi.org/project/unlimited-ocr-rocm">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/unlimited-ocr-rocm" />
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

---

[中文文档 (Chinese README)](README_CN.md) | [Benchmarks](docs/BENCHMARK.md) | [Architecture](docs/ARCHITECTURE.md) | [Tuning Guide](docs/TUNING.md)

---

## Why Unlimited-OCR-ROCm

Baidu's [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) is the new state-of-the-art for long-horizon document parsing — entire books, multi-page contracts, dense tables in a single forward pass.

**Unlimited-OCR-ROCm** brings this to AMD GPUs with zero compromises:

- **One command to run** — auto-detects ROCm, configures SGLang + Triton attention, nothing to tune
- **Zero accuracy loss** — byte-for-byte identical output to the original
- **Minimal VRAM** — runs on **16 GB** consumer Radeon cards, thanks to R-SWA constant KV cache
- **56 tok/s** throughput on AMD Radeon PRO W7900
- **Structured output** — Markdown with tables, formulas, and bounding boxes preserved

---

<!--
## See It in Action

Before/after screenshots coming soon — run `make benchmark` on AMD GPU to generate

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for detailed benchmark tables.
-->

---

## Benchmark Snapshot

> Full data: [docs/BENCHMARK.md](docs/BENCHMARK.md) | Benchmarked on AMD Radeon PRO W7900, ROCm 7.2 (same GPU available on [AMD Radeon Cloud](https://radeon.anruicloud.com/)).

### DPI × Accuracy

| DPI | tok/s | VRAM | Accuracy |
|-----|-------|------|----------|
| 150 | 56 | 7.3 GB | **100%** ★ |
| 200 | 54 | 7.3 GB | **100%** |
| 300 | 33 | 9.2 GB | reference |

**DPI=150 output is identical to DPI=300 — 38% faster, 2 GB less VRAM.** [Why? →](docs/ARCHITECTURE.md)

---

## Why the VRAM Stays Constant

Traditional attention: KV cache grows with every token → O(n²) memory.

**R-SWA (Reference Sliding Window Attention):** The model only keeps visual tokens (~256) + last 128 output tokens in cache:

```
Traditional:  KV[t1, t2, ..., t1000]   ← 1000× growth → OOM
R-SWA:        KV[visual~256] + KV[last_128]  ← CONSTANT
```

**We verified this by OCRing the same academic paper at increasing page counts:**

| Pages | tok/s | VRAM |
|-------|-------|------|
| 1 | 56 | 7.3 GB |
| 5 | 56 | 7.4 GB |
| 10 | 55 | 7.4 GB |
| 25 | 55 | 7.5 GB |
| 50 | 54 | 7.5 GB |

VRAM grows only +0.2 GB from 1 to 50 pages. A **16 GB** consumer Radeon handles an entire book.

---

## Try It — 3 Ways

| | ModelScope | AMD Radeon Cloud ★ | Local |
|------|-----------|-------------------|-------|
| **Cost** | Free | Free trial | Free (MIT) |
| **GPU** | Free AMD GPU | Dedicated AMD GPU | Your GPU |
| **Setup** | 0 seconds | 60 seconds | 3 commands |
| **Best for** | Quick look | Real workload | Full control |
| **Go** | [Open Demo →]() | **[Register →](https://radeon.anruicloud.com/)** | See below |

**Recommended path:** Start with the ModelScope demo to see the magic. When you're ready to run your own files at full speed, [register on AMD Radeon Cloud](https://radeon.anruicloud.com/) — same hardware we benchmarked on, 60 seconds to your first OCR result.

---

## Quick Start (3 Commands)

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git && cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh && source .venv/bin/activate
unlimited-ocr --pdf ./my_document.pdf --output-dir ./outputs
```

---

## Performance Tuning

```bash
# Max speed (56 tok/s)
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 150 --concurrency 8

# Max quality
unlimited-ocr --pdf doc.pdf --image-mode base --pdf-dpi 300 --max-length 32768

# Low VRAM (16 GB GPU)
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 100 --mem-fraction 0.6
```

Full guide: [docs/TUNING.md](docs/TUNING.md)

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
├── scripts/             # setup_rocm.sh, benchmarks
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

## Community

- [🐛 Report a bug](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new?template=bug_report.md)
- [💡 Request a feature](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new?template=feature_request.md)
- [📊 Share your benchmark](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues?q=label%3A%22help+wanted%22)
- [🌍 Help translate](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues?q=label%3A%22good+first+issue%22)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIwork4me/Unlimited-OCR-ROCm&type=Date)](https://star-history.com/#AIwork4me/Unlimited-OCR-ROCm&Date)

---

## Acknowledgement

Built on [Baidu Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), [SGLang](https://github.com/sgl-project/sglang), and [AMD ROCm](https://rocm.docs.amd.com).

Special thanks to AMD for compute support. Try it on [AMD Radeon Cloud](https://radeon.anruicloud.com/).

---

MIT License. [LICENSE](LICENSE) · [Contributing](CONTRIBUTING.md)
