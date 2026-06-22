<h1 align="center">Unlimited-OCR-ROCm</h1>

<div align="center">
  <sub>感谢 AMD 提供算力支持 · <a href="https://radeon.anruicloud.com/">AMD Radeon Cloud</a> 平台可快速验证</sub>
</div>

<div align="center">
  <a href="https://github.com/AIwork4me/Unlimited-OCR-ROCm">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-代码-181717?logo=github&logoColor=white" />
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
  <strong>在 AMD GPU 上运行 <a href="https://github.com/baidu/Unlimited-OCR">百度 Unlimited-OCR</a>。一行命令。56 tok/s。零精度损失。仅需 16GB 显存。</strong>
</p>

<p align="center">
  <img src="assets/Unlimited-OCR.png" width="900" alt="Unlimited-OCR 概览" />
</p>

<p align="center">
  <sub>上图：一篇 14 页学术论文在 AMD GPU 上被解析为结构化 Markdown — 41KB 干净输出。</sub>
</p>

---

[English README](README.md) | [性能评测](docs/BENCHMARK.md) | [调优指南](docs/TUNING.md) | [架构分析](docs/ARCHITECTURE.md)

---

## 谁适合用？

| 如果你是… | 你想要… | 本项目能给你… |
|-----------|---------|-------------|
| **AMD GPU 用户** (Instinct / Radeon) | 不用买 NVIDIA 硬件就跑最好的 OCR 模型 | 一行命令部署，自动 ROCm 检测 |
| **AI 创业/研究人员** | 以最低成本批量处理数千 PDF | 每 GPU **56 tok/s**，**7.3 GB** 显存峰值 |
| **文档管线工程师** | 生产级 OCR，结构化输出 (Markdown + 边界框) | OpenAI 兼容 API，Docker，SGLang 服务 |
| **ML 技术钻研者** | 理解模型为什么这样运行 | 完整根因分析见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |

### 同类对比

| 工具 | AMD GPU | 结构化输出 | 长篇章 | 成本 |
|------|---------|-----------|--------|------|
| **Unlimited-OCR-ROCm** | ✅ 原生 ROCm | ✅ Markdown + 边界框 | ✅ 32K 上下文，R-SWA | 免费 (MIT) |
| 原版 Unlimited-OCR | ❌ NVIDIA only | ✅ | ✅ | 免费 (MIT) |

---

## 快速开始（3 条命令）

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git && cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh && source .venv/bin/activate
unlimited-ocr --pdf ./我的文档.pdf --output-dir ./outputs
```

每个页面生成一个 `.md` 文件保存在 `./outputs/`。

---

## 安装

**前置条件:** AMD GPU + ROCm 6.0+。Python 3.10–3.12。

```bash
# 一键安装（推荐）
./scripts/setup_rocm.sh --rocm-version 6.2 --python 3.12
source .venv/bin/activate

# 或：pip 手动
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
pip install "sglang[all]>=0.4.0"
pip install -e .

# 或：Docker
ROCm_VERSION=6.2 docker compose build
docker compose run --rm unlimited-ocr --pdf /workspace/inputs/doc.pdf -o /workspace/outputs
```

## 两种推理方式

| | Transformers | SGLang（生产） |
|------|-------------|---------------|
| **适用** | 快速测试、单张图片 | 批量处理、服务部署 |
| **示例** | `python examples/transformers_infer.py --pdf doc.pdf` | `bash examples/sglang_server.sh` + `python examples/sglang_client.py --pdf doc.pdf` |
| **指南** | [examples/README.md](examples/README.md) | [examples/README.md](examples/README.md) |

---

## 性能速览

> 完整数据：[docs/BENCHMARK.md](docs/BENCHMARK.md) | AMD Radeon Graphics, ROCm 7.2, 热身后数据。

| DPI | tok/s | 显存 | 精度 vs DPI=300 |
|-----|-------|------|------------------|
| 100 | 54 | 7.3 GB | **100%** |
| 150 | 53 | 7.3 GB | **100%** |
| 200 | 54 | 7.3 GB | **100%** |
| 300 | 33 | 9.2 GB | 参照组 |

**关键发现：** DPI=150 产出与 DPI=300 **完全相同的文本**，速度快 38%，省 2 GB 显存。根因分析见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

> **关于 SGLang 基准测试：** 以上数据使用 HuggingFace Transformers 后端在真实 AMD 硬件上实测。SGLang 通过分页注意力、连续批处理和可选的 `torch.compile` 可进一步提升吞吐。欢迎社区贡献 SGLang 实测数据！

---

## 性能调优

详见 [docs/TUNING.md](docs/TUNING.md) 场景化调优指南。

```bash
# 极速模式
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 150 --concurrency 8

# 高精度模式
unlimited-ocr --pdf doc.pdf --image-mode base --pdf-dpi 300 --max-length 32768

# 低显存模式（16 GB 显卡）
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 100 --max-length 4096 --mem-fraction 0.6
```

---

## 使用速查

```
unlimited-ocr --image-dir ./images | --pdf ./doc.pdf \
              [--output-dir ./out] [--image-mode gundam|base] \
              [--gpu 0] [--concurrency 8] [--pdf-dpi 200] \
              [--page-size 16] [--torch-compile] [--quiet] [--version]
```

---

## 项目结构

```
Unlimited-OCR-ROCm/
├── src/rocm_ocr/        # Python 包 (CLI, GPU 检测, 推理, 服务管理)
├── examples/            # transformers_infer.py, sglang_server.sh, sglang_client.py
├── docs/                # BENCHMARK.md, TUNING.md, ARCHITECTURE.md
├── scripts/             # setup_rocm.sh, full_benchmark.py, accuracy_benchmark.py
├── tests/               # 单元测试
├── Makefile             # make install, make test, make benchmark
├── Dockerfile           # ROCm 6.0+ Docker 镜像
└── pyproject.toml       # PEP 621 包元数据
```

---

## 常见问题

<details>
<summary><b>SGLang: "No HIP GPUs available"</b></summary>

```bash
rocm-smi --showproductname
export HIP_VISIBLE_DEVICES=0
```
</details>

<details>
<summary><b>显存溢出 (OOM)</b></summary>

降低 `--mem-fraction` 或 `--pdf-dpi`。见 [docs/TUNING.md](docs/TUNING.md) 场景三。
</details>

<details>
<summary><b>torch.cuda.is_available() → False</b></summary>

```bash
pip uninstall torch torchvision torchaudio -y
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
```
</details>

---

## 路线图

- [ ] Instinct MI300X 上的 SGLang 基准测试
- [ ] vLLM 后端支持
- [ ] Web 界面（拖拽式 OCR）
- [ ] Radeon RX 7000 优化指南

---

## 致谢

基于 [百度 Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)、[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)、[SGLang](https://github.com/sgl-project/sglang) 和 [AMD ROCm](https://rocm.docs.amd.com)。

---

MIT 许可证。[LICENSE](LICENSE) · [贡献指南](CONTRIBUTING.md) · [反馈问题](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new/choose)
