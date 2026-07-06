<h1 align="center">Unlimited-OCR-ROCm</h1>

<p align="center">
  <strong>在 AMD GPU 上运行百度顶级长文档 OCR — 评测数据支撑，精度可复现。</strong>
</p>

<p align="center">
  OmniDocBench v1.6 Overall 92.04 · gate 通过 · 16 GB 显存 · R-SWA 恒定内存
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
  14 页学术论文 → 41KB 结构化 Markdown，AMD Radeon PRO W7900 48GB 显存。<br>
  格式零损失。
</blockquote>

<div align="center">

| OmniDocBench v1.6 | Gate | 速度 | 最低显存 |
|---|---|---|---|
| **92.04 Overall** ✓ | **PASS** | TODO: 评测待完成 | **16 GB** |

</div>

[English README](README.md) | [精度对齐报告](docs/PARITY.md) | [基准测试](docs/BENCHMARK.md) | [架构](docs/ARCHITECTURE.md) | [调优指南](docs/TUNING.md)

---

## OmniDocBench 评测

百度的 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) 是当前长文档解析的新标杆，单次前向传播即可处理整本书、多页合同和密集表格。我们将其移植到 AMD ROCm，并运行完整的 OmniDocBench v1.6 标准评测（1,651 页），建立精度对齐基线。

| | Overall ↑ | TextEdit ↓ | FormulaCDM ↑ | TableTEDS ↑ | TableTEDS_s ↑ | Read-orderEdit ↓ |
|---|---|---|---|---|---|---|---|
| **AMD ROCm**（本项目） | **92.04** | 0.094 | 95.7 | 89.8 | 93.1 | 0.145 |
| Baidu原始论文* | 93.92 | 0.042 | 95.79 | 90.16 | 93.32 | 0.129 |

*\*百度原始论文自报分数，来源：[arxiv:2606.23050](https://arxiv.org/abs/2606.23050)。我们的 AMD 实测分数约低 1.88pt，已知根因：~14 个固有循环页面（~1% 拖累）和内联 LaTeX 格式风格差异 — 并非识别错误（FormulaCDM 95.7 ≈ 论文 95.79）。每次评测结果均附带已提交的 manifest 和 gate 通过验证。*

**→ [完整精度对齐报告含模块拆解](docs/PARITY.md) · [复现方法](docs/PARITY.md#reproduction-recipe) →**

---

## 为什么选择 Unlimited-OCR-ROCm

- **评测可信** — 基于 OmniDocBench v1.6 标准评测，每次运行均有 manifest 可追溯，严格回归门控阻止静默质量下降。原版百度仓库完全没有这套体系。
- **AMD 原生** — 一条命令启动，ROCm 6.0+ 即装即用。16 GB 消费级 Radeon 跑完整本书。无需 NVIDIA GPU，无需 CUDA。
- **结构化输出** — 保留 Markdown 表格、公式、边界框 — 与原始模型相同的推理 API。

---

## 快速开始

### 本地（3 条命令）

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

### 没有 AMD GPU？

| | HuggingFace | ModelScope | AMD Radeon Cloud |
|---|---|---|---|
| **费用** | 免费 | 免费 | 免费试用 |
| **GPU** | 共享 | 共享 | 独享 AMD GPU |
| **配置** | 0 秒 | 0 秒 | ~60 秒 |
| **适用** | 快速体验 | 快速体验 | 真实工作 |

**推荐路径：** 先在 HuggingFace 或 ModelScope 体验效果，准备好跑自己的文件时，[注册 AMD Radeon Cloud](https://radeon.anruicloud.com/) — 使用与我们实测相同的硬件。

---

## 性能调优

```bash
# 最快速度
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 150 --concurrency 8

# 最高质量
unlimited-ocr --pdf doc.pdf --image-mode base --pdf-dpi 300 --max-length 32768

# 低显存（16 GB 显卡）
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 100 --mem-fraction 0.6
```

完整指南：[docs/TUNING.md](docs/TUNING.md)

---

## 为什么显存不变

传统注意力机制：KV 缓存随每个 token 线性增长 → O(n²) 内存。

**R-SWA（参考滑动窗口注意力）：** 模型仅保留视觉 token（~256 个）+ 最近 128 个输出 token：

```
传统:    KV[t1, t2, ..., t1000]      ← 1000× 增长 → OOM
R-SWA:  KV[视觉~256] + KV[最近128]    ← 恒定
```

**我们用同一篇学术论文逐页增加测试验证了这一点：**

| 页数 | 显存 |
|-----|------|
| 1 | 7.3 GB |
| 5 | 7.4 GB |
| 10 | 7.4 GB |
| 25 | 7.5 GB |
| 50 | 7.5 GB |

显存从 1 页到 50 页仅增长 0.2 GB。**16 GB** 消费级显卡可处理整本书。[原理详解 →](docs/ARCHITECTURE.md)

---

## 评测基础设施

**Unlimited-OCR-ROCm 是唯一拥有完整自动化评测管线的 Unlimited-OCR 发行版。** 原版百度仓库完全没有这套体系。

```
eval/ → omnidocbench 预测 → gate 门控 → manifest.yaml → release
                 ↓ 不通过则 BLOCK
```

- **Manifest** — 每次评测生成可追溯的 YAML 快照：git commit、模型版本、运行环境、逐模块指标。存储在 `eval/results/` 下，CI 强制 JSON Schema 校验。
- **Gate** — 严格回归门控。Overall 下降 >0.3 或任一模块 >0.005 → **BLOCK**。绝不盲目合入。
- **Release** — 全自动化流水线：eval → manifest → gate → PR → merge → git tag → PyPI 发布。每个 release 都有已提交的评测 manifest。

详见 [Release 跑](docs/RELEASE.md)。

---

## 使用速查

```
unlimited-ocr --image-dir ./images | --pdf ./doc.pdf \
              [--output-dir ./out] [--image-mode gundam|base] \
              [--gpu 0] [--concurrency 8] [--pdf-dpi 200] \
              [--page-size 16] [--torch-compile] \
              [--async] [--quiet] [--version] [--config .unlimited-ocr.yaml]
```

### 异步引擎

高并发批处理场景使用 `--async` 标志，底层使用 aiohttp + asyncio：

```bash
unlimited-ocr --pdf ./large_doc.pdf --async --concurrency 16
```

同步引擎（默认）使用 requests + ThreadPoolExecutor。简单场景用同步，大规模用异步。

### 配置文件（YAML）

```yaml
# .unlimited-ocr.yaml
output_dir: ./outputs
image_mode: base
pdf_dpi: 150
concurrency: 8
quiet: false
```

放到项目根目录或任意父目录即可自动发现，也可 `--config ./my-config.yaml`。

---

## 项目结构

```
Unlimited-OCR-ROCm/
├── src/rocm_ocr/        # Python 包（CLI、推理、评测管线、GPU 工具）
├── docs/                # 架构、基准测试、精度对齐、调优、Release 跑
├── eval/                # 评测 manifest + JSON Schema（CI 强制校验）
├── scripts/             # 安装、多 GPU 评测运行、基准测试
├── tests/               # 单元测试（conftest fixtures）
├── examples/            # transformers_infer.py、SGLang server/client
├── Makefile             # make install、make test、make benchmark、make eval-release
├── Dockerfile           # ROCm 6.0+ Docker 镜像
├── docker-compose.yml   # Docker Compose 编排
└── pyproject.toml       # PEP 621 包元数据
```

---

## 故障排除

<details>
<summary><b>SGLang: "No HIP GPUs available"</b></summary>

```bash
rocm-smi --showproductname
export HIP_VISIBLE_DEVICES=0
```
</details>

<details>
<summary><b>OOM（显存不足）</b></summary>

降低 `--mem-fraction` 或 `--pdf-dpi`。参见 [docs/TUNING.md](docs/TUNING.md) 场景 3。
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

**Phase 1 — 证据引擎：** OmniDocBench 精度对齐 + 可信文档 ✅  
**Phase 2 — 上游整合：** SGLang/vLLM on ROCm，消费级 Radeon 进入 AMD 官方文档 ⏳  
**Phase 3 — 轻量集成：** OpenAI 兼容端点，一键托管 Demo，RAG 框架示例 ⏳

→ [完整 Roadmap](ROADMAP.md)

---

## 社区

- [🐛 报告 Bug](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new?template=bug_report.md)
- [💡 请求功能](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new?template=feature_request.md)
- [📊 分享你的 Benchmark](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues?q=label%3A%22help+wanted%22)
- [🌍 帮忙翻译](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues?q=label%3A%22good+first+issue%22)

→ [社区 Benchmark](docs/COMMUNITY_BENCHMARKS.md)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIwork4me/Unlimited-OCR-ROCm&type=Date)](https://star-history.com/#AIwork4me/Unlimited-OCR-ROCm&Date)

---

## 致谢

基于 [百度 Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)、[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)、[SGLang](https://github.com/sgl-project/sglang) 和 [AMD ROCm](https://rocm.docs.amd.com) 构建。

特别感谢 AMD 提供的计算支持。在 [AMD Radeon Cloud](https://radeon.anruicloud.com/) 上体验。

---

MIT License. [LICENSE](LICENSE) · [贡献指南](CONTRIBUTING.md)
