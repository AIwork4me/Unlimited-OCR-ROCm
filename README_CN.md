<h1 align="center">Unlimited-OCR-ROCm</h1>

<p align="center">
  <strong>在 AMD GPU 上运行百度顶级长文档 OCR — 评测数据支撑，精度可复现。</strong>
</p>

<p align="center">
  OmniDocBench v1.6 Overall 92.337（PyTorch 快速路径） · gate 通过 · 1.88× 无损加速 · 16 GB 显存 · R-SWA 恒定内存
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
| **92.337 Overall** ✓ _（PyTorch 快速路径）_ | **PASS** | **1.88× 无损** · ~0.21 pp/s（4-GPU） | **16 GB** |

</div>

[English README](README.md) | [精度对齐报告](docs/PARITY.md) | [基准测试](docs/BENCHMARK.md) | [架构](docs/ARCHITECTURE.md) | [调优指南](docs/TUNING.md)

---

## OmniDocBench 评测

百度的 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) 是当前长文档解析的新标杆，单次前向传播即可处理整本书、多页合同和密集表格。我们将其移植到 AMD ROCm，并运行完整的 OmniDocBench v1.6 标准评测（1,651 页），建立精度对齐基线。

| Model | Overall ↑ | TextEdit ↓ | FormulaCDM ↑ | TableTEDS ↑ | TableTEDS_s ↑ | Read-orderEdit ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **PyTorch 快速路径**（本项目，AMD ROCm gfx1100） | **92.337** | 0.0879 | 95.90 | 89.90 | 93.18 | 0.1456 |
| _前基线_（PyTorch，torch 2.5.1+rocm6.2） | _91.97_ | _0.0939_ | _95.72_ | _89.58_ | _92.83_ | _0.1449_ |
| Baidu 原始论文\* | 93.92 | 0.042 | 95.79 | 90.16 | 93.32 | 0.129 |

*\*百度原始论文自报分数，来源 [arxiv:2606.23050](https://arxiv.org/abs/2606.23050) —— **不在 OmniDocBench 排行榜上、未被任何人独立复现**。我们的 **92.337**（PyTorch 快速路径，固定权重 `84757cb0`，torch 2.10+rocm7.0）是受控、可复现的实测（已提交 manifest，gate 通过），较前 91.97 基线提升 **+0.365**，所有模块均 ≥ 基线。该增益来自环境+权重，而非 batching 运气 —— Task-8 identity gate 在同一环境下确认 fast ≈ direct（Δ=0.0009）。*

**诚实的精度定位。** 与百度 93.92 的 ~1.58pt 差距**几乎全部来自 Text EditDist**，无损逐页分解（[`docs/parity/moderate-tail-attribution-2026-07-11.md`](docs/parity/moderate-tail-attribution-2026-07-11.md)）表明现实无损上限约为 **~92.5–93.0** —— 我们的 92.337 距其仅 ~0.2–0.7。差距构成约 35% 行内数学 LaTeX 风格（模型输出语义正确的 `\(...\)`、`\sin`，而 GT 用 `$...$`、`\operatorname{s i n}`；CDM 0.959 证实数学正确 —— 字符级 EditDist 惩罚的是分隔符/间距差异）+ ~25% 真实识别上限 + ~25% 密集版面分歧（书籍索引/报纸）+ ~15% 格式/间距。无损可弥合部分仅 ~+0.5 pts。37% 的页面占据 93.2% 的 EditDist 总量；62.8% 的页面为"良好"（EditDist <0.05）。完整诊断：[docs/PARITY.md](docs/PARITY.md)。

**速度。** 快速路径（分桶 batching）在受控 30 页 gate 上比直接逐页路径快 **1.88× 无损**（同环境、同 scorer，Overall Δ=0.0009），全量 1,651 页运行在 4× gfx1100 上达到 **~0.21 pages/s 聚合**（墙钟 ~7,840 s）。[基准测试 →](docs/BENCHMARK.md)*

**→ [完整精度对齐报告含模块拆解](docs/PARITY.md) · [复现方法](docs/PARITY.md#reproduction-recipe) →**

---

## 后端状态（两条路径，实话实说）

本项目在 AMD ROCm gfx1100 上通过**两种后端**运行 Unlimited-OCR：

| 后端 | 状态 | OmniDocBench |
|---|---|---|
| **PyTorch 快速路径**（分桶 batching，`model.infer`） | ✅ 已验证的对齐参考（即上面的 92.337）+ 1.88× 无损加速 | **Overall 92.337**，gate 通过 —— 1,651 页，已提交 manifest |
| **vLLM / ROCm** 服务 | ⏳ 受数值发散阻塞的预览 | 约 10% 页面灾难性退化（首 token 即 EOS） |

**PyTorch 快速路径**（分桶 batching）是已验证的对齐参考：Task-8 identity gate 确认它与直接逐页路径在 Δ=0.0009 内一致（30 页中 4 页差一个重音字符 —— bf16 batching 数值），因此既准确（92.337）**又**快 1.88×。**vLLM/ROCm 服务后端**在约 10% 的页面上出现首-token EOS 回归（150 页代表性样本、同一 scorer：vLLM Overall **22.3** vs PyTorch 后端 **66.4**）。原因是**前向数值发散** —— bf16 + 优化 MoE/注意力 kernel（TRITON/ROCM_ATTN）相对 PyTorch eager —— 发生在边界页上。**不是** R-SWA（已用直接消融排除：在 PyTorch 中强制全因果注意力**不能**复现该 EOS），也**不是**任何 config / 解码契约 / processor bug（均已排除）。本 README 宣传的 92.337 是 **PyTorch** 快速路径；vLLM/ROCm 服务路径目前没有通过的分数。

**⏳ vLLM + ROCm 后端 → 等官方发布 v0.25.0+。** 定论性复验（在真实 vLLM/ROCm 构建上 serve unlimited-ocr 并重新评分）推迟到 vLLM 发布官方 **v0.25.0+ ROCm wheel**（第一个含核心侧 R-SWA + Triton 后端的稳定版）。服务脚本已就绪：[`scripts/rswa_spike/`](scripts/rswa_spike/)。完整调查 + 复验触发条件：[`docs/parity/rswa-spike-verdict-2026-07-11.md`](docs/parity/rswa-spike-verdict-2026-07-11.md)。

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
