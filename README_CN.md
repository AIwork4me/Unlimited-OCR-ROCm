<h1 align="center">Unlimited-OCR-ROCm</h1>

<p align="center">
  <strong>AMD GPU 上的顶级 OCR — 一条命令，56 tok/s，零配置。</strong>
</p>

<p align="center">
  我们将百度 Unlimited-OCR 搬上了 AMD ROCm。同等精度。更少显存。
  现在即可<strong>在真实 AMD 硬件上</strong>体验。
</p>

<div align="center">
  <a href="https://radeon.anruicloud.com/">
    <img src="https://img.shields.io/badge/在_AMD_Radeon_Cloud_体验-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="在 AMD Radeon Cloud 体验" />
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
  14 页学术论文 → 41KB 结构化 Markdown，运行在 AMD Radeon PRO W7900 48GB 显存上。<br>
  格式零损失。在 <a href="https://radeon.anruicloud.com/">AMD Radeon Cloud</a> 上亲自复现。
</blockquote>

---

[English README](README.md) | [Benchmarks](docs/BENCHMARK.md) | [Architecture](docs/ARCHITECTURE.md) | [调优指南](docs/TUNING.md)

---

## 为什么选择 Unlimited-OCR-ROCm

百度的 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) 是当前长文档解析的新标杆，单次前向传播即可处理整本书、多页合同和密集表格。

**Unlimited-OCR-ROCm** 将其带到 AMD GPU 上，零妥协：

- **一条命令运行** — 自动检测 ROCm，配置 SGLang + Triton 注意力，无需调参
- **零精度损失** — 与原始模型逐字逐标点完全一致
- **极低显存** — 低至 **16 GB** 消费级 Radeon 即可运行，得益于 R-SWA 恒定 KV 缓存
- **56 tok/s 吞吐** — 于 AMD Radeon PRO W7900 实测
- **结构化输出** — 完整保留 Markdown 格式、表格、公式和边界框

---

<!--
## 效果展示

输入/输出截图即将推出 — 在 AMD GPU 上运行 `make benchmark` 生成

详见 [docs/BENCHMARK.md](docs/BENCHMARK.md) 获取详细测试数据表格。
-->

---

## Benchmark 速览

> 完整数据：[docs/BENCHMARK.md](docs/BENCHMARK.md) | 于 AMD Radeon PRO W7900, ROCm 7.2 实测（同款 GPU 在 [AMD Radeon Cloud](https://radeon.anruicloud.com/) 上可用）。

### DPI × 精度

| DPI | tok/s | 显存 | 精度 |
|-----|-------|------|------|
| 150 | 56 | 7.3 GB | **100%** ★ |
| 200 | 54 | 7.3 GB | **100%** |
| 300 | 33 | 9.2 GB | 基准 |

**DPI=150 输出与 DPI=300 完全一致 — 快 38%，省 2 GB 显存。** [为什么？→](docs/ARCHITECTURE.md)

---

## 为什么显存不变

传统注意力机制：KV 缓存随每个 token 线性增长 → O(n²) 内存。

**R-SWA（参考滑动窗口注意力）：** 模型仅保留视觉 token（~256 个）+ 最近 128 个输出 token：

```
传统:    KV[t1, t2, ..., t1000]      ← 1000× 增长 → OOM
R-SWA:  KV[视觉~256] + KV[最近128]    ← 恒定
```

**我们用同一篇学术论文逐页增加测试验证了这一点：**

| 页数 | tok/s | 显存 |
|-----|-------|------|
| 1 | 56 | 7.3 GB |
| 5 | 56 | 7.4 GB |
| 10 | 55 | 7.4 GB |
| 25 | 55 | 7.5 GB |
| 50 | 54 | 7.5 GB |

显存从 1 页到 50 页仅增长 0.2 GB。**16 GB** 消费级显卡可处理整本书。

---

## 三种体验方式

| | ModelScope | AMD Radeon Cloud ★ | 本地 |
|------|-----------|-------------------|-------|
| **费用** | 免费 | 免费试用 | 免费 (MIT) |
| **GPU** | 免费 AMD GPU | 独享 AMD GPU | 你的 GPU |
| **配置** | 0 秒 | 60 秒 | 3 条命令 |
| **适用** | 快速体验 | 真实工作 | 完全控制 |
| **入口** | [打开 Demo →]() | **[注册 →](https://radeon.anruicloud.com/)** | 见下方 |

**推荐路径：** 先在 ModelScope 感受效果，准备好跑自己的文件时，[注册 AMD Radeon Cloud](https://radeon.anruicloud.com/) — 同款实测硬件，60 秒产出第一条 OCR 结果。

---

## 快速开始（3 条命令）

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git && cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh && source .venv/bin/activate
unlimited-ocr --pdf ./my_document.pdf --output-dir ./outputs
```

---

## 性能调优

```bash
# 最快速度 (56 tok/s)
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 150 --concurrency 8

# 最高质量
unlimited-ocr --pdf doc.pdf --image-mode base --pdf-dpi 300 --max-length 32768

# 低显存 (16 GB 显卡)
unlimited-ocr --pdf doc.pdf --image-mode gundam --pdf-dpi 100 --mem-fraction 0.6
```

完整指南：[docs/TUNING.md](docs/TUNING.md)

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
├── src/rocm_ocr/        # Python 包 (CLI, GPU 检测, 推理, 服务)
├── examples/            # transformers_infer.py, sglang_server.sh, sglang_client.py
├── docs/                # BENCHMARK.md, TUNING.md, ARCHITECTURE.md
├── scripts/             # setup_rocm.sh, 基准测试脚本
├── tests/               # 单元测试
├── Makefile             # make install, make test, make benchmark
├── Dockerfile           # ROCm 6.0+ Docker 镜像
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

## 社区

- [🐛 报告 Bug](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new?template=bug_report.md)
- [💡 请求功能](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new?template=feature_request.md)
- [📊 分享你的 Benchmark](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues?q=label%3A%22help+wanted%22)
- [🌍 帮忙翻译](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues?q=label%3A%22good+first+issue%22)

---

## 致谢

基于 [百度 Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)、[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)、[SGLang](https://github.com/sgl-project/sglang) 和 [AMD ROCm](https://rocm.docs.amd.com) 构建。

特别感谢 AMD 提供的计算支持。在 [AMD Radeon Cloud](https://radeon.anruicloud.com/) 上体验。

---

MIT License. [LICENSE](LICENSE) · [贡献指南](CONTRIBUTING.md)
