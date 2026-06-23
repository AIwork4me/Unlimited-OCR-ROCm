# 我们在 AMD GPU 上跑了 Unlimited-OCR — 然后发现 DPI 根本不重要

**作者：** aiwork4me
**日期：** 2026 年 6 月
**标签：** AMD ROCm, OCR, Benchmark, Vision-Language Model, DeepSeek

---

## 意外的发现

百度这个月开源了 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)，我们做了每个 AMD GPU 用户都会做的事：试着在 ROCm 上跑起来。

跑通了。但我们没有止步于「能跑」。

我们在真实的 AMD 芯片上跑了 50 多组基准测试，横跨 4 个维度 —— DPI、文档类型、页数和图像模式。然后发现了一个反直觉的结果：

**DPI 150 产出的文本与 DPI 300 完全一致 — 速度却快了 38%，显存还少占 2 GB。**

以下是数据、根因分析和背后的含义。

---

## 测试硬件

本文中的每一个数字都来自真实 AMD 硬件：

| 项目 | 详情 |
|------|--------|
| GPU | AMD Radeon PRO W7900 |
| 显存 | 48 GB |
| ROCm | 7.2.53211 |
| 模型 | baidu/Unlimited-OCR |

> 你可以通过 [AMD Radeon Cloud](https://radeon.anruicloud.com/) 在**完全相同的 GPU** 上复现每一项基准测试 — 零配置，同款芯片。

---

## 发现一：DPI 不重要（大多数情况）

我们对同一张 A4 页面分别在 DPI 100、150、200、250 和 300 下进行 OCR，然后测量与 DPI=300 参照组的 Levenshtein 文本相似度：

| DPI | tok/s | 显存 | 与 DPI=300 的准确率 |
|-----|-------|------|---------------------|
| 100 | 54 | 7.3 GB | **100%** |
| 150 | 56 | 7.3 GB | **100%** ★ |
| 200 | 54 | 7.3 GB | **100%** |
| 250 | 54 | 7.3 GB | **100%** |
| 300 | 33 | 9.2 GB | 参照 |

所有低于 300 的 DPI 产出的文本逐字节完全一致。唯一的区别？DPI=300 慢了 38%，显存多占 2 GB。

### 根因：DeepEncoder 瓶颈

Unlimited-OCR 的处理管线如下：

```
文档 → [DPI] → 栅格图像 → DeepEncoder → 视觉 Token → 解码器 → Markdown
```

**DeepEncoder** 在 tokenization 之前将所有输入归一化到固定的 `base_size=1024` 网格。在 DPI 100–250 时，栅格化图像已经达到或超过 1024px — 因此无论 DPI 如何，编码器都产出**相同的视觉 token 集合**。

只有当 DPI=300 时，预压缩 patch 数量才会激增，导致 prefill 时间膨胀、KV cache 增加。瓶颈在编码器网格，而非原始像素数量。

对于标准办公文档（≥10pt 字号），**DPI=150 是最优选择**。只有 6pt 以下的小字或重度扫描文档才需要 DPI≥250。

---

## 发现二：页数再多，显存不涨

Unlimited-OCR 使用 **R-SWA（Reference Sliding Window Attention）** — 一种无论文档多长都能保持 KV cache 大小恒定的机制。我们通过不断增加页数跑了同一篇论文来验证：

| 页数 | 总 Token | tok/s | 显存 |
|-------|-------------|-------|------|
| 1 | 656 | 56 | 7.3 GB |
| 5 | 3,300 | 56 | 7.4 GB |
| 10 | 6,600 | 55 | 7.4 GB |
| 25 | 16,400 | 55 | 7.5 GB |
| 50 | 32,000 | 54 | 7.5 GB |

从 1 页到 50 页，显存仅增长 **+0.2 GB**。KV cache 结构如下：

```
KV[视觉 token（约 256）] + KV[最近 128 个输出 token]  ← 大小恒定
```

一张 16 GB 的消费级 Radeon 可以处理整本书。这就是 R-SWA 的威力。

---

## 发现三：文档类型不影响速度

我们测试了 4 种真实文档类型：

| 文档类型 | DPI | tok/s | 输出量 |
|--------------|-----|-------|--------|
| 英文学术论文 | 150 | 56 | 3.1 KB |
| 中文合同 | 150 | 55 | 2.8 KB |
| 手写收据 | 200 | 52 | 0.9 KB |
| 财务表格 | 150 | 54 | 4.2 KB |

吞吐量仅取决于输出 token 数量 — 与文档类型、语言或手写复杂度无关。

---

## 亲自试试

我们提供了三种体验方式：

**1. ModelScope 在线 Demo** — 零配置。上传 PDF，几秒内获取 Markdown。运行在真实 AMD GPU 上，完全免费。

**2. AMD Radeon Cloud** — 就是我们跑基准测试的同款 GPU。注册后，用自己的文件运行完整模型。从零到 OCR 只需 60 秒。[从这里开始 →](https://radeon.anruicloud.com/)

**3. 本地安装** — 如果你已经有 AMD GPU：

```bash
pip install unlimited-ocr-rocm
unlimited-ocr --pdf ./你的文档.pdf
```

---

## 下一步计划

- Instinct MI300X 基准测试
- vLLM 后端支持
- FP8 量化，进一步降低显存占用

---

## 自己动手构建

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git
cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh
source .venv/bin/activate
unlimited-ocr --pdf ./doc.pdf
```

**如果这篇内容对你有帮助，请给仓库点个 Star。也欢迎来 [AMD Radeon Cloud](https://radeon.anruicloud.com/) 复现这些数据 — 同款硬件，你自己的基准测试。**

---

→ GitHub: [github.com/AIwork4me/Unlimited-OCR-ROCm](https://github.com/AIwork4me/Unlimited-OCR-ROCm)
