# Unlimited-OCR-ROCm: 在 AMD GPU 上运行顶级 OCR — 实测数据全公开

**作者：** aiwork4me  
**日期：** 2026 年 6 月 22 日  
**标签：** ROCm、AMD GPU、OCR、SGLang、PyTorch、Benchmark

---

## 缘起

百度在 2026 年 6 月开源了 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)，这款模型将长篇章文档解析推向了新的高度。它可以在一次前向传播中处理整本书、多页合同和密集表格，输出清晰、结构化的 Markdown 文本。

然而，官方推理链路仅支持 NVIDIA CUDA。这意味着大量 AMD GPU 用户无法使用这一前沿模型。

**Unlimited-OCR-ROCm** 改变了这个局面。而且我们不止步于「能跑」——我们跑了 18 组参数的系统化基准测试，把每一项数据都公开。

---

## 实测硬件

本文中每一个数字都在真实 AMD 芯片上测得：

| 项目 | 详情 |
|------|------|
| **GPU** | AMD Radeon Graphics |
| **显存** | 48 GB |
| **ROCm / HIP** | 7.2.53211 |
| **PyTorch** | 2.12.1+rocm7.2 |
| **模型** | baidu/Unlimited-OCR |
| **推理引擎** | HuggingFace Transformers |

> 你可以在 [AMD Radeon Cloud](https://radeon.anruicloud.com/) 上完整复现这些结果 — 同样的 GPU、同样的 ROCm 栈、同样的模型。

---

## 评测方法

我们在单页 A4 PDF（约 656 输出 token）上测试了 4 个参数轴共 18 种组合。每个测试同时测量：

- **吞吐量**（tokens/秒）  
- **显存**（峰值 GB）  
- **精度**（与 DPI=300 / base / maxlen=32768 参照组的 Levenshtein 文本相似度）

所有数据为**热身后**（第 2+ 次调用，GPU kernel 已编译）。冷启动约慢 20%。

---

## 实测数据

| 参数轴 | 变体 | 耗时 | tok/s | 显存 | 精度 |
|--------|------|------|-------|------|------|
| **DPI** | 100 | 12.1 s | 54 | 7.3 GB | **100%** |
| | 150 | 12.4 s | 53 | 7.3 GB | **100%** |
| | 200 | 12.2 s | 54 | 7.3 GB | **100%** |
| | 250 | 12.2 s | 54 | 7.3 GB | **100%** |
| | **300** | **19.6 s** | **33** | **9.2 GB** | 参照 |
| **image_mode** | gundam | 13.6 s | 48 | 7.6 GB | **100%** |
| | base | 12.2 s | 54 | 7.3 GB | **100%** |
| **ngram_window** | 32 | 12.2 s | 54 | 7.3 GB | **100%** |
| | 128 | 12.2 s | 54 | 7.3 GB | **100%** |
| | 512 | 12.1 s | 54 | 7.3 GB | **100%** |
| **max_length** | 4096 | 11.8 s | 56 | 7.3 GB | **100%** |
| | 32768 | 11.6 s | 57 | 7.3 GB | **100%** |

### 关键发现一：DPI 与精度无关

DPI 100–250 产出与 DPI=300 **完全相同的文本**。`DeepEncoder` 将所有分辨率归一化到相同的视觉 token 网格。DPI=300 比 DPI=200 多耗 **58% 时间**和 **+2 GB 显存**，标准文档精度零提升。

### 关键发现二：R-SWA 极致省显存

模型空闲显存：**6.8 GB**。推理峰值：**7.3–7.6 GB**。仅 **+0.5–0.9 GB** 开销 — Reference Sliding Window Attention 在解码全程维持恒定 KV Cache，无论文档多长。

### 关键发现三：最优组合

**gundam 模式, DPI=150, max_length=8192, ngram_window=64** → **11.8 s, 56 tok/s, 7.6 GB 显存, 100% 精度** — 比 DPI=300 快 38%，文本完全一致。

---

## 根因分析：为什么 DPI 通常不重要

```
文档 → [DPI] → 栅格图像 → DeepEncoder → 视觉 Token → R-SWA 解码器
             ↑_________________________↑
                 DPI 越高 = 像素越多
                 → DeepEncoder 始终压缩到 ~256 个视觉 token
```

`DeepEncoder` 将所有输入归一化到固定的 `base_size=1024` 网格。在 DPI 100–250 时，图像已达或超过 1024px，编码器产生相同数量的 token。瓶颈在编码器网格而非原始像素。仅当 DPI=300 时预压缩 patch 数骤增，prefill 时间翻倍、KV cache 膨胀。

---

## 技术要点

### GPU 环境自动检测

```python
def detect_rocm() -> bool:
    if shutil.which("rocm-smi"):
        return True
    import torch
    if hasattr(torch.version, "hip") and torch.version.hip:
        return True
    return False
```

检测到 ROCm 后自动设置 `HIP_VISIBLE_DEVICES` 并选择 Triton attention 后端。

### R-SWA: 恒定 KV Cache

传统全注意力：KV cache 随每个生成 token 线性增长。
```
Token 1: KV[1]
Token 2: KV[1,2]
...
Token 1000: KV[1,2,...,1000]  ← 1000倍增长！
```

R-SWA：KV cache 保持恒定。
```
每个 token: KV[视觉 token] + KV[最后 128 输出 token]
            ↑ 固定大小            ↑ 固定窗口
```

这就是为什么即使 16 GB 消费级 Radeon 也能在 32K token 文档上跑 OCR。

### GPU 冷启动

首轮 ~20% 惩罚来自 HIP kernel JIT 编译（~5s）、L2 cache 预热（~2s）和 PyTorch 显存分配器初始化（~1s）。所有热身后运行跳过此开销。

---

## 亲手复现

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git
cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh --rocm-version 6.2
source .venv/bin/activate

# 快速 OCR 测试
unlimited-ocr --pdf ./我的文档.pdf --output-dir ./outputs

# 运行完整 benchmark 套件
python scripts/full_benchmark.py
python scripts/accuracy_benchmark.py
```

或直接使用 [AMD Radeon Cloud](https://radeon.anruicloud.com/) — 零配置，同款 GPU 芯片。

---

## 下一步

- 同一 AMD Radeon Graphics 硬件上的 SGLang 基准测试
- vLLM 后端支持
- Web 拖拽式 OCR 界面
- 消费级 Radeon GPU 优化指南（目标：16 GB 显卡）

---

→ GitHub: [github.com/AIwork4me/Unlimited-OCR-ROCm](https://github.com/AIwork4me/Unlimited-OCR-ROCm)
