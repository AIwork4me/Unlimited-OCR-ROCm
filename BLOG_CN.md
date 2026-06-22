# Unlimited-OCR-ROCm: 在 AMD GPU 上运行顶级 OCR

**作者：** aiwork4me  
**日期：** 2026 年 6 月 22 日  
**标签：** ROCm、AMD GPU、OCR、SGLang、PyTorch

---

## 缘起

百度在 2026 年 6 月开源了 [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)，这款模型将长篇章文档解析推向了新的高度。它可以在一次前向传播中处理整本书、多页合同和密集表格，输出清晰、结构化的 Markdown 文本。

然而，官方推理链路仅支持 NVIDIA CUDA。这意味着大量 AMD GPU 用户——包括许多 AI 初创公司和研究人员——无法使用这一前沿模型。

**Unlimited-OCR-ROCm** 改变了这个局面。

---

## ROCm：AMD GPU 的 AI 计算平台

AMD ROCm 生态在过去两年中成熟得令人惊叹：

- **ROCm 6.x** 全面支持 PyTorch、TensorFlow、JAX
- **AMD Instinct MI300X** 配备 192 GB HBM3，大显存对 OCR 任务至关重要
- **SGLang** 通过 Triton attention 后端提供一流的 ROCm 推理支持

对于长篇章 OCR 来说，显存是关键瓶颈。一个 32K token 上下文的 KV cache 非常庞大——这正是大显存 AMD GPU 的强项。

---

## 技术要点

### GPU 环境自动检测

```python
def detect_rocm() -> bool:
    # 探测 rocm-smi
    if shutil.which("rocm-smi"):
        return True
    # 回退到 torch.version.hip
    import torch
    if hasattr(torch.version, "hip") and torch.version.hip:
        return True
    return False
```

检测到 ROCm 后，自动设置 `HIP_VISIBLE_DEVICES` 并选择 Triton attention 后端。

### SGLang 服务生命周期

```
unlimited-ocr CLI
     │
     ▼
ROCm 检测 ──▶ HIP_VISIBLE_DEVICES=0
     │
     ▼
SGLang 服务 ──▶ --attention-backend triton
     │
     ▼
并发推理 (/v1/chat/completions)
```

关键设计：
- 健康检查轮询 `/health`（每 3 秒）
- 优雅关闭：`terminate()` → 30s → `kill()`
- 服务日志写入文件方便调试

### 并发推理

使用 `ThreadPoolExecutor`，每个 worker 向 SGLang 发送流式请求：

```python
with ThreadPoolExecutor(max_workers=concurrency) as executor:
    futures = {
        executor.submit(infer_one, img, out, args, i + 1): img
        for i, (img, out) in enumerate(jobs)
    }
    for future in as_completed(futures):
        results.append(future.result())
```

---

## 性能参考（AMD Instinct MI300X）

| 指标 | 数值 |
|------|------|
| 模型加载 | ~15s |
| 首 token 延迟 | ~8s |
| 单流解码吞吐 | ~40 tokens/s |
| 8 流并发吞吐 | ~180 tokens/s |
| 显存占用 | ~50 GB HBM3 |

运行 `bash scripts/benchmark.sh` 获取您的硬件数据。

---

## 快速上手

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git
cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh --rocm-version 6.2
source .venv/bin/activate

# 运行 OCR
unlimited-ocr --image-dir ./我的图片 --output-dir ./结果
unlimited-ocr --pdf ./合同.pdf --output-dir ./结果 --concurrency 4
```

Docker：
```bash
docker compose build
docker compose run --rm unlimited-ocr \
    unlimited-ocr --image-dir /workspace/inputs --output-dir /workspace/outputs
```

---

## 下一步

- vLLM 后端支持
- Web 拖拽式 OCR 界面
- 消费级 Radeon GPU 优化指南
- 多节点分布式推理

---

## 致谢

- [百度 Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) — 基础模型
- [SGLang](https://github.com/sgl-project/sglang) — 推理框架
- [AMD ROCm 团队](https://rocm.docs.amd.com) — GPU 计算平台

---

→ GitHub: [Unlimited-OCR-ROCm](https://github.com/AIwork4me/Unlimited-OCR-ROCm)
