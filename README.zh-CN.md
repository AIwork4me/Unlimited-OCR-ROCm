# Unlimited-OCR-ROCm

在 AMD ROCm GPU 上本地运行 Unlimited-OCR，将文档图片转换为 Markdown。

基于 SGLang/vLLM 推理后端，支持 gfx1100 (RDNA3)。

## Install（安装）

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git
cd Unlimited-OCR-ROCm
pip install -e ".[dev]"
# 平台集成（可选）：
pip install -e ".[platform]"
```

## Demo（演示）

无需 GPU 的 smoke 后端可以端到端验证 adapter 合同：

```bash
python adapter/run_adapter.py --img-dir examples --out-dir /tmp/out --platform linux-rocm --backend smoke
```

## Evaluation（评测）

OmniDocBench-ROCm 平台评测使用 `omnidocbench-rocm`：

```bash
omnidocbench-rocm run \
  --stage all \
  --platform linux-rocm \
  --version v16 \
  --revision 2b161d0 \
  --adapter adapter/run_adapter.py \
  --model-id unlimited-ocr \
  --backend sglang \
  --server-url http://127.0.0.1:30000 \
  --git-commit "$(git rev-parse HEAD)" \
  --results-dir results/omnidocbench/v16/linux-rocm \
  --cdm
```

## Reproducibility（可复现性）

硬件：AMD gfx1100 (Radeon PRO W7900)，48 GB VRAM，ROCm 7.2。
评测结果保存于 `eval/results/` 目录。平台标准 artifacts 通过
`omnidocbench-rocm score` + `publish` 生成。

## Known Gaps（已知限制）

- `windows-hip` 平台仍为 `community-wanted`，尚无正式结果
- 平台标准 artifacts 尚未生成（等待 score/publish 执行）
- SGLang 推理后端需要预先启动的 server
- 完整列表见 [README.md](README.md)
