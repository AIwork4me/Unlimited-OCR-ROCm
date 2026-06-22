# Examples / 示例

## 两种推理方式

| 方式 | 脚本 | 适用场景 |
|------|------|----------|
| **Transformers** | `transformers_infer.py` | 快速测试、单张图片、小批量 |
| **SGLang** | `sglang_server.sh` + `sglang_client.py` | 批量处理、生产部署、高吞吐 |

---

## 方式一：Transformers（最简单）

单脚本，直接加载模型运行 OCR。

### 安装依赖
```bash
# 安装 ROCm PyTorch
pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
    torch torchvision torchaudio

# 安装其他依赖
pip install transformers Pillow einops addict easydict pymupdf psutil
```

### 运行
```bash
# 单张图片
python examples/transformers_infer.py \
    --image ./photo.jpg \
    --mode gundam \
    --output-dir ./outputs

# PDF 文档
python examples/transformers_infer.py \
    --pdf ./document.pdf \
    --mode base \
    --output-dir ./outputs
```

### Python API
```python
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "baidu/Unlimited-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# 单张图片 — gundam 模式（局部细节）
model.infer(
    tokenizer,
    prompt="<image>document parsing.",
    image_file="photo.jpg",
    output_path="./outputs",
    base_size=1024, image_size=640, crop_mode=True,
    max_length=32768,
    no_repeat_ngram_size=35, ngram_window=128,
    save_results=True,
)

# PDF / 多页 — base 模式（整页上下文）
model.infer_multi(
    tokenizer,
    prompt="<image>Multi page parsing.",
    image_files=["page1.png", "page2.png", "page3.png"],
    output_path="./outputs",
    image_size=1024,
    max_length=32768,
    no_repeat_ngram_size=35, ngram_window=1024,
    save_results=True,
)
```

---

## 方式二：SGLang（生产环境）

适合批量处理和服务化部署。

### 第一步：安装环境
```bash
# 安装 ROCm PyTorch
pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
    torch torchvision torchaudio

# 安装 SGLang
pip install "sglang[all]>=0.4.0" kernels>=0.11.0 pymupdf requests
```

### 第二步：启动服务
```bash
bash examples/sglang_server.sh
```
等待输出 `The server is ready`（约 30-90 秒）。

### 第三步：发送请求
```bash
# 单张图片
python examples/sglang_client.py --image ./photo.jpg --mode gundam

# PDF 文档
python examples/sglang_client.py --pdf ./document.pdf
```

### 第四步：停止服务
```bash
pkill -f sglang.launch_server
```

### Python API
```python
import base64, json, requests
from sglang.srt.sampling.custom_logit_processor import (
    DeepseekOCRNoRepeatNGramLogitProcessor,
)

# 编码图片
with open("photo.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "Unlimited-OCR",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "document parsing."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]}],
    "temperature": 0,
    "stream": True,
    "images_config": {"image_mode": "gundam"},
    "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
    "custom_params": {"ngram_size": 35, "window_size": 128},
}

response = requests.post(
    "http://127.0.0.1:10000/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json=payload, stream=True,
)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        chunk = json.loads(line[6:])
        text = chunk["choices"][0]["delta"].get("content", "")
        if text:
            print(text, end="", flush=True)
```

---

## 图片模式

| 模式 | 分辨率 | 说明 |
|------|--------|------|
| `gundam` | 640×640 | 裁剪局部细节，适合收据/表格 |
| `base` | 1024×1024 | 整页上下文，适合多页/PDF |

---

## 常见问题

**Q: `ModuleNotFoundError: No module named 'sglang'`**
```bash
pip install "sglang[all]>=0.4.0"
```

**Q: SGLang 启动失败 "No HIP GPUs available"**
```bash
rocm-smi --showproductname
export HIP_VISIBLE_DEVICES=0
```

**Q: 显存溢出 (OOM)**
- 降低 `mem_fraction_static`（默认 0.8 → 0.6）
- 使用 `--image-mode base`

---

## Two Inference Methods

| Method | Script | Best For |
|--------|--------|----------|
| **Transformers** | `transformers_infer.py` | Quick testing, single images |
| **SGLang** | `sglang_server.sh` + `sglang_client.py` | Batch processing, production |

### Method 1: Transformers (Quick Start)

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
    torch torchvision torchaudio
pip install transformers Pillow einops addict easydict pymupdf psutil

python examples/transformers_infer.py --image ./photo.jpg --mode gundam
python examples/transformers_infer.py --pdf ./document.pdf --mode base
```

### Method 2: SGLang (Production)

```bash
pip install "sglang[all]>=0.4.0" kernels>=0.11.0 pymupdf requests

# Terminal 1: start server
bash examples/sglang_server.sh

# Terminal 2: send requests
python examples/sglang_client.py --image ./photo.jpg --mode gundam
python examples/sglang_client.py --pdf ./document.pdf
```

### Image Modes

| Mode | Resolution | Use |
|------|-----------|-----|
| `gundam` | 640×640 | Receipts, forms |
| `base` | 1024×1024 | Multi-page, PDFs |
