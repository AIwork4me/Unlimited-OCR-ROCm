# Unlimited-OCR-ROCm vLLM Recovery Guide

## Step 1: Extract the tarball

```bash
cd /workspace
tar -xzf unlimited-ocr-rocm-vllm.tar.gz
```

Verify the git branch:
```bash
cd /workspace/Unlimited-OCR-ROCm
git branch -a | grep vllm        # must show feat/vllm-fused-moe
git checkout feat/vllm-fused-moe
```

## Step 2: Create vLLM Python environment

```bash
python3.12 -m venv /workspace/vllm-env
source /workspace/vllm-env/bin/activate

# Install PyTorch for ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0
```

## Step 3: Install vLLM ROCm nightly wheel

```bash
# Find the exact wheel URL
VLLM_WHEEL=$(curl -s https://wheels.vllm.ai/rocm/nightly/rocm721/vllm/ | \
  grep -oP 'vllm-[^<>"]+\.whl' | tail -1)
VLLM_URL="https://wheels.vllm.ai/rocm/nightly/rocm721/vllm/$VLLM_WHEEL"

# Install without deps (wheel pins internal-only torch/trliton-kernels versions)
pip install --no-deps "$VLLM_URL"
```

## Step 4: Install vLLM dependencies

```bash
pip install \
  transformers tokenizers fastapi pydantic uvicorn \
  aiohttp openai tiktoken numpy pillow requests psutil \
  huggingface_hub gguf xgrammar msgspec pyzmq cbor2 blake3 \
  cachetools protobuf sentencepiece diskcache lark llguidance \
  outlines_core prometheus_client prometheus-fastapi-instrumentator \
  pyyaml tqdm einops safetensors pycountry packaging \
  setproctitle python-json-logger grpcio cloudpickle pybase64 \
  soundfile compressed-tensors numba scipy py-cpuinfo amdsmi ninja \
  openai-harmony mcp
```

## Step 5: Apply vLLM Unlimited-OCR model patches

```bash
VLLM_SITE=$(python -c "import vllm; print(vllm.__file__.rsplit('/',1)[0])")

cp patches/vllm/unlimited_ocr.py \
   $VLLM_SITE/model_executor/models/
cp patches/vllm/configs/unlimited_ocr.py \
   $VLLM_SITE/transformers_utils/configs/
cp patches/vllm/processors/unlimited_ocr.py \
   $VLLM_SITE/transformers_utils/processors/
```

## Step 6: Download model weights

Choose one:

```bash
# Option A: ModelScope (recommended for China)
git clone https://www.modelscope.cn/baidu/Unlimited-OCR.git /workspace/models/Unlimited-OCR

# Option B: HuggingFace
huggingface-cli download baidu/Unlimited-OCR \
  --local-dir /workspace/models/Unlimited-OCR
```

## Step 7: Verify imports

```bash
python -c "
import vllm; print('vLLM:', vllm.__version__)
import torch; print('torch:', torch.__version__)
print('HIP:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
from vllm.model_executor.models.unlimited_ocr import NGramPerReqLogitsProcessor, UnlimitedOCRForCausalLM
print('Unlimited-OCR model: OK')
print('NGramPerReqLogitsProcessor: OK')
"
```

Expected output:
```
vLLM: 0.20.2rc1.dev15+g...
torch: 2.10.0+rocm7.0
HIP: True AMD Radeon Graphics
Unlimited-OCR model: OK
NGramPerReqLogitsProcessor: OK
```

## Step 8: Serve and test inference

```bash
# Start server (background, avoid HF network calls)
export HF_HUB_OFFLINE=1
export HIP_VISIBLE_DEVICES=0

nohup vllm serve /workspace/models/Unlimited-OCR \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --port 10000 \
  --host 0.0.0.0 \
  --enforce-eager \
  > /tmp/vllm.log 2>&1 &

# Wait for server ready (model loading takes 5-10 min)
while ! curl -s http://localhost:10000/health 2>/dev/null | grep -q .; do
  echo "$(date): waiting..."
  sleep 10
done
echo "$(date): SERVER READY"

# Test single-page OCR
TEST_IMG=$(ls /workspace/OmniDocBench_data/images/*.png | head -1)
IMAGE_B64=$(base64 -w 0 "$TEST_IMG")

curl -s http://localhost:10000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"baidu/Unlimited-OCR\",
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"text\", \"text\": \"<image>document parsing.\"},
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMAGE_B64\"}}
    ]}],
    \"temperature\": 0.0,
    \"max_tokens\": 4096,
    \"skip_special_tokens\": false
  }" | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d['choices'][0]['message']['content']
print(f'Output: {len(text)} chars')
print(text[:500])
"
```

Expected: reasonable markdown OCR output (not empty, not BOS-loop, not "!!!!!").

## Step 9: Run full OmniDocBench eval (after verifying step 8)

```bash
# Single GPU test (10 pages)
python3 scripts/run_omnidocbench_vllm.py \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --output-dir /tmp/vllm-test \
  --base-url http://localhost:10000 \
  --limit 10

# Check output
cat /tmp/vllm-test/*.md 2>/dev/null | wc -l

# Full 4-GPU eval
bash scripts/run_omnidocbench_vllm_4gpu.sh
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'X'` | `pip install X` — the dep list in Step 4 may be incomplete |
| `/health` never responds | Check `/tmp/vllm.log` for errors; may need longer wait (up to 15 min first load) |
| Output is empty | Prompt must start with `<image>`; ensure `skip_special_tokens: false` |
| Output is BOS-loop | Ensure `--logits_processors` flag is set; check stderr for `NGramPerReqLogitsProcessor` loaded |
| `HIP out of memory` | Reduce `--gpu-memory-utilization` to 0.85 |
| `torch.cuda.is_available() == False` | Check ROCm: `rocm-smi`; reinstall torch from ROCm index |

## Document References

- Spec: `docs/superpowers/specs/2026-07-09-vllm-rocm-omnidocbench-alignment-design.md`
- Plan: `docs/superpowers/plans/2026-07-09-vllm-rocm-alignment.md`
- Handoff: `docs/superpowers/HANDOFF-vllm-rocm-2026-07-09.md`
- vLLM recipe: https://recipes.vllm.ai/baidu/Unlimited-OCR
