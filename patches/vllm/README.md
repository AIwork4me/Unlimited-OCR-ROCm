# vLLM Unlimited-OCR Model Patches

These files are extracted from the vLLM main branch
(https://github.com/vllm-project/vllm) because the ROCm rocm721
nightly wheel (v0.20.2) predates the Unlimited-OCR model registration
(which requires vLLM 0.25.0+).

## Background

Per the official vLLM recipe at https://recipes.vllm.ai/baidu/Unlimited-OCR,
vLLM has native support for `baidu/Unlimited-OCR` including:
- `UnlimitedOCRForCausalLM` model class (thin wrapper around `DeepseekOCRForCausalLM`)
- `NGramPerReqLogitsProcessor` logits processor (n-gram anti-repetition)

These were added in vLLM 0.25.0+, but the ROCm rocm721 nightly channel
is currently at v0.20.2. The base `deepseek_ocr.py` support required by
Unlimited-OCR *is* present in v0.20.2 — only these three wrapper files
are missing.

## Installation

After installing vLLM from the ROCm nightly wheel:

```bash
# Copy to vLLM site-packages
VLLM_SITE=$(python -c "import vllm; print(vllm.__file__.rsplit('/',1)[0])")

cp patches/vllm/unlimited_ocr.py $VLLM_SITE/model_executor/models/
cp patches/vllm/configs/unlimited_ocr.py $VLLM_SITE/transformers_utils/configs/
cp patches/vllm/processors/unlimited_ocr.py $VLLM_SITE/transformers_utils/processors/
```

## Serving

```bash
vllm serve /path/to/baidu/Unlimited-OCR \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```

## Source

Extracted from https://github.com/vllm-project/vllm main branch on 2026-07-09.
