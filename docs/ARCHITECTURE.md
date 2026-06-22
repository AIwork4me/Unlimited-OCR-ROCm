# Architecture & Root Cause Analysis

## Model Architecture

Unlimited-OCR is a vision-language model consisting of:

```
Document → [DPI] → Raster Image → DeepEncoder → Visual Tokens → R-SWA Decoder → Markdown
```

### DeepEncoder

A high-compression vision encoder that processes images into a fixed-size set of visual tokens. Key properties:

- **Resolution-agnostic output**: regardless of input DPI (100–300), the encoder compresses to ~256 visual tokens for a standard page
- **Base size = 1024**: encoder normalizes to a 1024×1024 grid
- This is why DPI 100–200 produce identical text: the encoder's compression is the bottleneck, not the raw pixel count

### R-SWA (Reference Sliding Window Attention)

The core innovation of Unlimited-OCR. Standard full attention has O(n²) KV cache growth. R-SWA maintains **constant** KV cache:

```
Traditional:  KV[token_1, token_2, ..., token_1000]  ← 1000× growth
R-SWA:        KV[visual_tokens] + KV[last_128_output_tokens]
              ↑ fixed (~256 tokens)   ↑ fixed (128 tokens)
```

This is why inference VRAM overhead is only **+0.5–0.9 GB** regardless of document length.

## Root Cause Analysis per Parameter

### DPI: The Image-to-Token Pipeline

- **DPI 100–200 cluster**: `DeepEncoder` normalizes to `base_size=1024`. At these DPIs, the image is already at or above 1024px, so the encoder compresses to a similar token count. Bottleneck = encoder grid, not raw pixels.
- **DPI=300 threshold**: raw image exceeds 3000px, encoder processes significantly more patches before compression. Visual token count inflates, tripling prefill time and adding 1.6 GB to KV cache.
- **Accuracy invariance**: Since encoder output is resolution-normalized, text accuracy is identical at DPI 100–300 for ≥10pt fonts. Only sub-6pt fonts or heavy noise benefit from DPI≥250.

### ngram_window: String Check, Not Attention

The `ngram_window` parameter controls **no-repeat-ngram suppression** — a lightweight string-deduplication on already-generated text. It does NOT affect:
- Attention computation (that's architecturally fixed to 128 output tokens in R-SWA)
- KV cache size
- Visual token processing

This is why ngram_window 32 vs 512 shows zero measurable difference in speed, VRAM, OR accuracy. You're tuning a string check, not a matrix multiply.

### max_length: Pre-allocation, Not Computation

`max_length` only reserves an upper bound for the generation buffer. For short pages (~656 tokens), the model stops decoding before hitting the limit. The only cost is memory reservation — no computational overhead.

### image_mode: Visual Token Density

| Mode | Input Size | Crop | Visual Tokens | Best For |
|------|-----------|------|---------------|----------|
| gundam | 640×640 | Yes | ~256 | Local details (receipts, tables) |
| base | 1024×1024 | No | ~256 | Global layout (multi-column, PDF) |

After DeepEncoder compression, both modes produce roughly the same visual token count. The difference is in **what** the tokens represent: gundam crops to a local region (higher detail density), base preserves full-page layout. Throughput is nearly identical because computation is bottlenecked by token count, not pixel count.

### GPU Cold Start

First-run ~20% penalty from three sources:
1. **HIP kernel compilation** (~5s): Triton and PyTorch JIT-compile attention kernels on first use
2. **L2 cache warmup** (~2s): GPU cache hierarchy is empty on first invocation
3. **Memory allocator priming** (~1s): PyTorch caching allocator initializes memory pools

Subsequent runs skip all three.

## When Accuracy Drops

18/18 parameter combinations scored 100% accuracy against DPI=300 reference on a clean A4 PDF. Edge cases:

| Scenario | Reason | Fix |
|----------|--------|-----|
| Sub-6pt font | DPI=100 may blur glyphs beyond recognition | DPI ≥ 250 |
| Scanned/noisy docs | Low DPI amplifies JPEG/scan artifacts | DPI = 300 |
| Wide documents | gundam's 640×640 crop loses right columns | Use base mode |
| Repetitive content | ngram_window < 32 may suppress valid rows | ngram_window ≥ 128 |
| Documents >10 pages | max_length too low truncates output | max_length = 32768 |
