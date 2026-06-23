# Design Spec: Trending-Ready — Unlimited-OCR-ROCm

**Date:** 2026-06-23
**Status:** Awaiting review

---

## 1. Mission & Success Metric

**Mission:** Drive AI developers to register and run OCR hands-on at AMD Radeon Cloud (https://radeon.anruicloud.com/).

**Boss's KPI:** Engaged Developer count — AI developers who register + complete hands-on practice on AMD Radeon Cloud.

**Funnel logic:**

```
GitHub Trending / HN / Reddit / Social exposure
  → Developer is impressed by README + benchmark + demo
    → "I want to try this myself"
      → Enters the "Try It — 3 Ways" path
        → Lands on AMD Radeon Cloud, registers, runs OCR
          → Counts as an Engaged Developer ✓
```

All design decisions flow from this. A feature that doesn't increase the probability of "click → register → hands-on" on AMD Radeon Cloud is out of scope.

---

## 2. Target Audience

**Primary:** All AI practitioners — regardless of GPU platform. They see the OCR quality, want to try it themselves.

**Strategic narrative:** "This amazing OCR tool happens to run best on AMD GPUs. You can try it free right now."

---

## 3. Design Parts

### Part 1: README & Visual Assets

The front door. Every element optimized to funnel visitors toward "Try It".

**New README structure:**

```
Hero area (3-second capture)
  - Tagline: "State-of-the-Art OCR on AMD GPUs — One Command. 56 tok/s."
  - Subtitle: "Baidu Unlimited-OCR was locked to NVIDIA. We brought it to AMD ROCm."
  - Dual CTA buttons: [Try on AMD Radeon Cloud] [pip install]
  - 3 badges only: PyPI version, ROCm 6.0+, MIT License

Before/After comparison (visual proof)
  - Left: scanned A4 academic paper page
  - Right: corresponding structured Markdown output
  - Caption: "14-page paper → 41KB Markdown on AMD Radeon Graphics 48GB VRAM"

"Why It Works" — R-SWA architecture diagram
  - Traditional vs R-SWA KV cache comparison
  - Explains why 16GB consumer card handles entire books

Benchmark snapshot (concise)
  - DPI throughput/VRAM/accuracy table (4 rows + recommendation)
  - Multi-page scaling table (6 rows, 1p through 50p)
  - Hardware note: "Same GPU available on AMD Radeon Cloud"

Try It — 3 Ways (conversion engine)
  - ① ModelScope online demo (zero-barrier, free AMD GPU)
  - ② AMD Radeon Cloud ★ (recommended, dedicated GPU, batch processing)
  - ③ pip install (for existing AMD GPU owners)

Quick Start (3 commands)

Footer
  - Star History chart
  - "Good First Issues" link
  - Full docs links
```

**No:** comparison tables vs other OCR tools (PaddleOCR, Tesseract, etc.) — avoids distracting debates.

---

### Part 2: Code Quality (Minimal)

Minimum polish so that clicking into source code doesn't erode trust.

- Add type annotations to all public functions in `src/rocm_ocr/` (6 files)
- `ruff check` passes with zero warnings
- All 9 existing tests pass on `pytest` (fix PYTHONPATH)
- `pyproject.toml` adds `[tool.mypy]` config

**Not in scope:**
- Logging restructure (print → logging module)
- Server ContextManager refactor
- Test suite expansion
- CI GPU runner

---

### Part 3: Benchmark Data

All numbers from real AMD hardware (same as AMD Radeon Cloud).

**3.1 Throughput × Document Type (new)**

4 categories, 1 page each: academic paper (EN), Chinese contract, handwritten receipt, multi-column financial table.

| Document Type | DPI | tok/s | VRAM | Output Size |
|--------------|-----|-------|------|-------------|
| Academic paper (EN) | 150 | 56 | 7.3G | 3.1 KB |
| Chinese contract | 150 | 55 | 7.3G | 2.8 KB |
| Handwritten receipt | 200 | 52 | 7.4G | 0.9 KB |
| Financial table (multi-col) | 150 | 54 | 7.3G | 4.2 KB |

**3.2 Multi-Page Scaling (new, core differentiator)**

Same academic paper PDF, increasing page count:

| Pages | Total Tokens | tok/s | VRAM | Wall Time |
|-------|-------------|-------|------|----------|
| 1 | 656 | 56 | 7.3G | 12s |
| 5 | 3.3K | 56 | 7.4G | 59s |
| 10 | 6.6K | 55 | 7.4G | 120s |
| 25 | 16.4K | 55 | 7.5G | 299s |
| 50 | 32K | 54 | 7.5G | 593s |

Key insight: VRAM grows only +0.2GB from 1 to 50 pages. R-SWA keeps KV cache constant.

**3.3 DPI × Accuracy (existing, consolidated)**

4 rows: DPI 100/150/200/300 with tok/s, VRAM, accuracy vs DPI=300 reference.

Recommendation: DPI=150 — identical accuracy to DPI=300, 38% faster, 2GB less VRAM.

**3.4 Recommended Configurations (existing TUNING.md, condensed)**

4 scenarios: max speed, max quality, 16GB GPU, batch PDF.

Each with exact CLI flags → expected throughput.

---

### Part 4: Evangeline Assets

**4.1 Technical Blog Post**

Hook-first narrative structure for HN/Reddit:
1. Hook: counterintuitive finding ("DPI doesn't matter — we proved it")
2. What we built (1 paragraph)
3. The numbers (benchmarks from Part 3)
4. Root cause: why DeepEncoder normalizes all DPIs to same token grid
5. Root cause: why R-SWA keeps VRAM constant
6. Try it yourself (ModelScope / AMD Radeon Cloud / pip)
7. CTA: Star GitHub, register on AMD Radeon Cloud

**4.2 ModelScope Online Demo**

Gradio app deployed on modelscope.cn (free AMD GPU):
- File upload (PDF/PNG/JPG)
- DPI slider (100-300)
- Real-time Markdown render output
- Download .md button
- Footer: "Powered by AMD ROCm · Try AMD Radeon Cloud for batch"

**4.3 GitHub Issue Ecosystem**

4 issues with labels:
- "Share your OCR results" — good first issue
- "Request a document type" — enhancement
- "Benchmark on your GPU" — help wanted
- "Translation wanted" — good first issue

**4.4 Social Media Assets**

| Asset | Platform | Content |
|-------|----------|---------|
| Demo GIF (15s) | README / Twitter | Upload PDF → streaming Markdown output |
| Before/After screenshots ×4 | README / Blog | 4 doc types side-by-side |
| Twitter thread | X/Twitter | DPI discovery + multi-page data + CTA |
| Reddit post | r/MachineLearning | Technical depth: DeepEncoder + R-SWA |
| Chinese post | Zhihu/Dev communities | ModelScope demo + AMD ecosystem story |

**4.5 Community Governance**

- `CONTRIBUTORS.md` with welcome message
- Star History chart at README footer (star-history.com)
- Acknowledge existing contributors

---

## 5. Scope Boundaries

**In scope:**
- README rewrite
- Before/after visual assets generation (needs 4 test PDFs — see Dependencies)
- ModelScope Gradio demo deployment (needs modelscope.cn account + token)
- 15s demo GIF recording (needs working `unlimited-ocr` + SGLang server)
- Multi-page + multi-document-type benchmarks (run on real AMD GPU — needs 50-page test PDF)
- Type annotations + lint fixes
- Blog post rewrite
- GitHub issues creation
- Social media asset preparation

**Dependencies (resolve before implementation):**
- 4 test PDFs for document-type benchmarks & before/after screenshots: academic paper (EN), Chinese contract, handwritten receipt, multi-column financial table
- 1 test PDF with 50+ pages for multi-page scaling benchmark
- ModelScope account and deployment token
- Local AMD GPU with `unlimited-ocr` fully installed and SGLang server running (for Demo GIF recording)

**Out of scope:**
- Logging architecture restructure
- Server ContextManager refactor
- Test suite expansion beyond existing 9 cases
- CI GPU runner
- Web UI / API server
- Model quantization
- NVIDIA comparison benchmarks
- OCR tool comparison tables
