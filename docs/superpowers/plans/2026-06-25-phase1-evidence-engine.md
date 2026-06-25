# Phase 1 — Evidence Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Unlimited-OCR-ROCm's accuracy claims verifiable on the OmniDocBench standard benchmark (v1.5 + v1.6), and reform the project's public face (README + community docs) to lead with credibility — the wedge for Phase 2 upstream adoption.

**Architecture:** Two clusters. **Part A (docs/community)** — no GPU needed: public `ROADMAP.md`, community benchmarks page, OmniDocBench parity-report template. **Part B (eval harness)** — a new `rocm_ocr.omnidocbench` module that runs our SGLang inference (`rocm_ocr.infer`) over OmniDocBench page images, writes one `.md` prediction per page in OmniDocBench's expected format, invokes the official `pdf_validation.py` scorer, and parses the result summary. All code is unit-testable with mocks; the actual benchmark *run* requires an AMD GPU + the 1.55 GB dataset and is deferred to the maintainer.

**Tech Stack:** Python ≥3.10, `rocm_ocr` (SGLang client), PyYAML, pytest; OmniDocBench evaluator (`opendatalab/OmniDocBench`, Apache-2.0) invoked as an external subprocess.

## Global Constraints

Copied verbatim from the design spec + verified research. Every task implicitly includes these.

- **OmniDocBench versions:** support BOTH **v1.5 and v1.6**. Versions are git *branches* (`main`=v1.6, `v1_5`=v1.5), not tags. Evaluating both = clone both branches, run the scorer twice. v1.5↔v1.6 deltas are NOT strictly comparable (annotation `anno_id` change + new MGAM matcher) — always label results with the exact branch + dataset version.
- **Credibility-first / no fabrication:** never invent benchmark numbers. Runtime-filled numeric cells must be explicit placeholders ("populate via `make eval`"), never fake values.
- **Cloud CTA stays demoted:** AMD Radeon Cloud remains a secondary "no local GPU" option, never the hero (README already restructured in commit `eb1a57d`).
- **OmniDocBench evaluator contract (verified):** model-agnostic — reads ONE `{image_basename}.md` per page (missing ⇒ page scores 0, no crash). Content = whole-page Markdown (HTML `<table>` tables, `\[...\]` display formulas, `\(...\)` inline, figures ignored), with markdown fences stripped. Scorer = `python pdf_validation.py --config configs/end2end.yaml`. Output `result/{save_name}_run_summary.json`; `save_name = basename(prediction.data_path) + "_" + match_method`.
- **Overall formula (verified):** `((1 − Text EditDist)×100 + Table TEDS + Formula CDM) / 3`. Four modules: `text_block`(Edit_dist), `display_formula`(Edit_dist+CDM), `table`(TEDS+TEDS_structure_only+Edit_dist), `reading_order`(Edit_dist).
- **Parity bar:** match the NVIDIA-reference run of Unlimited-OCR (≈93.92 v1.6 self-reported), NOT the board SOTA (95.75, MinerU2.5-Pro). Unlimited-OCR ≠ DeepSeek-OCR.
- **Branch:** `docs/top-tier-strategy-spec` (off `main`). Commit per task. Never commit to `main`.
- **Runtime deferred:** actually running the eval (AMD GPU + dataset), filling real numbers, making the demo GIF = maintainer steps, documented but NOT coded as pass/fail here. Do not claim "verified parity" without the run.

**Key interfaces this plan builds on (existing code):**
- `rocm_ocr.infer.infer_one(image_path, output_file, prompt="document parsing.", image_mode="gundam", ngram_window=128, host="0.0.0.0", port=10000) -> {"tokens","decode_time","text"}` — writes streamed Markdown to `output_file`.
- `rocm_ocr.infer.run_concurrent(jobs: list[(image_path, output_file)], concurrency=8, prompt=..., image_mode=..., ngram_window=..., host=..., port=..., show_progress=True) -> list[dict]`.
- `rocm_ocr.server.start_server(model_dir, ...) -> Popen|None` / `stop_server(process)`.
- `rocm_ocr.image.SUPPORTED_IMAGE_EXTS`.

---

## Part A — Docs & Community (no GPU, no dataset)

### Task A1: Public roadmap + community benchmarks

**Files:**
- Create: `ROADMAP.md`
- Create: `docs/COMMUNITY_BENCHMARKS.md`
- Modify: `README.md` (add `ROADMAP.md` + `docs/COMMUNITY_BENCHMARKS.md` links in the Community section)

**Requirements (success criteria — reviewer checks these):**

`ROADMAP.md`:
- One-line purpose + `Last updated: 2026-06-25` + link to `docs/superpowers/specs/2026-06-25-unlimited-ocr-rocm-top-tier-design.md`.
- **North star** (from spec §3): become the de-facto standard for running Baidu Unlimited-OCR on AMD Radeon/ROCm, MI300X → 16 GB consumer cards.
- **Status table:** Phase 1 Evidence Engine 🚧 In progress · Phase 2 Upstream Siege ⏳ Planned · Phase 3 Thin Integrations ⏳ Planned.
- **Phase summaries** (2–4 bullets each, from spec §5/§6/§7): Phase 1 = OmniDocBench parity (v1.5+v1.6) + credibility README + community flywheel; Phase 2 = consumer-Radeon first-class in SGLang + Baidu repo link; Phase 3 = OpenAI-compatible endpoint + one-click demo + one RAG example.
- **How to help:** link `docs/COMMUNITY_BENCHMARKS.md`, mention good-first-issues + Discussions.
- No fabricated dates/metrics. Markdown tables render on GitHub.

`docs/COMMUNITY_BENCHMARKS.md`:
- Intro: real-world results from the community on real AMD hardware; "add yours".
- Table columns: `GPU | VRAM | ROCm | OmniDocBench Overall | tok/s | VRAM peak | settings | by`.
- Seed row (real numbers only, from existing `scripts/benchmark_results.json` / README): `AMD Radeon PRO W7900 | 48 GB | 7.2 | _pending — run make eval_ | 56 | 7.3 GB | gundam, DPI 150 | @aiwork4me`.
- **How to submit:** (1) `make benchmark` for throughput/VRAM; (2) `make eval` for the OmniDocBench score (available with the v1.3 eval harness); (3) open a PR adding a row with GPU model + ROCm version. Note the consumer-Radeon coverage gap; explicitly invite RX 7900 XTX / 7800 XT / MI50 etc.

README: in the `## Community` section, add `- [Roadmap](ROADMAP.md)` and `- [Community benchmarks](docs/COMMUNITY_BENCHMARKS.md)`.

**Steps:**
- [ ] Draft `ROADMAP.md` per requirements.
- [ ] Draft `docs/COMMUNITY_BENCHMARKS.md` per requirements (seed row = real numbers only).
- [ ] Add the two links to README Community section.
- [ ] Self-check: no fabricated numbers; cross-links resolve; matches spec; cloud CTA not re-elevated.
- [ ] Commit: `docs: add public roadmap and community benchmarks page`.

---

### Task A2: OmniDocBench parity report template

**Files:**
- Create: `docs/PARITY.md`
- Modify: `README.md` (add `[Accuracy parity (OmniDocBench)](docs/PARITY.md)` to the nav line near `[Benchmarks]`)

**Requirements:**

`docs/PARITY.md`:
- Title + purpose: accuracy parity vs the NVIDIA reference, on OmniDocBench v1.5 + v1.6.
- **Headline** (explicit runtime placeholder, not a fake number): `Overall (v1.6): _populate via make eval_`.
- **OmniDocBench modules** (real names): `text_block` (Edit_dist ↓), `display_formula` (CDM ↑ / Edit_dist ↓), `table` (TEDS ↑ / TEDS-S ↑ / Edit_dist ↓), `reading_order` (Edit_dist ↓). State `Overall = ((1−TextEdit)×100 + TableTEDS + FormulaCDM)/3`.
- **AMD vs NVIDIA parity table:** columns `Metric | NVIDIA reference | AMD ROCm (this project) | Δ`. Every numeric cell = `_populate via make eval_`. Footnote: NVIDIA reference = same model weights/prompt/seed, CUDA backend.
- **Crowded-field positioning table:** rows `MinerU2.5-Pro (95.75) | GLM-OCR (95.22) | PaddleOCR-VL-1.5 (94.93) | Unlimited-OCR (~93.92, self-reported) | DeepSeek-OCR-2 (90.25) | Marker (78.44)` with source = official v1.6 leaderboard; our row = `_populate_`. Framed as positioning anchor, not a fight.
- **Reproduction recipe:** `pip install -e .[dev]` → get OmniDocBench (`huggingface-cli download opendatalab/OmniDocBench --repo-type dataset --local-dir ./OmniDocBench_data`) → start SGLang server → `make eval` (runs `scripts/eval_omnidocbench.py`, implemented in Task B2) → numbers land here. Note: clone `main` (v1.6) and `v1_5` branch to score both.
- **Methodology:** image-mode `gundam`, Unlimited-OCR native prompt, pinned weights/seed; only the GPU backend differs between the two runs.
- **Honest scope:** numbers populated by the maintainer on AMD hardware (deferred); this doc is the reproducible structure.

README: add the PARITY.md link to the nav line.

**Steps:**
- [ ] Draft `docs/PARITY.md` per requirements (real module/metric names; every numeric cell an explicit placeholder; leaderboard numbers cited as official).
- [ ] Add PARITY.md link to README nav.
- [ ] Self-check: no fabricated own-numbers; metric names correct; Overall formula correct; recipe references `make eval` / Task B2.
- [ ] Commit: `docs: add OmniDocBench parity report template`.

---

## Part B — OmniDocBench Eval Harness (code, unit-tested with mocks)

### Task B1: `rocm_ocr.omnidocbench` module + tests

**Files:**
- Create: `src/rocm_ocr/omnidocbench.py`
- Test: `tests/test_omnidocbench.py`

**Interfaces:**
- Consumes: `rocm_ocr.infer.run_concurrent`, `rocm_ocr.image.SUPPORTED_IMAGE_EXTS`, `rocm_ocr.infer.DEFAULT_HOST/DEFAULT_PORT/DEFAULT_NGRAM_WINDOW`.
- Produces: importable functions below; consumed by Task B2's CLI.

**Public API (exact signatures — implement these):**
```python
CANONICAL_OMNIDOCBENCH_PROMPT: str        # the verified whole-page prompt (constant)
DEFAULT_PREDICTION_PROMPT: str = "document parsing."   # Unlimited-OCR native

def clean_markdown(text: str) -> str: ...
def derive_prediction_filename(image_path: str) -> str: ...
def iter_page_images(omnidocbench_dir: str) -> list[str]: ...
def build_jobs(images: list[str], pred_dir: str) -> list[tuple[str, str]]: ...
def generate_predictions(jobs, *, prompt=DEFAULT_PREDICTION_PROMPT, image_mode="gundam",
                         ngram_window=128, host=DEFAULT_HOST, port=DEFAULT_PORT,
                         concurrency=8) -> list[dict]: ...
def write_eval_config(*, gt_json: str, pred_dir: str, out_path: str,
                      match_method: str = "quick_match", include_cdm: bool = True) -> str: ...
def parse_run_summary(result_dir: str, save_name: str) -> dict: ...
def run_scorer(*, omnidocbench_repo: str, config_path: str) -> subprocess.CompletedProcess: ...
def main(argv: list[str] | None = None) -> None: ...
```

**Behavior spec:**
- `clean_markdown`: strip a leading ```` ```markdown ````/```` ``` ```` fence and trailing ```` ``` ```` if present; otherwise return unchanged. (Mirrors OmniDocBench `clean_markdown`.)
- `derive_prediction_filename`: `Path(image_path).stem + ".md"`.
- `iter_page_images`: resolve `<omnidocbench_dir>/images/` (raise `FileNotFoundError` if absent), return sorted list of files whose suffix is in `SUPPORTED_IMAGE_EXTS`.
- `build_jobs`: `[(img, str(Path(pred_dir) / derive_prediction_filename(img))) for img in images]`.
- `generate_predictions`: ensure `pred_dir` exists; delegate to `rocm_ocr.infer.run_concurrent(jobs=jobs, concurrency=concurrency, prompt=prompt, image_mode=image_mode, ngram_window=ngram_window, host=host, port=port, show_progress=True)`; return its result list. (Each `infer_one` writes the `.md` via `output_file`.)
- `write_eval_config`: write a YAML with the verified `end2end_eval` structure — `ground_truth.data_path=gt_json`, `prediction.data_path=pred_dir`, `match_method`, the four metric modules; omit `CDM` from `display_formula.metric` when `include_cdm=False`. Return `out_path`.
- `parse_run_summary`: read `result/{save_name}_run_summary.json`; return `{"overall": <overall_notebook>, "text_edit_dist": ..., "formula_cdm": ..., "table_teds": ..., "table_teds_s": ..., "reading_order_edit": ...}`. Be robust: if a key is missing, set it `None`. (Validate against the demo `result/` shape; the Overall key is `overall_notebook`.)
- `run_scorer`: `subprocess.run([sys.executable, "pdf_validation.py", "--config", config_path], cwd=omnidocbench_repo, capture_output=True, text=True, check=False)`.
- `main`: argparse CLI (see Task B2 for the thin entry script). Args: `--omnidocbench-dir`, `--gt-json`, `--pred-dir` (default `./eval_predictions`), `--version` (logged only), `--omnidocbench-repo`, `--run-scorer` (flag), `--result-dir` (default `./result`), `--host`/`--port`, `--image-mode`, `--concurrency`, `--ngram-window`, `--prompt` (default native), `--no-cdm`. Flow: iter images → build_jobs → generate_predictions → (if `--run-scorer`) write_eval_config + run_scorer + parse_run_summary → print summary JSON.

**TDD steps (write failing test first, then implement, for each function):**
- [ ] `clean_markdown`: test strips ```` ```markdown\n...``` ```` ; leaves plain text unchanged.
- [ ] `derive_prediction_filename`: `"x/foo.pdf_7.jpg"` → `"foo.pdf_7.md"`.
- [ ] `iter_page_images`: tmp dir with `images/a.jpg`, `images/b.png`, `notes.txt` → returns `[a.jpg, b.png]` sorted; raises on missing `images/`.
- [ ] `build_jobs`: maps images → `(img, pred_dir/stem.md)` tuples.
- [ ] `generate_predictions`: monkeypatch `rocm_ocr.omnidocbench.run_concurrent` (or `rocm_ocr.infer.run_concurrent`) with a stub that creates the output `.md` files; assert files written + stub called with the jobs.
- [ ] `write_eval_config`: write to tmp; reload YAML; assert `ground_truth.data_path`, `prediction.data_path`, four modules present; assert CDM absent when `include_cdm=False`.
- [ ] `parse_run_summary`: write a sample `run_summary.json` (with `overall_notebook` + module keys) to tmp `result/`; assert extraction; assert `None` for missing keys.
- [ ] `run_scorer`: monkeypatch `subprocess.run`; assert invoked with `[python, pdf_validation.py, --config, config]` and `cwd=repo`.
- [ ] Run full test file: `pytest tests/test_omnidocbench.py -v` — all PASS.
- [ ] Self-review: no AMD/server required for tests; signatures match B2; constants correct.
- [ ] Commit: `feat(eval): OmniDocBench prediction + scoring harness module`.

---

### Task B2: eval CLI entry + Makefile target + docs wiring

**Files:**
- Create: `scripts/eval_omnidocbench.py`
- Modify: `Makefile` (add `eval` target)
- Modify: `docs/PARITY.md` (ensure `make eval` references resolve — already drafted in A2)
- Modify: `pyproject.toml` (add `pyyaml` is already a dep — confirm; no new deps expected)

**Interfaces:**
- Consumes: `rocm_ocr.omnidocbench.main` (from B1).
- Produces: a runnable `python scripts/eval_omnidocbench.py ...` and `make eval` (prints the canonical command; the run itself needs a live server + dataset, so the target documents/echoes the command rather than hard-failing without a GPU).

**Requirements:**
- `scripts/eval_omnidocbench.py`: shebang + `from rocm_ocr.omnidocbench import main` + `if __name__ == "__main__": main()`.
- `Makefile` `eval` target: a documented recipe that runs the prediction step and reminds the user to (a) start the SGLang server first, (b) point `--omnidocbench-dir` at the dataset, (c) `--run-scorer --omnidocbench-repo <path>` to score. Keep it honest — do not pretend it works without a GPU.
- `pyproject.toml`: confirm `pyyaml` present (it is, in `[project.dependencies]`). No changes unless a dep is missing.

**Steps:**
- [ ] Create `scripts/eval_omnidocbench.py` entry script.
- [ ] Add `eval` target to `Makefile` (documented, honest about prerequisites).
- [ ] Test the CLI parses args without a server: `python scripts/eval_omnidocbench.py --help` exits 0.
- [ ] Self-review: entry script thin; Makefile honest; PARITY.md `make eval` references consistent.
- [ ] Commit: `feat(eval): add eval CLI entry and make eval target`.

---

## Runtime (deferred to maintainer — NOT pass/fail tasks here)

Documented for when the maintainer is at an AMD box:
1. `huggingface-cli download opendatalab/OmniDocBench --repo-type dataset --local-dir ./OmniDocBench_data` (~1.55 GB).
2. `git clone https://github.com/opendatalab/OmniDocBench.git` (v1.6 = main) and `git clone -b v1_5 ...` (v1.5).
3. Start SGLang server (e.g. `unlimited-ocr --pdf <sample>` once, or `examples/sglang_server.sh`).
4. `make eval` (or `python scripts/eval_omnidocbench.py --omnidocbench-dir ./OmniDocBench_data --gt-json <full_gt>.json --omnidocbench-repo ./OmniDocBench --run-scorer`) — twice (v1.5, v1.6).
5. Paste Overall + per-module numbers into `docs/PARITY.md`; generate the README demo GIF from a real run.
6. (Optional) drop CDM (`--no-cdm`) if TeX Live/ImageMagick/Ghostscript unavailable.

## Self-review (controller runs after writing this plan)
- Spec coverage: Phase 1 Evidence Engine (spec §5) = eval harness (B1/B2) ✓, parity report (A2) ✓, README reform (done eb1a57d) ✓, community flywheel (A1 ROADMAP + COMMUNITY_BENCHMARKS) ✓. Discussions/good-first-issue *creation* are GitHub actions needing push access — deferred to maintainer, referenced in ROADMAP.
- No in-task placeholders: runtime numbers are explicit `_populate via make eval_` placeholders (legitimate, not fake).
- Type consistency: `derive_prediction_filename` / `build_jobs` / `run_concurrent` job tuple `(image_path, output_file)` consistent across B1 and existing `rocm_ocr.infer`.
