# HANDOFF — PyTorch eval + optimization (next dedicated session)

- **Date:** 2026-07-11
- **Author:** session that closed out the R-SWA spike
- **Purpose:** give the next session (dedicated to **PyTorch-path detailed evaluation + optimization**) everything it needs to continue without re-deriving context.
- **Repo state at handoff:** `main` @ `2e112e2` (R-SWA spike squash-merged via PR #57, CI green). This handoff is committed on `main` (local).

---

## 1. Why this session ended here — the decision that frames the next work

The R-SWA spike (merged to `main` as #57) settled the vLLM question: **R-SWA is NOT the cause** of vLLM's first-token EOS regression (direct PyTorch ablation, opus-verified). So building a newer vLLM would not fix it; the vLLM/ROCm serving backend is **deferred to the official vLLM v0.25.0+ ROCm wheel**.

**Locked decision:** the **PyTorch (`model.infer`) path is the aligned backend**. The next session's job is to push its **detailed evaluation + optimization**.

## 2. What is DONE (do not redo)

- **R-SWA spike — COMPLETE.** Verdict `R_SWA_NOT_CAUSAL`. Doc: `docs/parity/rswa-spike-verdict-2026-07-11.md`. Evidence: `docs/parity/evidence/phase0_rswa_ablation_results.json`. Scripts: `scripts/rswa_spike/`. The prior blocker doc (`docs/parity/vllm-rocm-rswa-blocker-2026-07-11.md`) has a correction note at its top — its "R-SWA = root cause" is overturned.
- **PyTorch reference run EXISTS.** Full 1,651-page OmniDocBench v1.6, Overall ≈ **91.97**, `gundam` mode, BF16, `model.infer`, on gfx1100. Manifests: `eval/results/pytorch-v1.6__4f8c5eb7ea__2026-07-03.yaml` (Overall 91.95 re-measurement) and `eval/results/pytorch-v1.6-142da29774__*.yaml` (the 91.97 committed reference). Predictions: `eval_predictions_v16_fix` (1,648 `.md`).
- **Honest docs landed on `main`.** README/README_CN/PARITY attribute 91.97 to the PyTorch backend; vLLM/ROCm marked as a numerics-blocked preview; the v0.25.0+ re-verification trigger is documented in all three. (Task the user deferred: a final README polish pass can revisit wording.)
- **OmniDocBench scorer — functional.** `/root/ocr-eval/OmniDocBench` (py3.11 venv, rebuilt the prior session): `nltk` (punkt/punkt_tab/tagger) + `texlive-lang-chinese` (CJK for CDM) + `magick→convert` symlink installed. CDM ≈ 0.957 working.

## 3. The PyTorch gap (from `docs/PARITY.md`) — known optimization threads

- Overall **91.97 vs Baidu self-report ~93.92**: ~1.95pt gap, **~entirely Text EditDist** (0.094 vs 0.042; FormulaCDM/table at parity).
- Attribution: **~47% failure tail** (looping/degenerate pages, ~3–5 pages) + **~53% "moderate tail"** (genuine output differences on ~386 pages).
- Levers already tried and **blocked** on gfx1100: SGLang (page-faults on fused-MoE kernel); global `ngram_size=5` (Overall crashes to 64.56 — do NOT re-apply); D1 looping-truncation (regressed the full eval, reverted — see `src/rocm_ocr/repetition_fix.py`).
- Likely-open threads (user to prioritize in the next session): **targeted/per-page** looping fix (not global); recognition / Text-EditDist improvement; reproducibility/manifest hygiene; a clean full 1,651-page re-run to re-confirm 91.97.

## 4. What "细致评测 + 优化" should target — **user to define in the next session**

This handoff captures state, not the next session's exact scope. The user opened the new session specifically for PyTorch eval+optimization; confirm with them which of the §3 threads (or new ones) to pursue before planning.

## 5. Environment & gotchas (CRITICAL — will bite you if ignored)

- **GPU:** gfx1100 ×4 (RDNA3 consumer card; not officially ROCm-supported), **ROCm 7.2.1**. `PYTORCH_ROCM_ARCH` includes gfx1100.
- **PyTorch venv: `/root/vllm-venv`** (python3.12, torch 2.10.0+rocm7.0, triton-rocm 3.6.0, vllm 0.20.2rc1.dev15+g321fa2d6d). **NOTE — modified this session:** `transformers` downgraded 5.13.0 → **4.57.1** (the repo pin; the model's remote code imports `is_torch_fx_available` removed in 5.x) and `addict easydict matplotlib pymupdf` installed (model remote-code deps). vLLM 0.20.2rc1 still imports fine.
- **Model:** `/root/models/Unlimited-OCR` (6.7 GB, `trust_remote_code=True`). Config has `sliding_window=128` AND `sliding_window_size=128`; `infer()` reads `sliding_window_size or sliding_window` (size wins).
- **Harness:** run `model.infer` via `/root/vllm-venv/bin/python` as a normal script (survives). **NEVER** run `vllm serve` in the foreground — the harness 144-kills it; use a background python launcher and `kill -9` the `EngineCore` child by PID, verify `rocm-smi --showmeminfo vram` → ~28 MB before restart. (See memory `harness-background-vllm`.)
- **Disk:** `/root` overlay = 2.1 TB free (builds/outputs go here). `/workspace` is a 10 GB NFS mount (repo + symlinks only — too small for venvs/builds).
- **`/root/vllm-main-venv` does NOT exist** — the Phase-1 vLLM-`main` build was cancelled (R-SWA ruled out). Don't look for it.
- **GitHub push from this env:** `git push` can CREATE a new branch but **cannot UPDATE an existing branch** (the egress proxy serves a broken/empty `git-receive-pack` ref advertisement → "cannot lock ref … reference already exists", even with `--force`). Workaround: push the commit to a throwaway NEW branch, then move the target ref via the GitHub REST API — `gh api --method PATCH repos/AIwork4me/Unlimited-OCR-ROCm/git/refs/heads/<branch> -f sha=<full-40-sha>` (`api.github.com` is NOT intercepted) — then delete the throwaway. `gh` is authed as `AIwork4me`; the proxy cert is trusted at `/usr/local/share/ca-certificates/proxy-github-docker-hf.crt`. (See memory `github-push-from-env`.)
- **`main` is branch-protected** (requires the 4 CI checks: lint-and-test 3.10/3.11/3.12 + manifest-schema). Merges need green CI (or `--admin`, which still can't bypass required-checks). CI runs `ruff check` + `ruff format --check` + `pytest tests/` on `src/` and `tests/` (NOT `scripts/`).

## 6. Key artifacts / pointers

| What | Where |
|---|---|
| R-SWA spike verdict + decision + v0.25.0+ trigger | `docs/parity/rswa-spike-verdict-2026-07-11.md` |
| Spike evidence (15 EOS pages, ablation results) | `docs/parity/evidence/phase0_rswa_ablation_results.json` |
| Spike scripts (ablation, pages, launcher, build) | `scripts/rswa_spike/` |
| Spec / Plan (the spike) | `docs/superpowers/specs/2026-07-11-vllm-main-rswa-spike-design.md`, `docs/superpowers/plans/2026-07-11-vllm-main-rswa-spike.md` |
| Overturned blocker (R-SWA root cause — DISPROVEN) | `docs/parity/vllm-rocm-rswa-blocker-2026-07-11.md` (correction at top) |
| PyTorch accuracy parity (gap analysis) | `docs/PARITY.md` |
| PyTorch eval manifests | `eval/results/pytorch-v1.6__*.yaml` |
| Scorer | `/root/ocr-eval/OmniDocBench` (py3.11 venv) |

## 7. Recommended first steps for the next session

1. Read this handoff, then `docs/parity/rswa-spike-verdict-2026-07-11.md` (the decision), then `docs/PARITY.md` (the gap).
2. **Confirm the exact goal with the user** (which §3 thread: full re-run? targeted looping fix? recognition gap? reproducibility?).
3. The PyTorch path is the aligned backend — that's what to optimize/score. **vLLM is parked** until vLLM v0.25.0+ ships an official ROCm wheel; do not spend effort there.
4. If pushing to GitHub: remember the existing-branch-update gotcha (§5) — use the API workaround.
