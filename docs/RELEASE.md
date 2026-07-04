# Release & eval runbook (一版一测一存一推送)

One command on the 4-GPU AMD host takes an eval from raw predictions to a
gated, tagged GitHub Release with a committed manifest.

## Prerequisites (one-time)

- 4-GPU host, ROCm 7.2.1, torch 2.5.1+rocm6.2 (see `scripts/setup_rocm.sh`).
- OmniDocBench dataset at `./OmniDocBench_data` (images + `omnidocbench.json`).
- OmniDocBench scorer repo at `./OmniDocBench` + its py3.11 venv (CJK toolchain:
  `texlive-lang-chinese` is required — without it CDM collapses, see docs/PARITY.md).
- `gh` authed. **Rotate to a fine-grained repo-scoped PAT** (Contents + Pull
  requests write, 90-day) via `gh auth login` in a separate terminal — the token
  must never enter plaintext (scripts/commits/chat). Status: currently a classic
  PAT is in use as an **accepted-risk override** of spec §11; rotation is pending.

## Run a full eval-release

```bash
make eval-release BACKEND=pytorch DATASET=v1.6
# → eval (~4h) → manifest → gate vs last pytorch-v1.6 manifest
# → manifest PR (CI validates schema) → merge → eval/<tag> → Release with predictions.zip
```

The gate **blocks** if Overall regresses > 0.3, any module > 0.005, or looping
pages increase. To override (recorded in the manifest + Release notes):

```bash
make eval-release BACKEND=pytorch DATASET=v1.6 ALLOW_REGRESSION="--allow-regression \"<reason>\""
```

## Smoke (no tag/release; ~2 min on 4 pages)

```bash
make eval-smoke
```

## What each artifact is

- **Manifest** (`eval/results/*.yaml`) — committed, reviewable evidence trail.
- **predictions.zip** — GitHub Release asset under the `eval/<tag>` tag (not in git).
- **Tag** — `eval/<backend>-<dataset>-<shortsha>-<date>` (no PyPI). `v<semver>` → PyPI.
