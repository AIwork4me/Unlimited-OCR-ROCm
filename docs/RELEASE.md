# Release & eval runbook (一版一测一存一推送)

One command on the 4-GPU AMD host takes an eval from raw predictions to a
gated, tagged GitHub Release with a committed manifest.

## Prerequisites (one-time)

- 4-GPU host, ROCm 7.2.1, torch 2.5.1+rocm6.2 (see `scripts/setup_rocm.sh`).
- OmniDocBench dataset at `./OmniDocBench_data` (images + `OmniDocBench.json`).
- OmniDocBench scorer repo at `./OmniDocBench` + its **py3.11 venv**. The scorer
  pins numpy 1.24 etc. and **cannot run in the model's py3.12 venv**. The
  orchestrator launches the scorer with this venv's interpreter via
  `SCORER_PY` (Makefile default `/workspace/OmniDocBench/.venv/bin/python`,
  CLI flag `--scorer-python`) — override it if your venv lives elsewhere.
  CJK toolchain: `texlive-lang-chinese` is required — without it CDM collapses
  (see docs/PARITY.md).
- `gh` authed. **Rotate to a fine-grained repo-scoped PAT** (Contents + Pull
  requests write, 90-day) via `gh auth login` in a separate terminal — the token
  must never enter plaintext (scripts/commits/chat). Status: currently a classic
  PAT is in use as an **accepted-risk override** of spec §11; rotation is pending.

## Run a full eval-release

```bash
make eval-release BACKEND=pytorch DATASET=v1.6
# → eval (~4h) → manifest → gate vs last pytorch-v1.6 manifest
# → manifest PR (CI validates schema) → merge → eval/<tag> → Release with predictions.zip
#   (scorer runs under SCORER_PY — the OmniDocBench py3.11 venv, not sys.executable)
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

## If a publish crashes mid-flight (crash recovery)

`publish_release` runs these steps in order: **branch → commit → push →
`gh pr create` → `_wait_ci` → `gh pr merge --squash --delete-branch` →
fetch/checkout main → reset --hard → `git tag -a` → `git push <tag>` →
`gh release create`**. A crash anywhere after the manifest PR is created leaves
the run partially done; on resume `_wait_ci` now short-circuits when the branch
is already merged/gone (Fix 2), but you may still need to finish the tail
manually. (Run 3's auto-publish crashed inside `_wait_ci` and was completed this
way.)

To complete a crashed publish manually:

```bash
# 1. Merge the manifest PR if it's still open and CI is green.
TAG="eval/<backend>-<dataset>-<shortsha>-<date>"        # from the manifest filename
BRANCH="${TAG//\//-}"                                   # eval/... → eval-...
gh pr merge "$BRANCH" --squash --delete-branch || true  # may be already merged

# 2. Get back to a clean main matching origin.
git checkout main
git fetch origin
git reset --hard origin/main

# 3. Create + push the tag (the manifest's predictions.zip must already exist
#    at predictions/<version>.zip from the eval run).
git tag -a "$TAG" -m "$TAG Overall=<overall>"
git push origin "$TAG"

# 4. Create the Release with the predictions asset.
gh release create "$TAG" "predictions/<version>.zip" \
  --title "$TAG" --notes "Eval manifest \`$TAG\`. Gate: PASS."
```

If the predictions zip is gone (e.g. a `make clean`), re-running the eval is not
required if the manifest's `predictions_ref` tag already exists on the remote —
rebuild just the zip from the still-tagged prediction files, or re-publish the
asset under the existing tag with `gh release upload "$TAG" <zip> --clobber`.

A full idempotent resume (existence checks before each step, automatic
short-circuit when the manifest is already on main, tree restore in
`try/finally`) is deferred to workstream #1.
