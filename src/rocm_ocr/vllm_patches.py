"""Idempotent patcher applying the Unlimited-OCR integration edits to vLLM.

Applies 5 edits to an installed vLLM site-packages tree, keeping
``patches/vllm/*.py`` byte-identical to upstream vLLM main. The arch fix
(edit 4) is the one documented local divergence, applied to the *copied*
``unlimited_ocr.py`` only.

Each edit checks its anchor before applying (idempotent re-runs are no-ops)
and raises ``RuntimeError`` if an insertion anchor is missing (loud signal of
vLLM-version drift). Verified against vLLM commit 321fa2d6d (rocm721, 0.20.2rc1).

CLI: ``python -m rocm_ocr.vllm_patches <vllm_site_dir> <repo_patches_dir>``
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REGISTRY_INSERT = '    "UnlimitedOCRForCausalLM": ("unlimited_ocr", "UnlimitedOCRForCausalLM"),'
REGISTRY_FIND = '"DotsOCRForCausalLM": ("dots_ocr", "DotsOCRForCausalLM")'
REGISTRY_DONE = '"UnlimitedOCRForCausalLM": ("unlimited_ocr"'

CONFIGS_DICT_INSERT = '    "UnlimitedOCRConfig": "vllm.transformers_utils.configs.unlimited_ocr",'
CONFIGS_DICT_FIND = '"DotsOCRConfig": "vllm.transformers_utils.configs.dotsocr"'
CONFIGS_DICT_DONE = '"UnlimitedOCRConfig": "vllm.transformers_utils.configs.unlimited_ocr"'
CONFIGS_ALL_INSERT = '    "UnlimitedOCRConfig",'
CONFIGS_ALL_FIND = '    "DotsOCRConfig",'
CONFIGS_ALL_DONE = '"UnlimitedOCRConfig",'

CONFIG_REGISTRY_BLOCK = (
    '# unlimited-ocr model_type has a hyphen so it cannot be a kwarg above;\n'
    '# register it post-construction (LazyConfigDict subclasses dict).\n'
    '_CONFIG_REGISTRY["unlimited-ocr"] = "UnlimitedOCRConfig"\n\n'
)
CONFIG_REGISTRY_FIND = "_SPECULATIVE_DECODING_CONFIGS"
CONFIG_REGISTRY_DONE = '_CONFIG_REGISTRY["unlimited-ocr"]'

DEEPSEEK_PARAM_INSERT = "        max_crops: int = MAX_CROPS,\n"
DEEPSEEK_PARAM_FIND = '        strategy: Literal["v1", "v2"] = "v1",'
DEEPSEEK_PARAM_DONE = "max_crops: int = MAX_CROPS,"
DEEPSEEK_ASSIGN_INSERT = "        self.max_crops = max_crops\n"
DEEPSEEK_ASSIGN_FIND = "        self.image_size = image_size"
DEEPSEEK_ASSIGN_DONE = "self.max_crops = max_crops"
DEEPSEEK_CALL_OLD = "image, image_size=self.image_size"
DEEPSEEK_CALL_NEW = "image, image_size=self.image_size, max_num=self.max_crops"
DEEPSEEK_CALL_DONE = "max_num=self.max_crops"

ARCH_FIX_LINE = (
    '        vllm_config.model_config.hf_config.text_config.architectures = ["DeepseekV2ForCausalLM"]  # noqa: E501\n'
)
ARCH_FIX_FIND = "        super().__init__(vllm_config=vllm_config, prefix=prefix)"
ARCH_FIX_DONE = 'text_config.architectures = ["DeepseekV2ForCausalLM"]'


def _insert_after(text: str, find: str, insert: str, label: str) -> str:
    idx = text.find(find)
    if idx == -1:
        raise RuntimeError(f"{label}: anchor not found ({find!r}) — vLLM version drift?")
    end = idx + len(find)
    return text[:end] + "\n" + insert + text[end:]


def _ensure_line_before(text: str, find: str, insert: str, label: str) -> str:
    idx = text.find(find)
    if idx == -1:
        raise RuntimeError(f"{label}: anchor not found ({find!r}) — vLLM version drift?")
    return text[:idx] + insert + text[idx:]


def apply_edits(site_dir: Path, patches_dir: Path) -> list[str]:
    """Apply the 5 edits to *site_dir* (the vllm/ package dir). Idempotent.

    Returns the list of edit names applied this call (empty on a re-run).
    """
    site = Path(site_dir)
    applied: list[str] = []

    # --- Edit 1: copy 3 upstream-identical patch files + registry line ---
    # configs/ + processors/ unlimited_ocr.py are never mutated after copy, so
    # an unconditional copy is idempotent. The model unlimited_ocr.py is mutated
    # by edit 4 (arch fix); skip re-copying it when the arch fix is already in
    # place, otherwise we would clobber edit 4 and force it to re-apply.
    uo_model_dest = site / "model_executor" / "models" / "unlimited_ocr.py"
    if not uo_model_dest.exists() or ARCH_FIX_DONE not in uo_model_dest.read_text(encoding="utf-8"):
        shutil.copy2(
            patches_dir / "vllm" / "unlimited_ocr.py",
            uo_model_dest,
        )
    shutil.copy2(
        patches_dir / "vllm" / "configs" / "unlimited_ocr.py",
        site / "transformers_utils" / "configs" / "unlimited_ocr.py",
    )
    shutil.copy2(
        patches_dir / "vllm" / "processors" / "unlimited_ocr.py",
        site / "transformers_utils" / "processors" / "unlimited_ocr.py",
    )

    reg_path = site / "model_executor" / "models" / "registry.py"
    reg = reg_path.read_text(encoding="utf-8")
    if REGISTRY_DONE not in reg:
        reg = _insert_after(reg, REGISTRY_FIND, REGISTRY_INSERT, "registry")
        reg_path.write_text(reg, encoding="utf-8")
        applied.append("registry")

    # --- Edit 2a: configs/__init__.py (_CLASS_TO_MODULE + __all__) ---
    ci_path = site / "transformers_utils" / "configs" / "__init__.py"
    ci = ci_path.read_text(encoding="utf-8")
    changed = False
    if CONFIGS_DICT_DONE not in ci:
        ci = _insert_after(ci, CONFIGS_DICT_FIND, CONFIGS_DICT_INSERT, "configs_init.dict")
        changed = True
    if CONFIGS_ALL_DONE not in ci:
        ci = _insert_after(ci, CONFIGS_ALL_FIND, CONFIGS_ALL_INSERT, "configs_init.all")
        changed = True
    if changed:
        ci_path.write_text(ci, encoding="utf-8")
        applied.append("configs_init")

    # --- Edit 2b: config.py (_CONFIG_REGISTRY post-construction) ---
    cfg_path = site / "transformers_utils" / "config.py"
    cfg = cfg_path.read_text(encoding="utf-8")
    if CONFIG_REGISTRY_DONE not in cfg:
        cfg = _ensure_line_before(cfg, CONFIG_REGISTRY_FIND, CONFIG_REGISTRY_BLOCK, "config_registry")
        cfg_path.write_text(cfg, encoding="utf-8")
        applied.append("config_registry")

    # --- Edit 3: deepseek_ocr.py (max_crops param + assign + dynamic_preprocess) ---
    ds_path = site / "transformers_utils" / "processors" / "deepseek_ocr.py"
    ds = ds_path.read_text(encoding="utf-8")
    changed = False
    if DEEPSEEK_PARAM_DONE not in ds:
        ds = _insert_after(ds, DEEPSEEK_PARAM_FIND, DEEPSEEK_PARAM_INSERT, "deepseek_max_crops.param")
        changed = True
    if DEEPSEEK_ASSIGN_DONE not in ds:
        ds = _insert_after(ds, DEEPSEEK_ASSIGN_FIND, DEEPSEEK_ASSIGN_INSERT, "deepseek_max_crops.assign")
        changed = True
    if DEEPSEEK_CALL_DONE not in ds:
        if DEEPSEEK_CALL_OLD not in ds:
            raise RuntimeError("deepseek_max_crops.call: anchor not found — vLLM version drift?")
        ds = ds.replace(DEEPSEEK_CALL_OLD, DEEPSEEK_CALL_NEW, 1)
        changed = True
    if changed:
        ds_path.write_text(ds, encoding="utf-8")
        applied.append("deepseek_max_crops")

    # --- Edit 4: arch fix in the copied unlimited_ocr.py ---
    uo_path = site / "model_executor" / "models" / "unlimited_ocr.py"
    uo = uo_path.read_text(encoding="utf-8")
    if ARCH_FIX_DONE not in uo:
        uo = _insert_after(uo, ARCH_FIX_FIND, ARCH_FIX_LINE, "arch_fix")
        uo_path.write_text(uo, encoding="utf-8")
        applied.append("arch_fix")

    return applied


def main(argv: list[str] | None = None) -> int:
    if len(argv or sys.argv[1:]) != 2:
        print("usage: python -m rocm_ocr.vllm_patches <vllm_site_dir> <repo_patches_dir>", file=sys.stderr)
        return 2
    args = argv if argv is not None else sys.argv[1:]
    site_dir = Path(args[0])
    patches_dir = Path(args[1])
    if not (site_dir / "model_executor").is_dir():
        print(f"ERROR: {site_dir} does not look like a vllm/ package dir", file=sys.stderr)
        return 1
    applied = apply_edits(site_dir, patches_dir)
    if applied:
        print(f"Applied {len(applied)} edit(s): {', '.join(applied)}")
    else:
        print("All edits already present (idempotent no-op).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
