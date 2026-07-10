"""Tests for the idempotent vLLM patch applier (against a fake site-packages tree)."""
from __future__ import annotations

from pathlib import Path

from rocm_ocr.vllm_patches import apply_edits

# Minimal stubs mirroring the fresh 321fa2d6d vLLM file anchors.
REGISTRY_STUB = (
    '_ARCH_MODELS = {\n'
    '    "DotsOCRForCausalLM": ("dots_ocr", "DotsOCRForCausalLM"),\n'
    '    "OtherForCausalLM": ("other", "OtherForCausalLM"),\n'
    '}\n'
)
CONFIGS_INIT_STUB = (
    '_CLASS_TO_MODULE: dict[str, str] = {\n'
    '    "DotsOCRConfig": "vllm.transformers_utils.configs.dotsocr",\n'
    '}\n'
    '__all__ = [\n'
    '    "DotsOCRConfig",\n'
    ']\n'
)
CONFIG_STUB = (
    '_CONFIG_REGISTRY = LazyConfigDict({\n'
    '    dotsocr="DotsOCRConfig",\n'
    '})\n'
    '_SPECULATIVE_DECODING_CONFIGS: set[str] = {"eagle"}\n'
)
DEEPSEEK_STUB = (
    'MAX_CROPS = 32\n'
    'class DeepseekOCRProcessor:\n'
    '    def __init__(\n'
    '        self,\n'
    '        image_size: int = 1024,\n'
    '        strategy: Literal["v1", "v2"] = "v1",\n'
    '        **kwargs,\n'
    '    ):\n'
    '        self.image_size = image_size\n'
    '    def tokenize_with_images(self, image):\n'
    '        x = dynamic_preprocess(\n'
    '            image, image_size=self.image_size\n'
    '        )\n'
)
UNLIMITED_OCR_STUB = (
    'class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):\n'
    '    def __init__(self, *, vllm_config, prefix: str = ""):\n'
    '        super().__init__(vllm_config=vllm_config, prefix=prefix)\n'
)


def _make_fake_tree(tmp: Path) -> tuple[Path, Path]:
    site = tmp / "vllm"
    (site / "model_executor" / "models").mkdir(parents=True)
    (site / "transformers_utils" / "configs").mkdir(parents=True)
    (site / "transformers_utils" / "processors").mkdir(parents=True)
    (site / "model_executor" / "models" / "registry.py").write_text(REGISTRY_STUB)
    (site / "transformers_utils" / "configs" / "__init__.py").write_text(CONFIGS_INIT_STUB)
    (site / "transformers_utils" / "config.py").write_text(CONFIG_STUB)
    (site / "transformers_utils" / "processors" / "deepseek_ocr.py").write_text(DEEPSEEK_STUB)
    patches = tmp / "patches"
    (patches / "vllm").mkdir(parents=True)
    (patches / "vllm" / "unlimited_ocr.py").write_text(UNLIMITED_OCR_STUB)
    (patches / "vllm" / "configs").mkdir(parents=True)
    (patches / "vllm" / "configs" / "unlimited_ocr.py").write_text("# config\n")
    (patches / "vllm" / "processors").mkdir(parents=True)
    (patches / "vllm" / "processors" / "unlimited_ocr.py").write_text("# proc\n")
    return site, patches


def test_apply_edits_applies_all_five(tmp_path: Path) -> None:
    site, patches = _make_fake_tree(tmp_path)
    applied = apply_edits(site, patches)
    assert set(applied) == {"registry", "configs_init", "config_registry", "deepseek_max_crops", "arch_fix"}
    reg = (site / "model_executor" / "models" / "registry.py").read_text()
    assert '"UnlimitedOCRForCausalLM": ("unlimited_ocr", "UnlimitedOCRForCausalLM")' in reg
    ci = (site / "transformers_utils" / "configs" / "__init__.py").read_text()
    assert '"UnlimitedOCRConfig": "vllm.transformers_utils.configs.unlimited_ocr"' in ci
    assert '"UnlimitedOCRConfig",' in ci
    cfg = (site / "transformers_utils" / "config.py").read_text()
    assert '_CONFIG_REGISTRY["unlimited-ocr"] = "UnlimitedOCRConfig"' in cfg
    ds = (site / "transformers_utils" / "processors" / "deepseek_ocr.py").read_text()
    assert "max_crops: int = MAX_CROPS," in ds
    assert "self.max_crops = max_crops" in ds
    assert "max_num=self.max_crops" in ds
    uo = (site / "model_executor" / "models" / "unlimited_ocr.py").read_text()
    assert 'text_config.architectures = ["DeepseekV2ForCausalLM"]' in uo
    # The arch fix MUST precede super().__init__: super() recursively loads the
    # text backbone (init_vllm_registered_model reads text_config.architectures);
    # setting it to DeepseekV2ForCausalLM before super() avoids a recursive
    # DeepseekOCR load that hits a vision_config AttributeError on DeepseekVLV2TextConfig.
    arch_idx = uo.index('text_config.architectures = ["DeepseekV2ForCausalLM"]')
    super_idx = uo.index("super().__init__(vllm_config=vllm_config, prefix=prefix)")
    assert arch_idx < super_idx


def test_apply_edits_repositions_wrongly_placed_arch_fix(tmp_path: Path) -> None:
    # Regression: arch fix present but AFTER super().__init__ (the bug that
    # crashed the server) -> patcher must re-copy upstream + re-insert BEFORE super.
    site, patches = _make_fake_tree(tmp_path)
    uo_path = site / "model_executor" / "models" / "unlimited_ocr.py"
    uo_path.write_text(
        'class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):\n'
        '    def __init__(self, *, vllm_config, prefix: str = ""):\n'
        '        super().__init__(vllm_config=vllm_config, prefix=prefix)\n'
        '        vllm_config.model_config.hf_config.text_config.architectures = ["DeepseekV2ForCausalLM"]  # noqa: E501\n'
    )
    applied = apply_edits(site, patches)
    assert "arch_fix" in applied  # re-applied because placement was wrong
    uo = uo_path.read_text()
    arch_idx = uo.index('text_config.architectures = ["DeepseekV2ForCausalLM"]')
    super_idx = uo.index("super().__init__(vllm_config=vllm_config, prefix=prefix)")
    assert arch_idx < super_idx


def test_apply_edits_is_idempotent(tmp_path: Path) -> None:
    site, patches = _make_fake_tree(tmp_path)
    apply_edits(site, patches)
    second = apply_edits(site, patches)
    assert second == []  # nothing re-applied


def test_apply_edits_copies_patch_files(tmp_path: Path) -> None:
    site, patches = _make_fake_tree(tmp_path)
    apply_edits(site, patches)
    assert (site / "model_executor" / "models" / "unlimited_ocr.py").is_file()
    assert (site / "transformers_utils" / "configs" / "unlimited_ocr.py").is_file()
    assert (site / "transformers_utils" / "processors" / "unlimited_ocr.py").is_file()


def test_apply_edits_raises_on_missing_anchor(tmp_path: Path) -> None:
    site, patches = _make_fake_tree(tmp_path)
    # Corrupt the registry anchor so the DotsOCR line is gone.
    (site / "model_executor" / "models" / "registry.py").write_text("_ARCH_MODELS = {}\n")
    import pytest
    with pytest.raises(RuntimeError, match="registry"):
        apply_edits(site, patches)
