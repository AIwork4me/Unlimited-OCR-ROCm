"""DEBUG instrumentation for the Unlimited-OCR multimodal image flow.

Imported by scripts/sglang_serve_native.py ONLY when SGLANG_MM_DEBUG=1. Prints
where the image is lost so the StopIteration in legacy_load_mm_data can be
localized (upstream content parsing vs downstream task building).

NOT a fix -- diagnostic only. Removed/ignored in normal operation.
"""

from __future__ import annotations

import sglang.srt.managers.template_manager as tm_mod
import sglang.srt.multimodal.processors.base_processor as bp
import sglang.srt.multimodal.processors.unlimited_ocr as uop


def _log(msg: str) -> None:
    print(f"[MM_DEBUG] {msg}", flush=True)


# 1) content_format (does SGLang treat Unlimited-OCR content as 'string' or 'openai'?)
_orig_fmt_get = tm_mod.TemplateManager.jinja_template_content_format.fget


def _fmt_get(self):
    v = _orig_fmt_get(self)
    _log(f"jinja_template_content_format = {v!r}")
    return v


tm_mod.TemplateManager.jinja_template_content_format = property(_fmt_get)

# 2) image_data length at the Unlimited-OCR processor entry (is the image reaching it?)
_orig_proc = uop.UnlimitedOCRProcessor.process_mm_data_async


async def _proc(self, image_data, input_text, *args, **kwargs):
    n = len(image_data) if image_data else 0
    has_tok = "<image>" in (input_text or "")
    _log(f"process_mm_data_async ENTER: len(image_data)={n} input_text_has_<image>={has_tok} input_text={input_text!r}")
    try:
        return await _orig_proc(self, image_data, input_text, *args, **kwargs)
    except Exception as e:
        import traceback

        _log(f"process_mm_data_async RAISED {type(e).__name__}: {e}\n" + traceback.format_exc()[:1800])
        raise


uop.UnlimitedOCRProcessor.process_mm_data_async = _proc

# 3) submit_data_loading_tasks output (are image tasks being built?)
_orig_submit = bp.BaseMultimodalProcessor.submit_data_loading_tasks


def _submit(self, *args, **kwargs):
    futures, task_info = _orig_submit(self, *args, **kwargs)
    _log(f"submit_data_loading_tasks -> len(task_info)={len(task_info)} len(futures)={len(futures)}")
    return futures, task_info


bp.BaseMultimodalProcessor.submit_data_loading_tasks = _submit

# 4) The actual prompt + <image> count reaching legacy_load_mm_data (does <image>
#    get expanded/duplicated into many tokens while there's only 1 image?).
_orig_legacy = bp.BaseMultimodalProcessor.legacy_load_mm_data


def _legacy(self, *args, **kwargs):
    prompt = kwargs.get("prompt") if "prompt" in kwargs else (args[0] if args else None)
    if isinstance(prompt, str):
        _log(
            f"legacy_load_mm_data: prompt_len={len(prompt)} <image>_count={prompt.count('<image>')} "
            f"prompt_head={prompt[:140]!r}"
        )
    return _orig_legacy(self, *args, **kwargs)


bp.BaseMultimodalProcessor.legacy_load_mm_data = _legacy

# 5) Catch the empty-message ValueError raised downstream of the processor
#    (generate_request -> _tokenize_one_request), so its real source is logged.
import sglang.srt.managers.tokenizer_manager as tmgr  # noqa: E402

_orig_tok = tmgr.TokenizerManager._tokenize_one_request


async def _tok(self, *args, **kwargs):
    try:
        result = await _orig_tok(self, *args, **kwargs)
    except Exception as e:
        import traceback

        _log(f"_tokenize_one_request RAISED {type(e).__name__}: {e!r}\n" + traceback.format_exc()[:2000])
        raise
    # Phase-1 evidence: the EXACT input_ids the model will see (decoded), so we can
    # tell a malformed prompt (root cause = prompt) from a good prompt + bad compute.
    ids = getattr(result, "input_ids", None)
    tok = getattr(self, "tokenizer", None)
    if ids is not None and tok is not None and hasattr(tok, "decode"):
        try:
            n = len(ids)
            import re as _re

            full = tok.decode(ids)
            # collapse runs of <image> so the structure (markers, text) is readable
            collapsed = _re.sub(r"(<image>)+", lambda m: f"<image>x{len(m.group()) // 7}", full)
            _log(f"FINAL input_ids: n={n} collapsed={collapsed!r}")
        except Exception as de:  # noqa: BLE001
            _log(f"FINAL input_ids decode failed: {de!r}")
    return result


tmgr.TokenizerManager._tokenize_one_request = _tok

# 6) Dump the image embeddings (projector output) the LLM receives for <image>
#    tokens -- NaN/anomaly check (does a vision/projector kernel blow up?).
import sglang.srt.models.unlimited_ocr as uom  # noqa: E402

_orig_embed = uom.UnlimitedOCRForCausalLM._pixel_values_to_embedding


def _embed(self, *args, **kwargs):
    out = _orig_embed(self, *args, **kwargs)
    try:
        import torch as _t

        ts = out if isinstance(out, (list, tuple)) else [out]
        for i, t in enumerate(ts):
            if isinstance(t, _t.Tensor):
                tf = t.float()
                _log(
                    f"image_embed[{i}]: shape={tuple(t.shape)} dtype={t.dtype} "
                    f"mean={tf.mean().item():.4f} std={tf.std().item():.4f} "
                    f"min={t.min().item():.4f} max={t.max().item():.4f} "
                    f"has_nan={_t.isnan(t).any().item()} has_inf={_t.isinf(t).any().item()}"
                )
    except Exception as e:  # noqa: BLE001
        _log(f"image_embed dump failed: {e!r}")
    return out


uom.UnlimitedOCRForCausalLM._pixel_values_to_embedding = _embed

_log("instrumentation installed")
