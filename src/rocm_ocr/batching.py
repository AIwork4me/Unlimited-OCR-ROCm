# src/rocm_ocr/batching.py
"""Batched crop-mode input construction for Unlimited-OCR.

The model's ``UnlimitedOCRModel.forward`` (modeling_unlimitedocr.py:449-592) already
accepts batched multimodal input: ``images`` is a list of per-sequence
``(patches, image_ori)`` tuples, ``images_spatial_crop`` a list of per-sequence
crop tensors, indexed by sequence against ``inputs_embeds[idx]`` /
``images_seq_mask[idx]``. This module builds those per-page inputs (factoring
``model.infer``'s construction, lines 825-993) and left-pads N of them into one
batched ``generate`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

IMAGE_TOKEN_ID = 128815
BOS_ID = 0


@dataclass
class PageInputs:
    """One page's pre-batch construction (mirrors model.infer lines 825-993)."""

    input_ids: list[int]
    images_seq_mask: list[bool]
    patches: torch.Tensor            # [n_local_crops, 3, image_size, image_size]
    image_ori: torch.Tensor          # [n_global_views, 3, base_size, base_size]
    spatial_crop: torch.Tensor       # [n_global_views, 2]


@dataclass
class BatchedInputs:
    """N pages left-padded into one generate() call."""

    input_ids: torch.Tensor           # [N, L_max]
    attention_mask: torch.Tensor      # [N, L_max]
    images: list[tuple[torch.Tensor, torch.Tensor]]  # N x (patches, image_ori)
    images_seq_mask: torch.Tensor     # [N, L_max]
    images_spatial_crop: list[torch.Tensor]  # N x [n_views, 2]


def build_page_inputs(
    model: Any,
    tokenizer: Any,
    image_file: str,
    *,
    prompt: str = "<image>document parsing.",
    base_size: int = 1024,
    image_size: int = 640,
) -> PageInputs:
    """Construct one page's inputs exactly as model.infer does (crop mode).

    Imports the model's own preprocessing helpers (``dynamic_preprocess``,
    ``BasicImageTransform``, ``format_messages``, ``text_encode``,
    ``load_pil_images``) so the result is byte-identical to model.infer — the
    identity gate depends on this. Raises if the model module does not expose them.
    """
    import math  # noqa: PLC0415

    # The model's remote-code helpers (same objects model.infer uses). The model is
    # a trust_remote_code PACKAGE whose modules use relative imports
    # (``from .modeling_deepseekv2 import ...``), so it CANNOT be imported as a bare
    # module. Resolve the helpers from the loaded model's own defining module
    # (modeling_unlimitedocr.py — all six helpers are defined there). This is the
    # single robust access path and guarantees byte-identical preprocessing.
    import sys  # noqa: PLC0415

    _mod = sys.modules[model.__class__.__module__]
    basic_image_transform_cls = _mod.BasicImageTransform
    dynamic_preprocess = _mod.dynamic_preprocess
    format_messages = _mod.format_messages
    load_pil_images = _mod.load_pil_images
    text_encode = _mod.text_encode

    patch_size, downsample_ratio = 16, 4
    conversation = [
        {"role": "<|User|>", "content": prompt, "images": [image_file]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    formatted = format_messages(conversations=conversation, sft_format="plain", system_prompt="")
    images = load_pil_images(conversation)
    image_transform = basic_image_transform_cls(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

    image_token = "<image>"
    text_splits = formatted.split(image_token)
    tokenized_str: list[int] = []
    images_seq_mask: list[bool] = []
    images_list: list[torch.Tensor] = []
    images_crop_list: list[torch.Tensor] = []
    images_spatial_crop: list[list[int]] = []

    # text before <image>
    sep = text_encode(tokenizer, text_splits[0], bos=False, eos=False)
    tokenized_str += sep
    images_seq_mask += [False] * len(sep)

    image = images[0]
    if image.size[0] <= image_size and image.size[1] <= image_size:
        crop_ratio = [1, 1]
        images_crop_raw: list = []
    else:
        images_crop_raw, crop_ratio = dynamic_preprocess(image)

    from PIL import ImageOps  # noqa: PLC0415

    global_view = ImageOps.pad(image, (base_size, base_size),
                               color=tuple(int(x * 255) for x in image_transform.mean))
    images_list.append(image_transform(global_view).to(torch.bfloat16))
    images_spatial_crop.append(crop_ratio)
    for crop in images_crop_raw:
        images_crop_list.append(image_transform(crop).to(torch.bfloat16))

    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
    num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
    w_crop, h_crop = crop_ratio
    tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
    tokenized_image += [IMAGE_TOKEN_ID]
    if w_crop > 1 or h_crop > 1:
        tokenized_image += ([IMAGE_TOKEN_ID] * (num_queries * w_crop) + [IMAGE_TOKEN_ID]) * (num_queries * h_crop)
    tokenized_str += tokenized_image
    images_seq_mask += [True] * len(tokenized_image)

    # text after <image>
    sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += sep
    images_seq_mask += [False] * len(sep)

    tokenized_str = [BOS_ID] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    image_ori = torch.stack(images_list, dim=0) if images_list else torch.zeros((1, 3, base_size, base_size))
    patches = (torch.stack(images_crop_list, dim=0) if images_crop_list
               else torch.zeros((1, 3, image_size, image_size)))
    return PageInputs(
        input_ids=tokenized_str,
        images_seq_mask=images_seq_mask,
        patches=patches,
        image_ori=image_ori,
        spatial_crop=torch.tensor(images_spatial_crop, dtype=torch.long) if images_spatial_crop
        else torch.zeros((1, 2), dtype=torch.long),
    )


class BatchedInputBuilder:
    """Left-pad N PageInputs into one batched generate() call."""

    @staticmethod
    def batch(pages: list[PageInputs], pad_token_id: int) -> BatchedInputs:
        n = len(pages)
        max_len = max(len(p.input_ids) for p in pages)
        input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        images_seq_mask = torch.zeros((n, max_len), dtype=torch.bool)
        for i, p in enumerate(pages):
            length = len(p.input_ids)
            # left-pad: real tokens at the right end
            input_ids[i, max_len - length:] = torch.tensor(p.input_ids, dtype=torch.long)
            attention_mask[i, max_len - length:] = 1
            images_seq_mask[i, max_len - length:] = torch.tensor(p.images_seq_mask, dtype=torch.bool)
        images = [(p.patches, p.image_ori) for p in pages]
        # Flatten each page's spatial_crop to a 1-D [w, h] tensor. The model's
        # forward (modeling_unlimitedocr.py:487,523) iterates ``zip(images,
        # images_spatial_crop)`` and indexes ``crop_shape[0], crop_shape[1]``,
        # so each per-sequence entry must be 1-D of shape (2,) — NOT (1, 2).
        # (In batch=1 via model.infer the single (1,2) tensor happens to iterate
        # to one (2,) row; for an explicit per-page list we flatten explicitly.)
        images_spatial_crop = [p.spatial_crop.view(-1) for p in pages]
        return BatchedInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )
