# tests/test_batching.py
"""BatchedInputBuilder — left-padding + mask alignment (CPU-only logic)."""

import torch

from rocm_ocr.batching import BatchedInputBuilder, PageInputs


def _fake_page(seq_len: int, n_image_tokens: int) -> PageInputs:
    """A page whose last n_image_tokens positions are image tokens."""
    ids = [100 + i for i in range(seq_len)]
    mask = [False] * (seq_len - n_image_tokens) + [True] * n_image_tokens
    return PageInputs(
        input_ids=ids,
        images_seq_mask=mask,
        patches=torch.zeros(2, 3, 640, 640),
        image_ori=torch.zeros(1, 3, 1024, 1024),
        spatial_crop=torch.tensor([[2, 2]]),
    )


def test_batch_left_pads_to_max_length():
    """Shorter sequences are left-padded; attention_mask is 0 on pad positions."""
    pages = [_fake_page(10, 4), _fake_page(7, 4)]
    out = BatchedInputBuilder.batch(pages, pad_token_id=0)
    assert out.input_ids.shape == (2, 10)
    assert out.attention_mask.shape == (2, 10)
    # Row 1 (length 7) has 3 pad tokens on the LEFT.
    assert out.attention_mask[1, 0].item() == 0
    assert out.attention_mask[1, 2].item() == 0
    assert out.attention_mask[1, 3].item() == 1
    # Row 0 (length 10) is fully attended.
    assert out.attention_mask[0].all().item() is True


def test_batch_images_list_one_per_page():
    """images is a list of (patches, image_ori) tuples, one entry per page."""
    pages = [_fake_page(10, 4), _fake_page(7, 4)]
    out = BatchedInputBuilder.batch(pages, pad_token_id=0)
    assert len(out.images) == 2
    assert out.images[0][0].shape[0] == 2  # patches count


def test_batch_images_seq_mask_left_padded_with_false():
    """Padded positions in images_seq_mask are False (not image tokens)."""
    pages = [_fake_page(10, 4), _fake_page(7, 4)]
    out = BatchedInputBuilder.batch(pages, pad_token_id=0)
    assert out.images_seq_mask.shape == (2, 10)
    # The 3 left-pad positions of row 1 are False.
    assert out.images_seq_mask[1, 0].item() is False
    # The real image tokens (last 4) remain True.
    assert out.images_seq_mask[1, -1].item() is True
