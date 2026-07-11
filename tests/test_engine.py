"""Engine — batched generate wrapper + postprocess (logic with mocked model)."""
from unittest.mock import MagicMock

import torch

from rocm_ocr import engine


def _fake_page(n: int):
    """A real PageInputs of input length n (so bucketing's len(page.input_ids) works)."""
    from rocm_ocr.batching import PageInputs

    return PageInputs(input_ids=list(range(n)), images_seq_mask=[False] * n,
                      patches=torch.zeros(1, 3, 640, 640), image_ori=torch.zeros(1, 3, 1024, 1024),
                      spatial_crop=torch.tensor([1, 1]))


def test_infer_batch_buckets_by_length_and_preserves_order(monkeypatch):
    """infer_batch groups pages by input length (same-length zero-pad batching only —
    Task 4 de-risk) and preserves input order."""
    lengths = {"a.png": 3, "b.png": 3, "c.png": 4}  # two length-3 pages, one length-4
    monkeypatch.setattr(engine, "build_page_inputs",
                        lambda model, tok, p, **kw: _fake_page(lengths[p]))
    seen_shapes: list[tuple[int, int]] = []
    model = MagicMock()

    def fake_generate(**kw):
        seen_shapes.append(tuple(kw["input_ids"].shape))
        n, length = kw["input_ids"].shape
        return torch.arange(n * (length + 2)).reshape(n, length + 2)  # distinct suffix per row

    model.generate.side_effect = fake_generate
    model.config = MagicMock(sliding_window_size=128, sliding_window=128)
    tok = MagicMock(pad_token_id=0, eos_token_id=1)
    tok.decode.side_effect = lambda ids, skip_special_tokens=False: ",".join(str(int(i)) for i in ids)
    out = engine.infer_batch(model, tok, ["a.png", "b.png", "c.png"], batch_size=8)
    assert len(out) == 3
    # length-3 bucket → one batch of shape (2,3); length-4 bucket → one batch of shape (1,4).
    assert (2, 3) in seen_shapes and (1, 4) in seen_shapes


def test_infer_batch_strips_eos(monkeypatch):
    """The EOS stop string is stripped from each page's output."""
    monkeypatch.setattr(engine, "build_page_inputs", lambda model, tok, p, **kw: _fake_page(3))
    model = MagicMock()
    model.generate.return_value = torch.tensor([[0, 1, 2, 5, 6]])  # prompt_len 3, suffix [5,6]
    model.config = MagicMock(sliding_window_size=128, sliding_window=128)
    tok = MagicMock(pad_token_id=0, eos_token_id=1)
    tok.decode.return_value = "hello<｜end▁of▁sentence｜>"
    out = engine.infer_batch(model, tok, ["a.png"], batch_size=1)
    assert out == ["hello"]


def test_infer_batch_async_parallel_preprocess(monkeypatch):
    """infer_batch_async builds pages in a thread pool then runs the shared bucketed
    generate; every page is preprocessed and one output per page is returned."""
    import torch

    from rocm_ocr.batching import PageInputs

    lengths = {f"p{i}.png": 3 + (i % 2) for i in range(6)}  # two length buckets
    built: list[str] = []

    def fake_build(model, tok, p, **kw):
        built.append(p)
        n = lengths[p]
        return PageInputs(input_ids=list(range(n)), images_seq_mask=[False] * n,
                          patches=torch.zeros(1, 3, 640, 640), image_ori=torch.zeros(1, 3, 1024, 1024),
                          spatial_crop=torch.tensor([1, 1]))

    monkeypatch.setattr(engine, "build_page_inputs", fake_build)

    def fake_gen(model, tok, batch, **kw):
        n, length = batch.input_ids.shape
        return torch.arange(n * (length + 2)).reshape(n, length + 2)

    monkeypatch.setattr(engine, "_generate_batch", fake_gen)
    model = MagicMock()
    model.config = MagicMock(sliding_window_size=128, sliding_window=128)
    tok = MagicMock(pad_token_id=0, eos_token_id=1)
    tok.decode.side_effect = lambda ids, skip_special_tokens=False: ",".join(str(int(i)) for i in ids)
    paths = [f"p{i}.png" for i in range(6)]
    out = engine.infer_batch_async(model, tok, paths, batch_size=8, n_workers=3)
    assert len(out) == 6                 # one output per page
    assert set(built) == set(paths)      # every page preprocessed (in parallel)
    assert all(r is not None for r in out)


def test_compile_disabled_returns_model_unchanged():
    """With enabled=False the model object is returned unchanged (no compile call)."""
    from rocm_ocr import engine

    m = MagicMock()
    out = engine.compile_for_inference(m, enabled=False)
    assert out is m


def test_compile_enabled_attempts_compile(monkeypatch):
    """With enabled=True, torch.compile is invoked on the forward."""
    import torch

    from rocm_ocr import engine

    called = {}
    real_model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(engine.torch, "compile", lambda fn, **kw: called.setdefault("compiled", fn) or fn)
    engine.compile_for_inference(real_model, enabled=True, mode="default")
    assert "compiled" in called
