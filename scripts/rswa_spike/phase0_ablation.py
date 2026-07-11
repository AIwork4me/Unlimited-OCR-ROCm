#!/usr/bin/env python3
"""Phase 0: PyTorch R-SWA ablation — does full attention reproduce vLLM's EOS?

Per page, runs Unlimited-OCR infer() under:
  baseline: config.sliding_window = 128   (R-SWA on  — the reference)
  ablated : config.sliding_window = 8192  (ring never evicts -> standard full
                                           causal attention == vLLM 0.20.2rc1)
Captures first-token argmax/top-5 + generated length/head via a generate() patch.
Run: /root/vllm-venv/bin/python scripts/rswa_spike/phase0_ablation.py --smoke|--full
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pages import EOS_PAGES, control_pages, resolve_image  # noqa: E402

MODEL = "/root/models/Unlimited-OCR"
OUT = Path("/root/ocr-eval/rswa_spike"); OUT.mkdir(parents=True, exist_ok=True)
PROMPT = "<image>document parsing."
MAXLEN = 4096
TOPK = 5
GENERIC = ("image contains", "solid horizontal", "empty string", "the image")
STOP = "<｜end▁of▁sentence｜>"


def load():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  use_safetensors=True, torch_dtype=torch.bfloat16)
    return m.eval().to("cuda"), tok


def capture(m):
    """Monkeypatch m.generate to force output_scores + stash prompt_len/seq/scores.
    infer() expects generate() to return a tensor; we return out.sequences."""
    cap = {}
    orig = m.generate

    def patched(*a, **kw):
        ii = kw.get("input_ids")
        cap["plen"] = int(ii.shape[1]) if ii is not None else None
        kw["return_dict_in_generate"] = True
        kw["output_scores"] = True
        out = orig(*a, **kw)
        cap["scores"] = list(out.scores) if out.scores else None
        cap["seq"] = out.sequences
        return out.sequences

    m.generate = patched
    return cap, orig


def _topk(cap, tok):
    import torch
    if not cap.get("scores"):
        return None
    probs = torch.softmax(cap["scores"][0][0].float(), dim=-1)
    t = torch.topk(probs, TOPK)
    return [{"id": int(i), "tok": tok.decode([int(i)]), "p": float(v)}
            for i, v in zip(t.indices, t.values)]


def run_one(m, tok, cap, img, sw):
    """Run infer() with config.sliding_window=sw; return len/head/first/topk."""
    m.config.sliding_window = sw            # infer() reads this into config._ring_window
    cap.clear()
    t0 = time.time()
    try:
        m.infer(tok, prompt=PROMPT, image_file=str(img), output_path=str(OUT),
                base_size=1024, image_size=640, crop_mode=True, max_length=MAXLEN,
                no_repeat_ngram_size=35, ngram_window=128, save_results=False)
    finally:
        m.config.sliding_window = 128        # restore default
    seq, plen = cap.get("seq"), cap.get("plen")
    if seq is None or plen is None:
        return {"error": "no-capture", "elapsed": time.time() - t0}
    gen = seq[0, plen:]
    txt = tok.decode(gen, skip_special_tokens=False)
    if txt.endswith(STOP):
        txt = txt[:-len(STOP)]
    ft = int(gen[0]) if len(gen) else -1
    return {"len": len(txt), "head": txt[:200],
            "first": tok.decode([ft]) if ft >= 0 else "", "first_id": ft,
            "topk": _topk(cap, tok), "elapsed": time.time() - t0}


def classify(base: dict, abl: dict) -> str:
    """Three-way per-page gate. `base` is R-SWA (expected real OCR)."""
    a_eos = abl["len"] < 50
    a_generic = abl["len"] >= 50 and any(g in abl["head"].lower() for g in GENERIC)
    if a_eos or a_generic:
        return "CAUSAL"          # ablated reproduces vLLM failure -> R-SWA is the cause
    if abl["len"] >= 200:
        return "NOT_CAUSAL"      # ablated still fine -> R-SWA not the cause
    return "PARTIAL"             # degraded but not collapsed


def smoke(m, tok, cap):
    pages = control_pages(1)
    if not pages:
        print("SMOKE FAIL: no control pages"); return 1
    img = resolve_image(pages[0])
    if img is None:
        print(f"SMOKE FAIL: no image for {pages[0]}"); return 1
    b = run_one(m, tok, cap, img, 128)
    ok = b.get("len", 0) >= 200
    print(f"SMOKE {'PASS' if ok else 'FAIL'}: control={pages[0]} baseline_len={b.get('len')} "
          f"first={b.get('first')!r} head={b.get('head','')[:80]!r}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = ap.parse_args()
    m, tok = load()
    cap, _orig = capture(m)
    if args.smoke:
        return smoke(m, tok, cap)
    raise SystemExit("--full implemented in Task 2")  # placeholder guard; replaced next task


if __name__ == "__main__":
    sys.exit(main())
