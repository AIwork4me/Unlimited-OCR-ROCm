# SGLang greedy output systematically diverges from HF Transformers over long generations — confirmed autoregressive bf16 accumulation (forward faithful via first-token logit match); measurable OCR-quality impact

> Draft of a GitHub issue for **sgl-project/sglang**. Review before posting.

## Checklist

- [x] Searched related issues: this is the same phenomenon as #23812, #1316, #6850, #3746, #17408 (SGLang greedy ≠ HF Transformers; unresolved). **Our contribution is a bisection that pinpoints the cause** (pure autoregressive accumulation — the forward is faithful), plus a quantified real-world accuracy impact, observed on **AMD ROCm gfx1100** with a **MoE VLM** (`baidu/Unlimited-OCR`).
- [x] Reproduced on `sglang 0.0.0.dev11416` (current dev).
- [x] Env + minimal repro below.
- [x] English.

## TL;DR

For the same model weights + same image + same prompt + greedy (`temperature=0`), SGLang's output systematically diverges from HuggingFace `transformers` over long generations, measurably degrading OCR quality. We confirmed via bisection that **the forward pass is faithful** (first-generated-token logit distributions match to cosine **1.000000**) and **SGLang is run-to-run deterministic** — so the divergence is **pure autoregressive bf16 accumulation** from kernel-path differences (paged-attention reduction order, MoE, `sgl_kernel`), not a forward/op bug and not run-to-run nondeterminism. We'd like help (a) confirming this root cause, and (b) identifying any serving flag / attention backend / kernel path that minimizes greedy divergence from HF.

## What we confirmed (bisection)

Model: `baidu/Unlimited-OCR` (BF16 MoE VLM, MHA + sliding-window attention, detection-tagged output). Same page image (`PPT_CalculusReview_page_002`), same prompt (`<image>document parsing.`), greedy, on the same host.

**1. The forward is faithful — first-token logits match.**
SGLang `/v1/chat/completions` (`logprobs=50, max_tokens=1`) vs HF `model.generate(max_new_tokens=1, output_scores=True)` (captured via a `model.generate` wrapper around `model.infer`):

| rank | SGLang token (logprob) | HF token (logprob) |
|---|---|---|
| 1 (greedy) | `<\|det\|>` (0.0) | `<\|det\|>` (0.0) ✓ |
| 2 | `<\|/det\|>` (-20.375) | `<\|/det\|>` (-21.5) ✓ |
| 3–50 | deep tail, logprob < -25 (prob ≤ ~1e-11) | different tokens, same noise floor |

Cosine of the top-50 probability vectors (HF's token set, SGLang-missing → ~0) = **1.000000** — all probability mass is on the greedy token, which both backends agree on. A forward/op bug would perturb the meaningful distribution or flip the greedy argmax; neither happens.

**2. SGLang is run-to-run deterministic.** Same page, two greedy requests → **byte-identical output** (sha256 match). So this is *not* #1316-style per-run nondeterminism; it is a *systematic* SGLang↔HF divergence.

**3. Divergence is autoregressive.** On a 30-page OCR subset (SGLang vs HF `model.infer`):
- **0/30** pages diverge at token 1; **19/30** share ≥5 words before diverging; one page is **byte-identical**; divergence correlates with output length (short page identical, long pages diverge). → ~1e-3 bf16 per-step differences accumulate and eventually flip an argmax at a low-confidence position.

## Real-world impact (quantified)

On **OmniDocBench v1.6** (standard document-parsing benchmark, official scorer), the divergence measurably degrades quality on a 30-page subset:

| metric | HF `model.infer` | SGLang |
|---|---|---|
| text EditDist ↓ | **0.020** | **0.121** |
| table TEDS ↑ | 0.982 | 0.930 |

SGLang also falls into loop trajectories that HF avoids (e.g. a URL degrades to `/ac/ac/acac/ac/…`; an n-gram-blocker retry reduces but doesn't fully recover it, because the loop is micro-varied). Long-output workloads (OCR, long-form generation) are most affected, because accumulation grows with sequence length.

## Root-cause hypothesis

Consistent with #23812 and the paged-attention literature: SGLang's paged-attention reduction order, MoE (`fused_moe` function path), and `sgl_kernel` micro-ops compute the same math as HF but with different bf16 reduction/rounding orders → tiny per-step logit differences → autoregressive accumulation → argmax flips at varied positions. (On gfx1100/RDNA3 we additionally force `MultiPlatformOp → torch-native` for `silu_and_mul`/`gelu_and_mul`/`rms_norm`/`rotary`/`topk_softmax`/`store_cache`/`clamp_position` because `sgl_kernel` miscomputes on RDNA3 — but point 1 above shows that path is itself faithful at the forward level, so the native patches are not the cause of the accumulation.)

## Ask

1. **Confirm root cause** — is this the expected paged-attention/MoE/`sgl_kernel` bf16-accumulation behavior, or is a specific op/config amplifying it beyond pure bf16 noise?
2. **Reducing divergence toward HF** — is there a recommended serving flag / attention backend / page-size / kernel path to *minimize* greedy divergence from HF `transformers`? (We currently use `--attention-backend triton --page-size 1 --disable-cuda-graph`; `torch_native` attention is available. Does `--enable-deterministic-inference` — which targets batch-invariance — also reduce HF divergence?)
3. **Roadmap** — is closer (or bit-exact) parity with HF `transformers` greedy a goal? For eval/reproducibility-critical workloads (benchmarks, RL reference rollouts, paper reproduction), SGLang↔HF greedy divergence is a practical blocker, and a documented "closest-to-HF" serving configuration would be very valuable.

## Minimal repro (bisection)

```bash
# SGLang side (first-token top-logprobs)
python -c "
import requests, base64, mimetypes
b64 = base64.b64encode(open('PAGE.png','rb').read()).decode()
r = requests.post('http://127.0.0.1:30000/v1/chat/completions', json={
  'model':'baidu/Unlimited-OCR','temperature':0.0,'max_tokens':1,
  'logprobs':True,'top_logprobs':50,
  'messages':[{'role':'user','content':[
    {'type':'text','text':'document parsing.'},
    {'type':'image_url','image_url':{'url':f'data:image/png;base64,{b64}'}}]}]}).json()
print(r['choices'][0]['logprobs']['content'][0]['top_logprobs'][:5])
"

# HF side (first-token top-logprobs)
python -c "
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True)
m = AutoModel.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
cap = {}
orig = m.generate
def wrap(*a, **k):
    k.update(output_scores=True, return_dict_in_generate=True); k.pop('max_length', None); k['max_new_tokens']=1
    o = orig(*a, **k); cap['s'] = o.scores; return o
m.generate = wrap
m.infer(tok, prompt='<image>document parsing.', image_file='PAGE.png', image_size=640, crop_mode=True, base_size=1024, no_repeat_ngram_size=35, ngram_window=128, save_results=False, temperature=0.0)
lp = F.log_softmax(cap['s'][0][0].float(), -1); val, idx = torch.topk(lp, 5)
print([(tok.decode([i]), round(v.item(),3)) for v, i in zip(val, idx)])
"
# → both print the same greedy token + same top-2; the full greedy sequences diverge after N tokens.
```

## Environment

- **GPU:** 4× AMD gfx1100 (Radeon PRO W7900-class, RDNA3), 48 GB
- **ROCm/HIP:** 7.2.1 / 6.2; **PyTorch:** 2.5.1+rocm6.2; **transformers:** 4.57.1
- **sglang:** 0.0.0.dev11416 (`sgl_kernel` unpacked, not pip-registered)
- **Model:** `baidu/Unlimited-OCR` (BF16, MoE `n_routed_experts=64 num_experts_per_tok=6`, MHA + SWA, weights rev `84757cb0`)
- **Native-HIP patches** (gfx1100/RDNA3-specific; the divergence phenomenon itself reproduces on NVIDIA per #23812): `MultiPlatformOp → torch-native` for the ops listed above + MoE function-path → native.

## Artifacts (available on request)

First-token top-50 logprob dumps (SGLang + HF), the 30-page longest-common-prefix analysis, the OmniDocBench scoring. Happy to attach.
