# scripts/sglang_serve_native.py
"""SGLang server entry that applies the native-MoE override before launch.

Mirrors sglang/launch_server.py __main__ (prepare_server_args -> run_server)
but imports rocm_ocr.sglang_native_moe first so the FusedMoE->native patch is
in place before model load. Invoke:
  python scripts/sglang_serve_native.py <sglang args>

Spawn note: SGLang forces the `spawn` multiprocessing start method
(engine.py: `mp.set_start_method("spawn", force=True)`). When this file is run
as a plain script (`__main__.__spec__ is None`), each spawn child re-executes
the *whole* file via `runpy.run_path(...)` to recreate the parent's `__main__`
module. `run_path` runs the code with `__name__ == "__main__"`, so a plain
`if __name__ == "__main__"` guard does NOT stop that re-execution from calling
`run_server` again inside the child, which would re-enter launch_server and
trip `_check_not_importing_main()` (recursive spawn during bootstrap). We block
that re-entry with the `_SGLANG_SERVE_LAUNCHING` env sentinel: only the true
parent process has it set at the moment of the real launch. The override import
still runs in every child (so the monkeypatch is present in each scheduler
worker) — only the *launch* is suppressed in re-runs.
"""

import os
import sys

# ORDERING INVARIANT: these three rocm_ocr override imports MUST run before any
# sglang model/layer module is imported (i.e. before launch_server -> model load).
# They patch CLASS-level forward_hip/forward_cuda dispatch on MultiPlatformOp
# subclasses (FusedMoE, RMSNorm, SiluAndMul/GeluAndMul, RotaryEmbedding, TopK).
# SGLang's MultiPlatformOp.__init__ binds `self._forward_method = self.dispatch_forward()`
# (forward_hip on HIP) at INSTANCE-CREATION time, then `MultiPlatformOp.forward` calls
# that bound `_forward_method` directly -- NOT the class `forward_hip`. So patching the
# class fixes instances created AFTER the patch but NOT before. Patching instances
# themselves is not done (call-time dispatch in sglang_native_moe covers the MoE case;
# the others are fine because every model instance is created during/after load). Keep
# these imports at the top of this file (they are), ahead of `from sglang...` below.
import rocm_ocr.sglang_conv_template  # noqa: F401  (unlimited-ocr SFT template fix)
import rocm_ocr.sglang_jit_native  # noqa: F401

# Always import: applies the FusedMoE->native patch when SGLANG_MOE_NATIVE_ON_HIP=1,
# and the JIT-micro-op->native patch when SGLANG_NATIVE_JIT_ON_HIP=1. In spawn
# children these run during the run_path re-execution, so each scheduler worker
# gets the patched paths before model load / batch init.
import rocm_ocr.sglang_native_moe  # noqa: F401

# Optional debug: trace the multimodal image flow (where the image is lost).
# Imported only when SGLANG_MM_DEBUG=1; no-op otherwise.
if os.environ.get("SGLANG_MM_DEBUG", "0") == "1":
    import rocm_ocr.sglang_mm_debug  # noqa: F401

_LAUNCH_SENTINEL = "_SGLANG_SERVE_LAUNCHING"


def main() -> None:
    # Re-execution guard: a spawn child re-running this file via runpy.run_path
    # inherits the parent's env, including this sentinel. Only the true parent
    # path below clears-then-sets it around the launch so the re-run bails out.
    if os.environ.get(_LAUNCH_SENTINEL) == "1":
        return

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    # Cheap invariant check: if the JIT-native gate was set, the override MUST have
    # applied before we reach model load. A False here means the import-time
    # auto-apply was silently skipped (e.g. sglang layout changed) -- fail serve
    # loudly rather than serve garbage OCR. (See ordering-invariant comment above.)
    if os.environ.get("SGLANG_NATIVE_JIT_ON_HIP", "0") == "1":
        from rocm_ocr.sglang_jit_native import _APPLIED as _JIT_APPLIED

        assert _JIT_APPLIED, (
            "SGLANG_NATIVE_JIT_ON_HIP=1 but rocm_ocr.sglang_jit_native did not "
            "apply its patches at import time (sglang layout changed, or sglang is "
            "absent). Refusing to launch: silent OCR corruption would result."
        )

    server_args = prepare_server_args(sys.argv[1:])
    os.environ[_LAUNCH_SENTINEL] = "1"
    try:
        run_server(server_args)
    finally:
        os.environ.pop(_LAUNCH_SENTINEL, None)
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
