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

# Always import: applies the FusedMoE->native patch when SGLANG_MOE_NATIVE_ON_HIP=1,
# and the JIT-micro-op->native patch when SGLANG_NATIVE_JIT_ON_HIP=1. In spawn
# children these run during the run_path re-execution, so each scheduler worker
# gets the patched paths before model load / batch init.
import rocm_ocr.sglang_native_moe  # noqa: F401
import rocm_ocr.sglang_jit_native  # noqa: F401

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

    server_args = prepare_server_args(sys.argv[1:])
    os.environ[_LAUNCH_SENTINEL] = "1"
    try:
        run_server(server_args)
    finally:
        os.environ.pop(_LAUNCH_SENTINEL, None)
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
