# Environment setup notes (B200 / CUDA 13)

Reproducing the `main` / `yahya/proposal-finetune` install on a fresh CUDA-13
box (verified on 8×B200, Ubuntu, conda env `test`, Python 3.12). Two snags
the README doesn't cover:

1. **Rust toolchain + `protoc`.** The vendored SGLang now builds a Rust gRPC
   extension (`rust/sglang-grpc`, via `setuptools-rust`). `pip install -e
   3rdparty/sglang/python` fails first with "can't find Rust compiler", then
   with "Could not find `protoc`". Fix before installing:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   source "$HOME/.cargo/env"
   sudo apt-get install -y protobuf-compiler   # provides protoc
   ```
   (The gRPC extension is unused by the offline `SMCEngine`, but the build is
   not optional in this submodule snapshot.)

2. **`kernels` version conflict.** SGLang pins `transformers==5.6.0` and an
   unpinned `kernels` (resolves to 0.15.2). transformers 5.6.0's
   `integrations/hub_kernels.py` builds `_KERNEL_MAPPING` at import with
   `LayerRepository(...)` calls that omit a version; kernels ≥0.11 makes the
   version mandatory, so any `transformers` import (e.g. `AutoModelForCausalLM`
   in `train_proposal.py`) raises `ValueError: Either a revision or a version
   must be specified.`. SGLang only uses `kernels` for `get_kernel/load_kernel`
   (the FA3 JIT path), a stable old API, so downgrade is safe:
   ```bash
   pip install "kernels==0.10.5"
   ```

After that, `import torch, transformers, sglang, smcsd` and
`from smcsd.engine import SMCEngine` all succeed (torch 2.11.0, CUDA 13,
transformers 5.6.0, kernels 0.10.5).

**Blackwell attention backend.** Use `--attention-backend triton` on B200
(sm_100); FA3 is Hopper-targeted. The collection script already defaults to
triton.
