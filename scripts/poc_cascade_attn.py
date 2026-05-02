"""POC: FlashInfer cascade attention for the SMC verify path.

Models one verify forward at a single transformer layer:

  G groups, each group has N particles, each particle issues Q_per = γ+1
  verify queries.  Within a group, the first L tokens of KV are *physically
  shared* across all N particles (same paged-KV block ids, refcounted in
  production).  The next S tokens are private per particle.

We compute attention two ways and compare:

  1. Reference: BatchPrefillWithPagedKVCacheWrapper over G*N particles, each
     with full block table = [shared_prefix_blocks ; private_suffix_blocks].
     This is what FlashInfer (or FA3) does today inside SGLang -- it re-reads
     the L shared pages N times per group per layer.

  2. Cascade: MultiLevelCascadeAttentionWrapper with 2 levels.
     - Level 0 (shared prefix): batch element = group, batched over G.  All
       N*Q_per queries in a group are ganged against the L shared prefix
       pages, read once.
     - Level 1 (private suffix): batch element = particle, batched over G*N.
       Each particle's Q_per queries attend over its S private suffix pages.
     The wrapper does the merge internally via online-softmax stats.

Outputs of (1) and (2) must match within fp16 numerical noise.

Usage::

    python scripts/poc_cascade_attn.py
    python scripts/poc_cascade_attn.py --prefix-len 4096 --suffix-len 256

The script also reports wall time per call to characterize the speedup.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch

from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper


@dataclass
class Shape:
    G: int            # groups
    N: int            # particles per group
    Q_per: int        # verify queries per particle (γ+1)
    L: int            # shared prefix length in tokens (per group)
    S: int            # private suffix length in tokens (per particle)
    H_q: int          # query heads
    H_kv: int         # key/value heads (GQA)
    D: int            # head dim
    page_size: int    # tokens per page
    dtype: torch.dtype = torch.float16

    @property
    def P(self) -> int:
        return self.G * self.N

    @property
    def total_q(self) -> int:
        return self.P * self.Q_per

    @property
    def pages_prefix(self) -> int:
        # shared prefix pages PER GROUP -- one set of pages, refcounted across N
        assert self.L % self.page_size == 0, "POC: keep prefix page-aligned"
        return self.L // self.page_size

    @property
    def pages_suffix(self) -> int:
        assert self.S % self.page_size == 0, "POC: keep suffix page-aligned"
        return self.S // self.page_size


def _alloc_paged_kv(shape: Shape, device: torch.device) -> Tuple[torch.Tensor, dict]:
    """Build one unified paged KV pool for all groups + particles.

    Layout NHD: (num_blocks, 2, page_size, H_kv, D).
    Pages 0..G*pages_prefix-1                : per-group shared prefix
    Pages next G*N*pages_suffix              : per-particle private suffix
    """
    pp = shape.pages_prefix
    ps = shape.pages_suffix
    n_prefix_pages = shape.G * pp
    n_suffix_pages = shape.P * ps
    total_pages = n_prefix_pages + n_suffix_pages
    cache = torch.randn(
        total_pages, 2, shape.page_size, shape.H_kv, shape.D,
        dtype=shape.dtype, device=device,
    )
    layout = {
        "prefix_block_id": lambda g: list(range(g * pp, (g + 1) * pp)),
        "suffix_block_id": lambda particle_global_idx: list(
            range(
                n_prefix_pages + particle_global_idx * ps,
                n_prefix_pages + (particle_global_idx + 1) * ps,
            )
        ),
        "n_prefix_pages": n_prefix_pages,
        "n_suffix_pages": n_suffix_pages,
    }
    return cache, layout


def _build_reference_inputs(shape: Shape, layout: dict) -> Tuple[torch.Tensor, ...]:
    """Per-particle full block tables: each particle gets prefix + own suffix."""
    indptr = [0]
    indices: List[int] = []
    last_page = []
    pp = shape.pages_prefix
    ps = shape.pages_suffix
    for g in range(shape.G):
        prefix = layout["prefix_block_id"](g)
        for n in range(shape.N):
            particle_global = g * shape.N + n
            suffix = layout["suffix_block_id"](particle_global)
            full = prefix + suffix
            indices.extend(full)
            indptr.append(len(indices))
            # POC keeps both page-aligned, so last page is always full
            last_page.append(shape.page_size)
    qo_indptr = list(range(0, shape.total_q + 1, shape.Q_per))
    return (
        torch.tensor(qo_indptr, dtype=torch.int32, device="cuda"),
        torch.tensor(indptr, dtype=torch.int32, device="cuda"),
        torch.tensor(indices, dtype=torch.int32, device="cuda"),
        torch.tensor(last_page, dtype=torch.int32, device="cuda"),
    )


def _build_cascade_inputs(shape: Shape, layout: dict) -> Tuple[List[torch.Tensor], ...]:
    """Two-level cascade plan.

    Level 0 (shared prefix): G batch elements, each containing N*Q_per
    queries against pages_prefix shared pages.

    Level 1 (private suffix): P=G*N batch elements, each containing Q_per
    queries against pages_suffix private pages.
    """
    pp = shape.pages_prefix
    ps = shape.pages_suffix

    # Level 0: per-group
    qo_indptr_lvl0 = [0]
    kv_indptr_lvl0 = [0]
    kv_indices_lvl0: List[int] = []
    kv_last_page_lvl0: List[int] = []
    queries_per_group = shape.N * shape.Q_per
    for g in range(shape.G):
        qo_indptr_lvl0.append(qo_indptr_lvl0[-1] + queries_per_group)
        kv_indices_lvl0.extend(layout["prefix_block_id"](g))
        kv_indptr_lvl0.append(len(kv_indices_lvl0))
        kv_last_page_lvl0.append(shape.page_size)

    # Level 1: per-particle
    qo_indptr_lvl1 = list(range(0, shape.total_q + 1, shape.Q_per))
    kv_indptr_lvl1 = [0]
    kv_indices_lvl1: List[int] = []
    kv_last_page_lvl1: List[int] = []
    for particle_global in range(shape.P):
        kv_indices_lvl1.extend(layout["suffix_block_id"](particle_global))
        kv_indptr_lvl1.append(len(kv_indices_lvl1))
        kv_last_page_lvl1.append(shape.page_size)

    def _t(xs, dt=torch.int32):
        return torch.tensor(xs, dtype=dt, device="cuda")

    qo_indptr_arr = [_t(qo_indptr_lvl0), _t(qo_indptr_lvl1)]
    kv_indptr_arr = [_t(kv_indptr_lvl0), _t(kv_indptr_lvl1)]
    kv_indices_arr = [_t(kv_indices_lvl0), _t(kv_indices_lvl1)]
    kv_last_page_arr = [_t(kv_last_page_lvl0), _t(kv_last_page_lvl1)]
    return qo_indptr_arr, kv_indptr_arr, kv_indices_arr, kv_last_page_arr


def make_reference(shape: Shape, plan_args):
    """Build wrapper + plan once.  Returns a closure that just calls run()."""
    qo_indptr, kv_indptr, kv_indices, kv_last_page = plan_args
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last_page,
        num_qo_heads=shape.H_q,
        num_kv_heads=shape.H_kv,
        head_dim_qk=shape.D,
        page_size=shape.page_size,
        causal=True,
        q_data_type=shape.dtype,
        kv_data_type=shape.dtype,
    )
    return wrapper


def make_cascade(shape: Shape, plan_args):
    qo_indptr_arr, kv_indptr_arr, kv_indices_arr, kv_last_page_arr = plan_args
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = MultiLevelCascadeAttentionWrapper(num_levels=2, float_workspace_buffer=workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr_arr=qo_indptr_arr,
        paged_kv_indptr_arr=kv_indptr_arr,
        paged_kv_indices_arr=kv_indices_arr,
        paged_kv_last_page_len=kv_last_page_arr,
        num_qo_heads=shape.H_q,
        num_kv_heads=shape.H_kv,
        head_dim=shape.D,
        page_size=shape.page_size,
        causal=True,
        q_data_type=shape.dtype,
        kv_data_type=shape.dtype,
    )
    return wrapper


# Back-compat one-shot helpers (build wrapper + run once)

def run_reference(q, kv_cache, shape, plan_args):
    return make_reference(shape, plan_args).run(q, kv_cache)


def run_cascade(q, kv_cache, shape, plan_args):
    return make_cascade(shape, plan_args).run(q, kv_cache)


def _bench(fn, *args, warmup=3, iters=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--particles", type=int, default=12)
    p.add_argument("--gamma", type=int, default=8, help="γ+1 = q-per-particle")
    p.add_argument("--prefix-len", type=int, default=1024)
    p.add_argument("--suffix-len", type=int, default=128)
    p.add_argument("--h-q", type=int, default=32)     # Llama-3.1-8B: 32 q heads
    p.add_argument("--h-kv", type=int, default=8)     # GQA: 8 kv heads
    p.add_argument("--d", type=int, default=128)      # head_dim
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    shape = Shape(
        G=args.groups,
        N=args.particles,
        Q_per=args.gamma + 1,
        L=args.prefix_len,
        S=args.suffix_len,
        H_q=args.h_q,
        H_kv=args.h_kv,
        D=args.d,
        page_size=args.page_size,
    )

    print(f"shape: G={shape.G} N={shape.N} Q_per={shape.Q_per} "
          f"L={shape.L} S={shape.S} H_q={shape.H_q} H_kv={shape.H_kv} D={shape.D} "
          f"page={shape.page_size}")

    q = torch.randn(shape.total_q, shape.H_q, shape.D, dtype=shape.dtype, device=device)
    kv_cache, layout = _alloc_paged_kv(shape, device)
    print(f"paged kv: {kv_cache.shape}, total pages={kv_cache.size(0)}")

    ref_plan = _build_reference_inputs(shape, layout)
    cas_plan = _build_cascade_inputs(shape, layout)

    print("\n[correctness] running reference...")
    out_ref = run_reference(q, kv_cache, shape, ref_plan)
    print(f"  ref output shape: {out_ref.shape}")

    print("[correctness] running cascade...")
    out_cas = run_cascade(q, kv_cache, shape, cas_plan)
    print(f"  cas output shape: {out_cas.shape}")

    diff = (out_ref.float() - out_cas.float()).abs()
    print(f"  max abs diff: {diff.max().item():.4e}")
    print(f"  mean abs diff: {diff.mean().item():.4e}")
    rel = diff / (out_ref.float().abs().clamp(min=1e-3))
    print(f"  max rel diff:  {rel.max().item():.4e}")

    print("\n[bench] reference (per-particle full block-table, prefix re-read N times)...")
    t_ref, _ = _bench(run_reference, q, kv_cache, shape, ref_plan, iters=args.iters)
    print(f"  ref: {t_ref*1e3:.3f} ms / call")

    print("[bench] cascade (prefix broadcast, online-softmax merge)...")
    t_cas, _ = _bench(run_cascade, q, kv_cache, shape, cas_plan, iters=args.iters)
    print(f"  cas: {t_cas*1e3:.3f} ms / call")
    print(f"  speedup: {t_ref / t_cas:.2f}x")


if __name__ == "__main__":
    main()
