"""Sweep prefix length, N, and batch (groups) to map cascade speedup.

Reuses the helpers in poc_cascade_attn.py.

  python scripts/poc_cascade_attn_sweep.py
"""

from __future__ import annotations

import torch

from poc_cascade_attn import (
    Shape, _alloc_paged_kv, _build_reference_inputs, _build_cascade_inputs,
    make_reference, make_cascade, _bench,
)


def _row(prefix, suffix, G, N, Q_per, page, h_q, h_kv, d, iters=15):
    shape = Shape(
        G=G, N=N, Q_per=Q_per,
        L=prefix, S=suffix,
        H_q=h_q, H_kv=h_kv, D=d,
        page_size=page,
    )
    q = torch.randn(shape.total_q, shape.H_q, shape.D, dtype=shape.dtype, device="cuda")
    kv_cache, layout = _alloc_paged_kv(shape, torch.device("cuda"))
    ref_plan = _build_reference_inputs(shape, layout)
    cas_plan = _build_cascade_inputs(shape, layout)

    # Plan once -- mirrors how SGLang amortizes plan() per cycle.
    ref_w = make_reference(shape, ref_plan)
    cas_w = make_cascade(shape, cas_plan)

    out_ref = ref_w.run(q, kv_cache)
    out_cas = cas_w.run(q, kv_cache)
    max_diff = (out_ref.float() - out_cas.float()).abs().max().item()

    t_ref, _ = _bench(ref_w.run, q, kv_cache, iters=iters)
    t_cas, _ = _bench(cas_w.run, q, kv_cache, iters=iters)

    # Free workspace memory between rows
    del ref_w, cas_w, kv_cache, q
    torch.cuda.empty_cache()

    return t_ref * 1e3, t_cas * 1e3, t_ref / t_cas, max_diff


def main():
    # Llama-3.1-8B-ish: 32 q heads, 8 kv heads (GQA), head_dim 128
    h_q, h_kv, d = 32, 8, 128
    Q_per = 9  # γ=8
    page = 16
    suffix = 128

    print(f"\n{'='*100}")
    print(f"  SMC verify shape sweep (γ+1={Q_per}, suffix={suffix}, page={page}, "
          f"GQA {h_q}/{h_kv} × {d})")
    print(f"  ref  = BatchPrefillWithPagedKVCacheWrapper, full block table per particle")
    print(f"  cas  = MultiLevelCascadeAttentionWrapper, 2 levels (prefix shared, suffix private)")
    print(f"{'='*100}")

    for G in [1, 4]:
        for N in [4, 8, 12]:
            print(f"\n  G={G} groups × N={N} particles  ({G*N} total particles, "
                  f"{G*N*Q_per} verify queries)")
            print(f"  {'L (prefix)':>12} | {'ref ms':>9} | {'cas ms':>9} | "
                  f"{'speedup':>8} | {'max-diff':>9}")
            print(f"  {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}")
            for L in [512, 1024, 2048, 4096, 8192, 16384]:
                t_ref, t_cas, sp, dd = _row(
                    L, suffix, G, N, Q_per, page, h_q, h_kv, d,
                )
                marker = "🚀" if sp >= 1.05 else ("≈" if sp >= 0.95 else "  ")
                print(f"  {L:>12} | {t_ref:>9.3f} | {t_cas:>9.3f} | "
                      f"{sp:>7.2f}x | {dd:>9.2e} {marker}")


if __name__ == "__main__":
    main()
