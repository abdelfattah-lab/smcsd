"""Primary per-group SMC state, contiguous on GPU.

Replaces the three legacy dicts — `group_log_weights`, `group_interval_weights`,
and the per-group slot lists — with `(max_G, N)` tensors on a single object.
Hot-path ops (weight accumulation, fused collect) read/write only contiguous
tensors; the legacy dicts are kept as thin views into the same storage so
existing callers keep working.

Layout
------
    (example: max_G=4, N=8; rows 0 and 2 claimed; row 0 has 3 particles;
     row 2 has 5 particles)

              col:   0     1     2     3     4     5     6     7
                   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      row 0 lw:    │ w0  │ w1  │ w2  │  0  │  0  │  0  │  0  │  0  │
      row 0 p2s:   │ s3  │ s4  │ s5  │ -1  │ -1  │ -1  │ -1  │ -1  │
      row 0 acm:   │  T  │  T  │  T  │  F  │  F  │  F  │  F  │  F  │
                   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
              n_active[0] = 3,   row_in_use[0] = True

      row 1  (free — on the _free_rows stack)
              n_active[1] = 0,   row_in_use[1] = False

              col:   0     1     2     3     4     5     6     7
      row 2 lw:    │ w0    w1    w2    w3    w4  │  0     0     0  │
      row 2 acm:   │  T     T     T     T     T  │  F     F     F  │
              n_active[2] = 5,   row_in_use[2] = True

column ≡ particle index
-----------------------
Each particle's index inside its group (the `smc_particle_idx` assigned at
materialisation time) IS the column number in its row — stable for the entire
group lifetime.  `particle_indices[slot]` in the legacy slot-major layout
equals the stacked column, so `log_weights[row, particle_indices[slot]]`
points to that particle's cumulative weight.

`active_cell_mask` is a STATIC "col is an allocated particle slot" mask —
it flips True at `register_group` and False at `unregister_group`.  Particle
finish does NOT flip it (matching the legacy `collect_resample_jobs` which
includes finished particles in the resample candidate set and relies on the
copy-propagation of `finished_mask` through `resample_copy_slot`).

Invariants
----------
  * active_cell_mask[row, col] = True   ↔  col < n_active[row] (allocated)
  * n_active[row] = number of particles in this group (static after register)
  * row_in_use[row] = True              ↔  row is claimed by a group
  * Padded cells (col >= n_active[row]) have p2s == -1 and never participate
    in resample (the kernel masks them via active_cell_mask).

Hot-path ops
------------
  * accumulate_weights(rows, cols, diffs): one `index_put_` into the stacked
    tensors.  Called once per decode step from `process_batch_result`.
  * The fused collect kernel reads everything above via pointers and emits
    flat dst/src slot tensors in one launch.

Lifecycle ops (rare; called on group materialize/finalize only)
---------------------------------------------------------------
  * register_group(gid, slots)
  * unregister_group(gid)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


class StackedGroupState:
    """Contiguous GPU storage for every group's SMC state."""

    def __init__(
        self,
        max_groups: int,
        n_particles: int,
        device: torch.device,
    ) -> None:
        self.max_groups = max_groups
        self.N = n_particles
        self.device = device

        # ── weight tensors (max_G, N) float64 ──────────────────────────
        # Values at inactive cells do NOT matter for the kernel (it masks
        # them to -inf via active_cell_mask).  We keep them at 0 so that
        # numerical debugging stays readable.
        self.log_weights = torch.zeros(
            (max_groups, n_particles), dtype=torch.float64, device=device,
        )
        self.interval_weights = torch.zeros(
            (max_groups, n_particles), dtype=torch.float64, device=device,
        )

        # ── bookkeeping ────────────────────────────────────────────────
        self.particle_to_slot = torch.full(
            (max_groups, n_particles), -1, dtype=torch.int32, device=device,
        )
        self.active_cell_mask = torch.zeros(
            (max_groups, n_particles), dtype=torch.bool, device=device,
        )
        self.n_active = torch.zeros(max_groups, dtype=torch.int32, device=device)
        self.row_in_use = torch.zeros(max_groups, dtype=torch.bool, device=device)

        # ── free-row bookkeeping (small, CPU-side) ─────────────────────
        # Kept on CPU because it's only touched on register/unregister,
        # which are O(1) per group lifetime.
        self._free_rows: List[int] = list(range(max_groups))

        # ── group-id ↔ row mapping (CPU dicts) ─────────────────────────
        self.group_id_to_row: Dict[str, int] = {}
        self.row_to_group_id: Dict[int, str] = {}

        # ── persistent scratch for the fused collect kernel ────────────
        # Sized for the worst case: every active particle in every row is
        # dead/survivor simultaneously (max_G * N emissions).  Reused across
        # decode steps — no per-step allocation.
        flat_cap = max_groups * n_particles
        self.scratch_dst_flat = torch.empty(
            flat_cap, dtype=torch.int32, device=device,
        )
        self.scratch_src_flat = torch.empty(
            flat_cap, dtype=torch.int32, device=device,
        )
        self.scratch_row_of_job = torch.empty(
            flat_cap, dtype=torch.int32, device=device,
        )
        self.scratch_counter = torch.zeros(1, dtype=torch.int32, device=device)
        self.scratch_resample_mask = torch.zeros(
            max_groups, dtype=torch.int32, device=device,
        )

    # ── lifecycle (rare) ────────────────────────────────────────────────

    def register_group(self, group_id: str, slots: List[int]) -> int:
        """Claim a free row for a newly materialised group.

        Returns the row index.  Initialises log_weights and interval_weights
        to zero across all active particles; active cells are marked True.
        """
        if not self._free_rows:
            raise RuntimeError(
                f"StackedGroupState: no free rows (max_groups={self.max_groups})"
            )
        n = len(slots)
        if n > self.N:
            raise ValueError(
                f"StackedGroupState: group has {n} particles, "
                f"storage supports at most {self.N}"
            )
        row = self._free_rows.pop()
        self.group_id_to_row[group_id] = row
        self.row_to_group_id[row] = group_id

        slot_t = torch.as_tensor(slots, dtype=torch.int32, device=self.device)
        self.particle_to_slot[row, :n] = slot_t
        self.particle_to_slot[row, n:] = -1
        self.active_cell_mask[row, :n] = True
        self.active_cell_mask[row, n:] = False
        self.n_active[row] = n
        self.row_in_use[row] = True
        self.log_weights[row].zero_()
        self.interval_weights[row].zero_()
        return row

    def unregister_group(self, group_id: str) -> None:
        """Release a group's row back to the free stack.  Called on
        `finalize_group`."""
        row = self.group_id_to_row.pop(group_id, None)
        if row is None:
            return
        self.row_to_group_id.pop(row, None)
        self.active_cell_mask[row] = False
        self.n_active[row] = 0
        self.row_in_use[row] = False
        self.particle_to_slot[row] = -1
        self.log_weights[row].zero_()
        self.interval_weights[row].zero_()
        self._free_rows.append(row)

    # ── views for legacy dict callers ───────────────────────────────────

    def log_weights_view(self, group_id: str, n_particles: int) -> torch.Tensor:
        """Return a length-`n_particles` view into this group's row of
        `log_weights`.  Writes through to the stacked storage — this is the
        backing store for the legacy `group_log_weights[gid]` dict entry.
        """
        row = self.group_id_to_row[group_id]
        return self.log_weights[row, :n_particles]

    def interval_weights_view(self, group_id: str, n_particles: int) -> torch.Tensor:
        row = self.group_id_to_row[group_id]
        return self.interval_weights[row, :n_particles]

    def row_of(self, group_id: str) -> Optional[int]:
        return self.group_id_to_row.get(group_id)

    # ── hot-path API ────────────────────────────────────────────────────

    def accumulate_weights(
        self,
        rows: torch.Tensor,      # (B,) int64
        cols: torch.Tensor,      # (B,) int64
        diffs: torch.Tensor,     # (B,)
    ) -> None:
        """Batched weight update.  Used by `process_batch_result`."""
        d = diffs.to(dtype=torch.float64)
        self.log_weights[rows, cols] += d
        self.interval_weights[rows, cols] += d

    # ── introspection / diagnostics ─────────────────────────────────────

    def n_active_groups(self) -> int:
        return len(self.group_id_to_row)

    def active_rows(self) -> List[int]:
        return sorted(self.group_id_to_row.values())
