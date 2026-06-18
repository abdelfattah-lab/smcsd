# Decode variants — pipeline timelines (drain / inherit / redraw / ragged-lag)

Gantt-style timelines of the streaming full-bonus SMC decode variants, at resample-every-window (K=1), on the decoupled engine (drafter GPU1, verifier GPU0).

**Legend:** time flows left→right; `█` = that unit is busy; **bars sharing columns = overlap**; **an extra `DRAFT`/`REDRAFT` row = an extra drafter pass** (the throughput killer). Bar widths ≈ the measured costs: `DRAFT`≈13, `VERIFY`≈11, `RESAMPLE`≈2 (≈ 40 ms / 36 ms / ~2 ms). Drafter is memory-bound, so a pass costs ~40 ms regardless of how many rows it covers.

All variants commit the **exact target bonus** every window (full bonus). They differ only in (a) whether the resample is *hidden* under the next draft (drain vs cross-barrier) and (b) what a **cloned** particle does at the resample (re-draw fresh = diverse, vs copy the survivor's in-flight window = coupled), and whether re-drawing costs an extra pass.

---

## ① Mode A — *drain*, clones re-draw → 70% @ 91 tok/s
```
DRAFT t          █████████████
VERIFY t                      ███████████
RESAMPLE t                               ██
DRAFT t+1                                  █████████████
VERIFY t+1                                              ███████████
```
Resample sits in a **gap** — the drafter idles through verify+resample, then cold-drafts t+1. **1 pass/window.** Clones re-draw their own next window at the resample → **diverse**. `SMCSD_BET_DISCARD`.

## ② Inherit — *cross-barrier*, clones copy → 65% @ 92 tok/s
```
DRAFT t          █████████████
VERIFY t                      ███████████
FIRE DRAFT t+1                           █████████████
RESAMPLE t                               ██            <- hidden UNDER the draft
VERIFY t+1                                            ███████████
```
Fire t+1 *before* the resample, so **RESAMPLE hides under DRAFT t+1** (saves the gap → 92 vs 91). Still **1 pass**, but clones *copy* the survivor's in-flight draft → **coupled** (clone = parent for one window) → −5pt. `SMCSD_COPYAHEAD_RESAMPLE`.

## ③ Redraw — *cross-barrier*, clones re-draw → 73% @ 68 tok/s
```
DRAFT t          █████████████
VERIFY t                      ███████████
FIRE DRAFT t+1                           █████████████          <- survivors
RESAMPLE t                               ██
REDRAFT clones                                        █████████████   <- 2nd PASS (the cost)
VERIFY t+1                                                         ███████████
```
Resample hidden too, but the re-drawn clones can't use the survivors' draft → they need a **whole second drafter pass** (uniform-catchup). **2 passes/window** → 68. Diverse (73%); the extra row is exactly why it's slow. `SMCSD_COPYAHEAD_REDRAW`.

## ④ Ragged-lag — *cross-barrier*, clones re-draw but LAG → projected ~73% @ ~91 (NOT yet built)
```
DRAFT t          █████████████
VERIFY t                      ███████████
FIRE DRAFT t+1                           █████████████      <- survivors only; clones re-draw INSIDE
RESAMPLE t                               ██                    the NEXT pass, 1 window behind
VERIFY t+1                                            ███████████   (survivors; clones verify next round)
```
Same **1 pass** as inherit, but clones re-draw (diverse) by **folding into the next draft** instead of a separate catch-up pass — they just **lag one window** (ragged batch). No extra row → no extra pass. The only shape that gets ③'s diversity at ①'s speed — *if* the ragged bookkeeping lands.

---

## Summary
| variant | resample | draft rows / window | clones | accuracy | tok/s |
|---|---|---|---|---|---|
| ① Mode A | gap (idle) | **1** | re-draw | 70% | 91 |
| ② inherit | hidden | **1** | copy (coupled) | 65% | 92 |
| ③ redraw | hidden | **2** | re-draw | 73% | 68 |
| ④ ragged-lag *(unbuilt)* | hidden | **1** (clones lag) | re-draw | ~73%? | ~91? |

**Reading it:** hiding the resample (①→②) buys ~nothing — it's a 2-char block. The thing that actually moves throughput is the **number of draft rows**: ③'s second row is the *entire* 68-vs-91 gap. Clone behavior sets accuracy: re-draw (diverse) = ~70–73%, copy (coupled) = 65%.

**Conclusions (this 2-GPU box, 100q):**
- **Mode A is the balanced optimum.** ② inherit is strictly dominated (same speed, −5pt for coupling). ③ redraw matches Mode A's accuracy (73 vs 70 is within 100q noise) but pays a 2nd pass → slower.
- **④ ragged-lag is the only untried lever** for "re-draw diversity at 1-pass speed." If its projected `~73% @ ~91` holds, it would be a genuine improvement over Mode A; if not, Mode A stands.
- Reference: **no-bonus** is shape ① with the bonus/re-draft stripped (fastest, 135 @ 66%); the bonus is what buys the +4–7pt and costs the throughput.

Flags: `SMCSD_BET_DISCARD` (①), `SMCSD_COPYAHEAD_RESAMPLE` (②), `SMCSD_COPYAHEAD_REDRAW` (③); all with `SMCSD_DROP_BONUS=1 SMCSD_ANCHOR_TEMP=0.3 SMCSD_RESAMPLE_INTERVAL=1` on `--mode smc_async`. Related: `docs/smc/copyahead_redraw_handoff.md` (the redraw KV-corruption bug + fix), `docs/smc/async_bonus_design.md` (design background).

---

## ⑤ Lag-1 (`SMCSD_LAG1_BONUS`) — as implemented (`_run_lag1_bonus_train`)

Bounded-lag (≤1 window/group) full-bonus pipeline, K=1. GPU-marked timeline:
```
GPU1·drafter   DRAFT t        █████████████
GPU0·verifier  VERIFY t                    ███████████
GPU1·drafter   DRAFT t+1                   █████████████   ← ONE mixed pass; fired BEFORE verify
GPU0·verifier  RESAMPLE t                             ▓▓   ← AFTER verify (needs weights)
GPU0·verifier  VERIFY t+1                               ███████████
GPU1·drafter   DRAFT t+2                                █████████████
```
Per-cycle loop order (`recv → fire → verify → resample`): `_lag_receive_pending`(932) → `_lag_fire_mixed_step`(961, **before** verify) → `_lag_verify_ready`(966) → `_lag_resample`(968, after verify).

**Timeline (View A): DERIVED ✓**
- **1 drafter pass/cycle** — `_lag_fire_mixed_step` sends ONE `send_step` (`:812`) over `verify_slots + stale_slots + cold_slots` (matched run-ahead `rollback=0`; bet-miss catch-up `rollback=γ+1, truncate_kv=True`; cold). `_passes_sent += 1` once. The `[LAG1_BONUS_TIMING]` print's `PASSES/CYCLE` should read ~1.0.
- **No hidden 2nd pass** — `_lag_resample` only does the verifier KV clone + `send_commit` (`:913`, cheap drafter mirror), **no `send_step`**.
- **Fire before verify + non-blocking** → GPU1 computes DRAFT t+1 while GPU0 does verify+resample → drafter back-to-back, low `recv-wait`.

**Diversity (View B): NOT as drawn — clones INHERIT, they do not re-draw.**
- Resample clones adopt the survivor's **in-flight run-ahead tokens** via the ancestor map (`_lag_apply_resample_plan:883`, `a[dst]=src`) + copy of `_lag_ready`. So a clone's next window = the survivor's → **depth-1 coupling** (inherit/② behavior, ~65% class), NOT the re-draw/73% View B promised.
- (Bet-**misses** DO re-draw — they go `stale` and catch up off the exact bonus `b` next cycle. Only resample **clones** inherit.)
- This is the deliberate choice ("no redraw is fine"): inheriting the run-ahead is **throughput-optimal** (clones add zero extra draft-attempts — they ride the survivor's in-flight pass). Re-drawing clones would add a catch-up draft-attempt each → slower. So inherit = faster, ~65% accuracy; re-draw = slower, ~73%.

## Can Lag-1 reach the fully-async **no-bonus** speed (135)? — conceptually, NO

**Ceiling ≈ (1 − miss_rate) × no-bonus ≈ 0.84 × 135 ≈ ~116 tok/s.** Reasoning:
- **No-bonus is fast because it has ZERO misses.** Its committed anchor *is* the drafter's `x_g1`, so there's no target bonus to mismatch → every run-ahead is valid → **no re-draft, zero wasted work** → drafter 100% productive → 135.
- **The bonus *is* the misses.** Committing the exact `b` means `x_g1 != b` on ~16% of windows (per-particle, at anchor-temp 0.3). Each miss → the run-ahead is wasted + a **catch-up re-draft** → ~**1 + miss_rate ≈ 1.16 draft-attempts per committed window** vs no-bonus's 1.0. On the memory-bound drafter (pass cost independent of useful rows), the wasted/catch-up rows ride existing passes "for free," but they **cut committed-windows-per-pass by the miss fraction** → tok/s ≈ no-bonus / 1.16 ≈ 86%.
- **Inherit (not re-draw) is what *keeps* it near that ceiling:** clones cost no extra draft-attempt, so the only unavoidable overhead is the bet-miss catch-ups. (Re-drawing clones would push it further below.)
- **Realistically a bit under ~116**, deducting for: (a) the **bounded-lag cap** throttling leaders held ≤1 ahead of laggards, and (b) **per-cycle GPU0 work** — resample + the `_lag_privatize_stale_suffix` host syncs — if it exceeds the ~40 ms GPU1 pass, GPU0 becomes the bottleneck.

**Bottom line:** Lag-1 cannot be on-par with no-bonus — the gap is exactly the bonus's miss rate (you can't get the bonus's accuracy *and* no-bonus's zero-waste, because the bonus is what creates the mismatches). But **~110–116 would beat Mode A (107) and be the fastest full-bonus mode** — a real win, just not no-bonus-level. Lower `SMCSD_ANCHOR_TEMP` (fewer misses) shrinks the gap, at an accuracy cost. Verify with `SMCSD_TIMING` (recv-wait should be low, PASSES/CYCLE ~1.0) and `SMCSD_BET_STATS` (the `x_g1!=b` % is literally the throughput gap vs no-bonus).

---

## ⓪ No-bonus async — the throughput ceiling (no flag; `else: _run_decode_train`)

The reference. Commits the **drafter's own** anchor `x_g1` (no target bonus), so there is **nothing to mismatch** → no re-draft, no catch-up, no lag — ever. GPU-marked, it's the **same pipeline shape** as ⑤ Lag-1:
```
GPU1·drafter   DRAFT t        █████████████
GPU0·verifier  VERIFY t                    ███████████
GPU1·drafter   DRAFT t+1                   █████████████   ← off x_g1 = the COMMITTED anchor → ALWAYS valid
GPU0·verifier  RESAMPLE t                             ▓▓
GPU0·verifier  VERIFY t+1                               ███████████
GPU1·drafter   DRAFT t+2                                █████████████
```

**The difference from Lag-1 is NOT the timeline shape (both 1 pass/cycle, drafter back-to-back) — it's what fills each draft pass.** The drafter pass is memory-bound (~40 ms regardless of useful-row count), so the contrast is *composition*, not duration:
```
Per DRAFT pass (12 particle-rows, all ~40 ms):
  no-bonus:  ████████████   12/12 fresh run-aheads → ALL commit          → 100% productive
  Lag-1:     ██████████▒▒   ~10 fresh run-aheads + ~2 catch-ups (▒)       →  ~84% productive
                       └ the ▒ rows re-draft the bonus's ~16% misses — wasted vs no-bonus
```

So no-bonus is the ceiling for one reason: **every row of every pass is fresh forward progress.** The bonus variants spend the miss fraction (~16%) of each same-cost pass re-doing missed windows, so they commit fresh windows at ~86% of no-bonus's rate. **The gap is wasted rows *inside* the pass, not an extra pass** — which is exactly why it can't be closed without removing the bonus (i.e., removing the misses).

**Env note:** the **135 tok/s** figure is the **old 4-GPU box**; this 2-GPU box runs ~15% slower (Mode A 107→92 here), so the *live* no-bonus ceiling here is ~**115**, and Lag-1's ceiling ~0.86×115 ≈ **~99**. Re-measure no-bonus on the same box before comparing Lag-1 against it. Run it with **no mode flag**: `SMCSD_DROP_BONUS=1 --mode smc_async` (routes to `else: _run_decode_train`, `async_scheduler.py:408`).
