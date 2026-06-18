# Decoupled-SMC full-bonus throughput — STATUS (2026-06-18)

Branch `smc-async-draft` @ `3a04f2b70` (pushed to `abdelfattah-lab/smcsd`). All work below is **flag-gated**; flag-off is byte-identical to the no-bonus baseline.

## TL;DR
- **Accuracy goal: MET.** Committing the exact target bonus `b` ("full bonus") buys **+4–7pt** over no-bonus (66% → ~71–73%), validated end-to-end.
- **Throughput goal (beat Mode A at full accuracy): NOT met, and proven structural.** Every full-bonus *scheduling* variant we built (async-bonus, depth-2, copy-ahead, redraw, Lag-1) **ties Mode A** (~92–107 depending on box). **Throughput is work-bound, not schedule-bound — you cannot out-schedule a fixed workload.**
- **The one unexploited lever is the *work*, not the pipeline:** reduce the bet-miss rate via **coupled-bonus** (maximal coupling). Empirically worth ~+14% but **unbuilt** (needs an unbiased residual) and likely only *matches* the ~107 Mode-A ceiling.

## Measured frontier (full-bonus = ~71–73% unless noted)
| config | accuracy | tok/s | note |
|---|---|---|---|
| no-bonus async | 66% | ~135 (4-GPU) / ~115 (2-GPU) | throughput **ceiling** (no bonus → no misses → zero waste) |
| SBP (deep free-run) | 62% | 177 | fast, −4.4pt diversity coupling |
| **Mode A** (`SMCSD_BET_DISCARD`) | ~71% | ~107 (4-GPU) / ~92 (2-GPU) | **full-bonus throughput optimum** |
| lockstep+bonus (`--mode smc_decoupled`) | 73.5% | ~107 | max accuracy (K=1, synchronous) |
| copy-ahead inherit (`SMCSD_COPYAHEAD_RESAMPLE`) | ~65% | ~92 | clones inherit run-ahead (depth-1 coupling) — dominated |
| copy-ahead redraw (`SMCSD_COPYAHEAD_REDRAW`) | ~73% | ~68 | clones re-draw (2nd pass) — slower |
| **Lag-1** (`SMCSD_LAG1_BONUS`) | ~63–73% | ~93.8 | bounded-lag pipeline; **== Mode A** on identical box/config |
| async-bonus depth-1/2 (`SMCSD_ASYNC_BONUS[_DEPTH2]`) | ~71% | ~92–93 | dominated; K-sweep closed |

(`SMCSD_BONUS_WINDOWS` = a bonus-coverage knob generalizing barrier-bonus.)

## Why no scheduling variant beats Mode A (profiler-grounded, 2026-06-18)
On identical box/config: **Lag-1 93.8 == Mode A K=1 91.5 == Mode A K=2 94.0** (all 63.3% acc).
- **Same per-window work** for every variant: draft AR ~47 ms (GPU1, memory-bound, CUDA-graphed ~89% GEMM, no fusion headroom) + verify ~36 ms (GPU0, dense target GEMM) + bet-miss re-draft (~16%).
- **Same cross-GPU dependency**: the next *correct* draft needs the verify's bonus `b`, so the drafter waits on the verify regardless of choreography.
- Lag-1 actually *schedules better* (drafter busy ~62% vs ~52%, recv-wait ~30% vs ~58%) **but** its per-cycle coordination stalls (~27% both-GPU-idle) exactly cancel the overlap → **same tok/s**.
- **drafter-busy% is at a structural optimum** (verify GPU0 > draft-window FLOPs; target ~2× draft FLOPs + lag-1 dependency). A 5-round profile→optimize→benchmark loop shipped **zero** changes; only lag-2 fills the verify-tail and lag-2 is already known insufficient. **Do not re-chase drafter busy% or re-build pipeline variants.**

## Next step (the only proven tok/s lever)
**Coupled-bonus** — cut the bet-miss/stale waste (= cut *work*) via target-marginal-preserving maximal coupling. A biased prototype hit 93.8 → **107** (+14%) but crashed accuracy. The correct, **non-architectural** fix: emit the per-row anchor draft distribution on the wire so `worker.finish_decode` builds the true residual `(p−q)₊/‖(p−q)₊‖`. **Caveats:** ~107 == the Mode-A ceiling (likely *matches*, not beats), and the +14% rests on one biased run (unproven). Orthogonal lever: cut the drafter floor itself (GRAPH_AR / smaller draft model / fewer AR steps) — lifts every config.

## Pointers
- `docs/smc/decode_variant_timelines.md` — Gantt + trace-backed overlap timelines for all variants (drain/inherit/redraw/lag-1/no-bonus).
- `docs/smc/copyahead_redraw_handoff.md` — the redraw shared-KV corruption bug + the private-cell fix.
- `docs/smc/async_bonus_design.md` — async-bonus design, slice plan, K-sweep, the §5b/§8 outcome.
- Validation: `scripts/accuracy_test_gsm8k.py` with `.venv/bin/python` (GSM8K accuracy + tok/s); validate every SMC decode change against it.
