# DFlash vs EAGLE Gamma Sweep Report

Date: 2026-05-13

## Setup

- Target model: `meta-llama/Llama-3.1-8B-Instruct`
- DFlash draft: `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`
- EAGLE draft: `lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge`
- Mode: `smc_engine`
- Particles: `12`
- Temperature: `0.7`
- Attention backend: `fa3`
- Questions: first `50` GSM8K test examples
- Command shape:
  - DFlash: `python scripts/accuracy_test_gsm8k.py --mode smc_engine --model meta-llama/Llama-3.1-8B-Instruct --draft-model z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat --smc-draft-mode dflash --particles 12 --gamma <g> --temperature 0.7 --attention-backend fa3 --num-questions 50`
  - EAGLE: `python scripts/accuracy_test_gsm8k.py --mode smc_engine --model meta-llama/Llama-3.1-8B-Instruct --draft-model lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge --smc-draft-mode eagle3 --particles 12 --gamma <g> --temperature 0.7 --attention-backend fa3 --num-questions 50`

Note: in this code path, DFlash block size is `gamma + 1`. The DFlash paper's LLaMA block size `10` corresponds to `--gamma 9`.

## Results

| Gamma | DFlash Accuracy | DFlash Invalid | DFlash Tok/s | EAGLE Accuracy | EAGLE Invalid | EAGLE Tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 43/50 (86.0%) | 0/50 (0.0%) | 175.1 | 37/50 (74.0%) | 1/50 (2.0%) | 158.0 |
| 3 | 42/50 (84.0%) | 0/50 (0.0%) | 206.1 | 34/50 (68.0%) | 0/50 (0.0%) | 209.1 |
| 4 | 37/50 (74.0%) | 0/50 (0.0%) | 230.8 | 34/50 (68.0%) | 1/50 (2.0%) | 232.1 |
| 5 | 37/50 (74.0%) | 0/50 (0.0%) | 243.8 | 33/50 (66.0%) | 3/50 (6.0%) | 263.6 |
| 6 | 38/50 (76.0%) | 0/50 (0.0%) | 250.1 | 34/50 (68.0%) | 1/50 (2.0%) | 290.9 |
| 7 | 38/50 (76.0%) | 0/50 (0.0%) | 261.7 | 26/50 (52.0%) | 3/50 (6.0%) | 299.2 |
| 8 | 38/50 (76.0%) | 1/50 (2.0%) | 263.2 | 30/50 (60.0%) | 0/50 (0.0%) | 312.1 |
| 9 | 38/50 (76.0%) | 0/50 (0.0%) | 272.1 | 30/50 (60.0%) | 0/50 (0.0%) | 327.3 |
| 10 | 40/50 (80.0%) | 1/50 (2.0%) | 271.5 | 29/50 (58.0%) | 3/50 (6.0%) | 337.0 |

## Readout

- DFlash is consistently more accurate than EAGLE in this SMC setup.
- EAGLE is faster at higher gamma because the local EAGLE SMC path commits a fixed full stride every cycle.
- DFlash at the paper-equivalent LLaMA block size (`gamma=9`, block size `10`) achieved `38/50` accuracy at `272.1 tok/s`.
- EAGLE at the same gamma achieved `30/50` accuracy at `327.3 tok/s`.
- If comparing at similar accuracy, DFlash is already stronger: DFlash `gamma=10` gives `40/50` at `271.5 tok/s`, while EAGLE's closest accuracy point is `gamma=2` with `37/50` at `158.0 tok/s`.

## Research Notes

- The DFlash paper defines speed through `(T_draft + T_verify) / tau`, where `tau` is accepted tokens per verification cycle. DFlash reduces `T_draft` by drafting a block in one pass, but speed still depends heavily on accepted length and verification overhead.
- SGLang DFlash PR #16818 reports DFlash H200 FA3 throughput around `681 tok/s` at TP=1/concurrency=1 against a target-only baseline of `183 tok/s`, and much higher at larger concurrency.
- SGLang DFlash spec-v2/overlap PR #20547 reports `9,688 tok/s -> 12,360 tok/s` at concurrency 32, with mean accept length around `6.47`.
- z-lab/dflash issue #3 reports LLaMA-3.1 GSM8K batch size 1 throughput of DFlash `335.93 tok/s` versus EAGLE-3 `187.00` to `219.46 tok/s`, but this is upstream DFlash/EAGLE benchmarking, not this repository's SMC-modified EAGLE.
- The current SMC EAGLE path is not upstream rejection-based EAGLE: it commits the full `gamma + 1` stride and uses SMC weighting. The current default DFlash path still uses upstream prefix acceptance, so it does not receive the same fixed-stride throughput benefit.

## New Profiling Results

- Added `SMCSD_TIMING=1` instrumentation for DFlash and EAGLE decode paths. It reports draft time, target verify time, other bookkeeping time, per-step latency, and average committed length.
- At `gamma=9`, batch size `1`, `N=12`, DFlash measured `279.6 tok/s`, `14/20` accuracy, and average committed length `4.56`. Time split was about `12.4%` draft, `78.5%` target verify, and `9.1%` other.
- At the same point, EAGLE measured `337.0 tok/s`, `10/20` accuracy, and average committed length `10.00`. Time split was about `27.8%` draft, `41.7%` target verify, and `30.5%` rewrite/bookkeeping.
- This confirms the bottleneck: DFlash draft is already cheap. The gap against high-gamma EAGLE comes from DFlash committing about `4.5` tokens per verify cycle while SMC EAGLE forcibly commits `10`.

## Optimization Experiments

- Larger DFlash blocks did not help this LLaMA checkpoint. `gamma=15` measured `263.5 tok/s`, `15/20` accuracy, and average committed length around `4.32`, worse than `gamma=9`.
- Forcing DFlash to commit post-mismatch tokens is not viable. `SMCSD_DFLASH_MIN_ACCEPT=3` reached `303.0 tok/s` but fell to `3/20`; `MIN_ACCEPT=4` reached `396.6 tok/s` but fell to `1/20`; `MIN_ACCEPT=5` and `6` collapsed to `0/20`.
- Reducing particles did not materially improve single-request throughput because the GPU step time stayed nearly flat. DFlash `N=4`, `gamma=9` measured `297.1 tok/s` and `17/20`, but not enough to beat high-gamma EAGLE raw throughput.
- Concurrency is the useful serving-side optimization. With batch size `4`, `N=12`, `gamma=9`, DFlash measured `40/50` accuracy at `526.2 tok/s`. At the same concurrency, an accuracy-nearer EAGLE point (`gamma=2`) measured `36/50` at `390.8 tok/s`. EAGLE `gamma=9` measured `33/50` at `834.1 tok/s`, but that is the low-quality full-stride regime.
- The GSM8K benchmark now auto-sizes `max_running_requests` from `batch_size * particles` when the flag is omitted. This avoids silently under-provisioning batched SMC runs.

## High-Concurrency Follow-Up

- Re-ran the recommendation frontier with the current code on the first `50` GSM8K test questions.
- At `N=12`, DFlash `gamma=9`, batch size `4` measured `41/50`, `0/50` invalid, and `514.4 tok/s`. EAGLE `gamma=2`, batch size `4` measured `34/50`, `1/50` invalid, and `420.1 tok/s`. This is a clean accuracy-matched DFlash win at moderate concurrency.
- At `N=12`, EAGLE still wins raw throughput by spending accuracy. EAGLE `gamma=9`, batch size `4` measured `28/50`, `4/50` invalid, and `862.9 tok/s`; EAGLE `gamma=1`, batch size `32` measured `39/50`, `1/50` invalid, and `995.4 tok/s`.
- Scaling DFlash alone helped but saturated below the fastest `N=12` EAGLE points: DFlash `gamma=9`, batch size `8` measured `39/50` at `579.9 tok/s`; batch size `16` measured `39/50` at `627.3 tok/s`; batch size `32` measured `40/50` at `649.0 tok/s`.
- Reducing particles is the strongest throughput knob for DFlash at high concurrency. DFlash `N=4`, `gamma=9`, batch size `32` measured `41/50`, `0/50` invalid, and `1546.9 tok/s`.
- At the same `N=4`, batch size `32` setting, EAGLE `gamma=1` measured `37/50`, `2/50` invalid, and `1329.2 tok/s`. DFlash therefore beats this nearest-quality EAGLE point on both accuracy and throughput. EAGLE can still run faster by accepting more quality loss: `gamma=2` measured `32/50` at `1799.6 tok/s`, and `gamma=9` measured `26/50` at `3205.5 tok/s`.

## Current Recommendation

- Keep DFlash on the prefix-verified native SGLang path for correctness.
- Use `gamma=9` or `gamma=10` for the LLaMA DFlash checkpoint; larger blocks did not improve accepted length.
- Compare DFlash to EAGLE at accuracy-matched points, not only at the same gamma. Under that comparison, DFlash now clearly wins at moderate concurrency, and the best current high-concurrency point is DFlash `N=4`, `gamma=9`, batch size `32`: `41/50` at `1546.9 tok/s`, beating EAGLE `N=4`, `gamma=1`, batch size `32`: `37/50` at `1329.2 tok/s`.
- Do not enable forced DFlash minimum acceptance for production; the experiment is retained only as a disabled research knob.

## Follow-Up Bottleneck Investigation

- DFlash and EAGLE spend time in different places:
  - DFlash `gamma=9`: about `12%` draft, `78%` target verify, `9%` bookkeeping, average committed length around `4.5`.
  - EAGLE `gamma=9`: about `28%` draft, `42%` target verify, `30%` rewrite/bookkeeping, fixed committed length `10`.
- The target verify percentage is high for DFlash because the draft is cheap and the accepted length is only about half of EAGLE's forced stride. In absolute step time, DFlash is faster per cycle, but it needs more cycles per output token.
- Upstream DFlash spec-v2/overlap PRs (`#20547`, `#23000`) are not present in this vendored SGLang tree. The patch adds `DFlashWorkerV2`, `DFlashDraftInputV2`, scheduler overlap hooks, `maybe_wait_verify_done`, overlap-plan streams, auto-memory sizing, and request restrictions. This is a vendor-port-sized change, not a small local flag.
- `page_size=64` cannot currently be tested in SMC: `ServerArgs` rejects `speculative_algorithm=SMC` with `page_size != 1`. The deeper reason is correctness: SMC particles share KV prefixes, and paged KV allocation shares physical partial pages. If two particles diverge after a shared partial page, extending in-place can overwrite the same physical page. Correct support needs page-level refcounting plus copy-on-write for the tail partial page, or page-aligned private suffix materialization.
- Available v1 knobs were tested:
  - `--speculative-dflash-draft-window-size 64`: `16/20`, `137.2 tok/s`, average committed length around `2.2`; worse because compacting the draft cache harms acceptance.
  - `--speculative-draft-attention-backend triton`: `13/20`, `269.0 tok/s`; slightly worse than default `fa3`.
  - `fa4` draft attention is rejected by SMC validation; current SMC only accepts `fa3` and `triton`.

## Optimization Plan

1. Done: add instrumentation before changing behavior:
   - report average committed length for DFlash and EAGLE;
   - report per-phase timing for DFlash and EAGLE, not just dense mode;
   - separate draft time, target verify time, and other decode bookkeeping.

2. Establish upstream parity baselines on this machine:
   - run native SGLang DFlash outside SMC with the same LLaMA checkpoint and block size 10;
   - run upstream EAGLE/SpecForge-style settings where possible;
   - run with greedy decoding (`temperature=0`) because most upstream DFlash benchmarks are greedy.

3. Port the upstream DFlash performance optimizations into the SMC path:
   - use DFlash spec-v2/overlap scheduling ideas where possible;
   - evaluate larger page sizes, since upstream DFlash benchmarks use `--page-size 64` while current SMC state currently requires page size 1;
   - ensure the DFlash draft path is using the fastest available draft attention backend and CUDA graph path;
   - avoid redundant allocations and repeated req-to-token restore work in `_dflash_native_propose_batch`.

4. Make DFlash SMC semantics symmetric with EAGLE without committing incoherent raw blocks:
   - do not simply force full-block acceptance; that was tested and quality collapsed;
   - investigate confidence/acceptance-aware SMC weighting, where DFlash's accepted-prefix signal contributes to particle weights while preserving correct KV state;
   - explore block repair/resampling strategies that fill low-confidence suffixes without a full extra autoregressive target decode;
   - keep the working prefix-verified DFlash path as the correctness baseline.

5. Continue re-running sweeps after each change:
   - compare DFlash and EAGLE at the same gamma;
   - compare DFlash against the EAGLE point with similar accuracy;
   - track both output tok/s and accuracy, since raw tok/s alone rewards degraded high-gamma EAGLE runs.
