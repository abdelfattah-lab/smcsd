# auto-smc-debug

This is an experiment scripts to debug SMC Algorithms (a kind of speculative decoding algorithm) accuracy and implementation into SGLang. 

## Status
We have a almost ready SMC implementation into SGLang. Basic concepts is that we make a fork on original speculative decoding implementation. Basically we borrow how SGlang's EAGLE_v2 does (`python/sglang/srt/speculative/eagle_worker_v2.py`) with a draft, verify, and draft extend cycle.

We make a tweak to this engine into a `smc_worker_v2.py` which does draft, verify and draft extend as well, but we don't have any rejection (we accepted all tokens). Additionally we compute the logprob difference to get the score so that we can guide SMC. 

Currently, on my debugging, I notice that the accuracy compared to the reference implementation fall short, and it's non-trivial. So I want you to  design experiments to help me tackle. 

You may refer to `/home/cc2869/repositories/sglang/smc_design_docs/smc_architecture_overview.md` for more detail

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar28`). The branch `smc_debug/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b smc_debug/<tag>` from the current working branch (e.g. `smc_v0`).
3. **Read the SMC-related files**: Start with the architecture overview, then read each layer.

   **Architecture doc** (read first):
   - `smc_design_docs/smc_architecture_overview.md` — end-to-end data flow, weight math, KV cache lifecycle

   **Core SMC implementation** (most likely where bugs live):
   - `python/sglang/srt/speculative/smc_info.py` — core data structures (`SMCDraftInput`, `SMCScoreInput`, `SMCDraftInputV2Mixin`), resampling algorithms (`systematic_resample`, `multinomial_resample`), particle cloning, mask/position builders
   - `python/sglang/srt/speculative/smc_worker_v2.py` — `SMCWorkerV2` (extends `EAGLEWorkerV2`) for draft/score/draft-extend orchestration; `SMCDraftWorker` (extends `StandaloneDraftWorker`) for multi-step draft forward
   - `python/sglang/srt/speculative/smc_manager.py` — `SMCManager` + `SMCGroupState`: group lifecycle, synchronous weight updates/resampling, ready-queue admission, finalization, best-particle selection
   - `python/sglang/srt/speculative/smc_draft_cuda_graph_runner.py` — fused γ-step draft CUDA graph (`SMCDraftCudaGraphRunner`, `SMCDraftInputBuffers`, `SMCDraftSamplingSignature`)
   - `python/sglang/srt/speculative/smc_debug_utils.py` — debugging utilities and probe records

   **Scheduler & batch management** (integration layer):
   - `python/sglang/srt/managers/scheduler.py` — `event_loop_normal_smc()`, `event_loop_overlap_smc()`, SMC group waiting queue policy
   - `python/sglang/srt/managers/schedule_batch.py` — `SMCGroupSpan`, `build_smc_group_spans()`, per-request SMC fields (`smc_group_id`, `smc_particle_idx`)
   - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` — post-prefill SMC init, decode result processing
   - `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py` — memory leak detection accounting for SMC held tokens
   - `python/sglang/srt/managers/overlap_utils.py` — `FutureMap` with SMC circular buffers (`last_token_ids_buf`, `new_seq_lens_buf`)
   - `python/sglang/srt/managers/schedule_policy.py` — SMC-aware scheduling policies

   **Model executor** (forward pass & KV cache):
   - `python/sglang/srt/model_executor/model_runner.py` — forward pass with `SMCScoreInput` creation, `materialize_smc_parent_draft_prefix()`
   - `python/sglang/srt/model_executor/cuda_graph_runner.py` — CUDA graph with SMC awareness
   - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` — KV cache capacity multiplier (`max_num_reqs *= 2 * smc_n_particles + 1`)

   **Shared speculative infrastructure** (reference for correctness):
   - `python/sglang/srt/speculative/eagle_worker_v2.py` — reference EAGLE implementation that SMC extends
   - `python/sglang/srt/speculative/spec_info.py` — `SpeculativeAlgorithm` enum (`is_smc()`), `SpecInputType.SMC_DRAFT`/`SMC_SCORE`
   - `python/sglang/srt/speculative/spec_utils.py` — `generate_smc_draft_decode_kv_indices` and shared utilities

   **Configuration**:
   - `python/sglang/srt/server_args.py` — SMC flags: `smc_n_particles`, `smc_gamma`, `smc_draft_temperature`, `smc_target_temperature`, `smc_resample_threshold`, `smc_resample_method`, `smc_resampling_overlap`
4. **Initialize results.tsv**: Create `results.tsv` with just the header row:
   ```bash
   echo -e "commit\taccuracy\tstatus\tdescription" > results.tsv
   ```
5. **Confirm and go**: Confirm setup looks good with the user.

Once you get confirmation, kick off the experimentation with a baseline run first.

## Experimentation

Each experiment runs on a single GPU. You launch the benchmark by running:
```bash
source .venv/bin/activate
python scripts/smc/accuracy_test_gsm8k_new.py \
    --mode smc \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    -N 8 -g 32 --num-questions 20 2>&1 | tee run.log
```

**What you CAN do:**
- Modify the SMC-related files under `python/sglang/srt/speculative/` (e.g. `smc_worker_v2.py`, `smc_manager.py`).
- Modify SMC-related scheduler logic in `python/sglang/srt/managers/` (e.g. `scheduler.py`, `schedule_batch.py`, `scheduler_output_processor_mixin.py`).
- Refer to SGLang's reference EAGLE speculative decoding implementation for comparison (e.g. `eagle_worker_v2.py`).

**What you CANNOT do:**
- Install new packages or add dependencies. You can only use what's already installed in python environments
- Modify the evaluation scripts.
- Modify the model combination. You shouldn't hack the performance by switching to smaller models size etc.
- Simply tweak the hyperameter like temperature. (gamma, temperature, N, resample threshold)
- Don't change any SGLang engine args and hyper-mater
- You shouldn't modify the attention backend
- Do not change the the `draft` --> `verify (score)` --> `draft extend`
   - In our SGlang implmenation, we inlcude bonus tokens this is intend, should not need to modified.
   - We always draft `gamma` tokens and no early stopping checking when drafting. This is also intended. Please don't change the behabior, it's the scheduler and output processing responsibility to filter. 
- Don't looking into retraction part, it's out of scope. 


**The goal is: Ensure there are no any subtle bug of implementation of porting SMC into SGLang. And Try to Get the Accuracy as high as possible**
- So far our accuracy is fall behind the simpler implementation even under same number of particles and gamma setting. 
    - For instance, under the 8B + 1B pair, `/home/cc2869/repositories/sglang/scripts/smc/smc_native.py` could achieve higher accuracy.
    - You can reproduce with the native reference (note: `smc_native.py` defaults to the Llama 8B+1B pair already):
      ```bash
      uv run scripts/smc/smc_native.py -N 8 -g 32 --eval
      ```
      Or explicitly:
      ```bash
      uv run scripts/smc/smc_native.py \
          --target-model meta-llama/Llama-3.1-8B-Instruct \
          --draft-model meta-llama/Llama-3.2-1B-Instruct \
          -N 8 -g 32 --eval
      ```
      Or you can aslo refer to 
      `scripts/smc/accuracy_test_gsm8k_new.py` and set mode to `native`

**The first run**: Your very first run should always be to establish the baseline with zero code changes.
    - Run the engine-level SMC benchmark as-is and record the result as `baseline` in `results.tsv`.
    - The reference implementation (`scripts/smc/smc_native.py`) achieves ~62.5% accuracy (on 40 questions with `accuracy_test_gsm8k_new.py`) under the 8B+1B Llama pair. You can reproduce with:
      ```bash
      uv run scripts/smc/smc_native.py -N 8 -g 32 --eval
      ```
    - The goal is to close the gap between engine-level SMC and this reference.

## Output format

The benchmark prints per-sample progress lines followed by a summary block. The key metric to extract is **Accuracy**.

Per-sample progress (printed during the run):
```
[1/20] acc=1/1 (100.0%) tps=45 elapsed=5s
[2/20] acc=2/2 (100.0%) tps=42 elapsed=12s
...
```

Final summary block (printed at the end):
```
=======================================================
  Engine-level SMC
  N=8, γ=32, temp=0.7
=======================================================
  Accuracy:          14/20 (70.0%)
  Invalid:           2/20 (10.0%)
  Output throughput: 45.2 tok/s
  Total tokens:      8234
  Wall time:         182.3s
=======================================================
```

After a run, grep for the accuracy:
```bash
grep "Accuracy:\|Invalid:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	accuracy	status	description
```

1. `commit` — git commit hash (short, 7 chars), from `git rev-parse --short HEAD`
2. `accuracy` — accuracy percentage from the summary block (e.g. `70.0`)
3. `status` — one of: `keep` (improvement or baseline), `discard` (regression), or `crash` (run failed)
4. `description` — short text description of what this experiment tried

To log a result after a successful run:
```bash
COMMIT=$(git rev-parse --short HEAD)
ACC=$(grep "Accuracy:" run.log | grep -oP '[\d.]+(?=%)')
echo -e "${COMMIT}\t${ACC}\t<status>\t<description>" >> results.tsv
```

Example `results.tsv`:
```
commit	accuracy	status	description
a1b2c3d	70.0	keep	baseline
b2c3d4e	72.5	keep	fix logprob accumulation
c3d4e5f	68.0	discard	alternative resampling strategy
d4e5f6g	0.0	crash	broken weight normalization
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `smc_debug/mar28` or `smc_debug/mar28-gpu0`), matching the tag chosen during setup.

LOOP FOREVER:

1. **Check git state**: `git log --oneline -3` to see where we are.
2. **Analyze SMC code**: Check the SMC related files — KV-cache maintenance, SMC group states and scheduler, resample logic under `python/sglang/srt/speculative/` and `python/sglang/srt/managers/`. Also, check whether our tweak from Eagle V2 style have some flaw or mismatch.
3. **Make a change and commit**: Implement a hypothesis on the bug fix, then `git add <files> && git commit -m "<description>"`.
4.0 ** Run the `/home/cc2869/repositories/sglang/scripts/smc/quick_quality_check.py` ensure that no cross request context mixing and non-coherent or clearly wrong output. If ourput is very terriable, say non-readable symbols then your implementation is kinda buggy think again, and give up if you couldn't get up for a few trials. Note that we are using 0.5B model for this, so some mistake might be possible. But it co-herent is generally OK in the model.
4. **Run the benchmark**:
   ```bash
   source .venv/bin/activate
   python scripts/smc/accuracy_test_gsm8k_new.py \
       --mode smc \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --draft-model meta-llama/Llama-3.2-1B-Instruct \
       -N 8 -g 32 \
       --num-questions 40 \
       2>&1 | tee run.log
   ```
5. **Read out the results**:
   ```bash
   grep "Accuracy:\|Invalid:" run.log
   ```
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt some fixes. If you can't get things to work after more than a few attempts, give up.
7. **Record the results** in `results.tsv` (NOTE: do not commit the results.tsv file, leave it untracked by git):
   ```bash
   COMMIT=$(git rev-parse --short HEAD)
   ACC=$(grep "Accuracy:" run.log | grep -oP '[\d.]+(?=%)')
   echo -e "${COMMIT}\t${ACC}\t<status>\t<description>" >> results.tsv
   ```
8. **Decide**: If accuracy improved → `keep`. If accuracy drop → `discard` (revert with `git revert HEAD`). If crashed → `crash`. If accuracy stay like a noise level (~2.5%) → `keep` We didn't run from a lot of samples, also under high-temperature the flunctuation might be normal. 

The idea is that you are a completely autonomous researcher/engineering trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes (no more than 30mins) total (+ a few seconds for startup and eval overhead). If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert). 

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.
- **If the crash is about the radix memory leakage, then this is some good signal. This means that we actually have some bad/unproper KV-Cache maintain.**

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

# Note
All dependency are installed. You can simply run `source .venv/bin/activate` to get the environments to execute desired code. Or you can use `uv run`.
