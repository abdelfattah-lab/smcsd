# Terminal-Bench 2.1 agent integration

The first agentic scope is **turn-local SMC**: Pi owns one Terminal-Bench
container and executes tools, while SM-CSD serves every model turn. Particles
branch only while decoding the next assistant response. They do not own cloned
terminal environments.

## External harnesses

Keep the benchmark repositories outside this repository:

```bash
mkdir -p ~/agentbench
git clone https://github.com/Pinned-Memory/tb-with-pi.git ~/agentbench/substrate
git clone https://github.com/harbor-framework/terminal-bench-2-1.git \
  ~/agentbench/terminal-bench-2-1
```

The checkout name `substrate` is required by its Harbor adapter's Python module
path. Pin both repositories to recorded commits before collecting paper results.

## 1. Launch SM-CSD

The Docker launcher uses the locally available SGLang image and mounts the
checkout's patched SGLang source plus the Hugging Face cache:

```bash
GPU_DEVICE=0 \
TARGET_MODEL=Qwen/Qwen3.5-9B \
DRAFT_MODEL=Qwen/Qwen3.5-2B \
PARTICLES=4 GAMMA=4 \
RANDOM_SEED=0 \
scripts/terminal_bench/launch_smcsd_pi_docker.sh
```

The server advertises the target model through the OpenAI API and enables
automatic Qwen reasoning/tool parser detection.

Qwen3.5-2B/Qwen3.5-9B now captures target prefill/verify, draft verify/decode,
and deferred-cycle CUDA graphs on B200. Runtime logs also show graph dispatch.
`DISABLE_CUDA_GRAPH=true` and `DISABLE_FLASHINFER_AUTOTUNE=true` remain
available as correctness/debugging fallbacks; do not use them for final
performance numbers.

For a short dispatch audit, launch with
`SMC_GRAPH_STATS=1 SMC_GRAPH_STATS_INTERVAL=1`. The B200 tool contract emitted
eight `cycle_graph` hits in eight decode cycles and no fallback.

The agent launcher skips SGLang's generic server warmup by default because its
Qwen3.5 warmup includes multimodal input, which turn-local SM-CSD intentionally
does not support. The explicit streamed text/tool contract test is the warmup
and readiness check for this path.

## 2. Launch the matched target-only AR baseline

Use stock SGLang with the same target, OpenAI tool interface, seed, metrics,
and request limit. This process loads no draft model and does not use the SMC
scheduler:

```bash
GPU_DEVICE=0 \
TARGET_MODEL=Qwen/Qwen3.5-9B \
RANDOM_SEED=0 \
scripts/terminal_bench/launch_ar_pi_docker.sh
```

Only one server may own a port at a time. Both launchers enable `/metrics` by
default so every run gets the same Prometheus instrumentation.

## 3. Validate the Pi API contract

In a second shell:

```bash
python scripts/terminal_bench/smcsd_pi_contract.py \
  --base-url http://127.0.0.1:30000 \
  --model Qwen/Qwen3.5-9B
```

This prompts one automatic streamed tool call, validates its JSON arguments,
returns a synthetic tool result, and confirms that a second turn can consume
the complete assistant/tool history. Named/required tool choice is not used
because SGLang implements it as constrained decoding, which SM-CSD does not yet
support.

## 4. Run one or several Terminal-Bench tasks

Install Harbor, then run:

```bash
SUBSTRATE_ROOT=$HOME/agentbench/substrate \
TB21_ROOT=$HOME/agentbench/terminal-bench-2-1 \
scripts/terminal_bench/run_pi_smoke.sh
```

The default task is `fix-git`, with one trial and one active task. Request
capture is enabled so the exact Pi workload can later be replayed against AR,
SM-CSD, and optimized server configurations.

Pass a comma-separated task list and a stable job name when needed:

```bash
TASK_IDS=fix-git,regex-log \
JOB_NAME=manual-agent-smoke \
SUBSTRATE_ROOT=$HOME/agentbench/substrate \
TB21_ROOT=$HOME/agentbench/terminal-bench-2-1 \
scripts/terminal_bench/run_pi_smoke.sh
```

Harbor writes run artifacts to the `jobs/` directory beside the external
`substrate` checkout by default, keeping trajectories and captured provider
payloads out of this repository. Override the location with `JOBS_DIR`.

## 5. Frozen likelihood-only development matrix

`configs/terminal_bench/likelihood_dev_v1.json` freezes 12 development tasks,
the two model names, external repository commits, Pi settings, three seeds,
target AR, and the SM-CSD Cartesian sweep `N={1,4,8,16}` ×
`gamma={2,4,8}`. It expands to 39 Harbor jobs and 468 task trials. This is a
tuning set, not the held-out paper test set.

Preview the whole matrix without launching anything:

```bash
python scripts/terminal_bench/run_likelihood_matrix.py
```

Run a small filtered pilot:

```bash
python scripts/terminal_bench/run_likelihood_matrix.py \
  --execute \
  --methods ar,smcsd --particles 4 --gamma 4 --seeds 0 \
  --tasks fix-git,regex-log --run-tag pilot-fix-git-regex --gpu 0
```

Omit the filters after the pilot to run the full sequential matrix. The runner
refuses to overwrite jobs, checks pinned external commits, launches and stops
each server, validates the tool-call contract before measurement, and saves
`experiment.json` plus pre/post Prometheus snapshots in each Harbor job.

Summarize completed settings with:

```bash
python scripts/terminal_bench/summarize_pi_jobs.py ~/agentbench/jobs \
  --experiment-id tb21-likelihood-dev-v1 \
  --run-tag full \
  --csv-output likelihood-dev.csv \
  --trials-csv-output likelihood-dev-trials.csv
```

The summary joins Terminal-Bench reward, agent wall time, Pi request/token
counts, correct tasks per agent GPU-hour, and SGLang request latency, TTFT,
inter-token latency, request rate, and model token counters. Its server QPS is
active model requests divided by summed model-request latency. Saturated QPS
and latency percentiles still require the frozen trace-replay benchmark.

### No-resampling diversity ablation

`configs/terminal_bench/no_resample_dev_v1.json` freezes the seed-0
`N={16,32}` × `gamma={4,8}` ablation with ESS resampling disabled. Preview it
with:

```bash
python scripts/terminal_bench/run_likelihood_matrix.py \
  --manifest configs/terminal_bench/no_resample_dev_v1.json \
  --run-tag no-resample-seed0
```

The effective resampling threshold is stored in every job's `experiment.json`
and included in aggregate/trial CSVs. `SMC_SMC_STATS=1` enables ESS telemetry
for separate diagnostic runs only; it synchronizes the GPU every cycle and
must remain disabled for performance measurements.

## B200 smoke result (2026-08-21)

The first end-to-end smoke used one NVIDIA B200, Qwen3.5-2B as draft,
Qwen3.5-9B as target, `gamma=4`, Pi 0.84.2 with thinking disabled, and one
`fix-git` trial at each particle count. Each server was warmed with the API
contract above before Harbor's agent-execution timer began.

| Particles | Reward | Agent wall time | Provider requests | Input tokens | Output tokens | Observed outcome |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 18.25 s | 19 | 76,723 | 1,332 | Found the reflog but incorrectly concluded the lost work was already on `master`. |
| 4 | 0 | 21.26 s | 14 | 39,741 | Recovered the about-page change but missed the lost layout change. |
| 8 | 1 | 15.16 s | 13 | 34,471 | Merged the lost commit, resolved the conflict, and restored both expected files. |

These are correctness smoke tests, not paper results: there is only one task
and one stochastic trial per setting, and different trajectories perform
different amounts of tool and model work. They show that the integration is
functional and that particle count can change an agent trajectory. A claim
requires a frozen task set, multiple seeds, confidence intervals, and the
matched-accuracy serving protocol in `steps.md`.

The smoke used the eager correctness path, so none of the wall times above are
suitable for the final serving comparison. The later dense-draft hybrid layer
map fix unblocked Qwen3.5 FlashInfer autotuning and CUDA-graph capture. Exact
revisions for the original eager smoke were:

- SM-CSD base `438c2c391e5679c7a605ce6ad5fdb2d45f3c679b` plus the `tts`
  working-tree integration changes;
- `tb-with-pi` `ccb5e129d6cdbd8f761143e253adef48f9e6a477`;
- Terminal-Bench 2.1 `7131e4375048a0e408a8fb404b5f499d726b695b`;
- Harbor 0.21.0 and SGLang image digest
  `sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0`.

## CUDA-graph AR/SM-CSD pilot (2026-08-21)

The matrix runner was validated on the same B200 with one `fix-git` trial for
target-only AR and SM-CSD `N=4, gamma=4`. Both received reward 1 and populated
all job sidecars and Prometheus snapshots.

| Setting | Reward | Agent wall | Mean model query | Mean TTFT | Active model queries/s | Generation tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| target AR | 1 | 6.3 s | 0.304 s | 0.049 s | 3.285 | 196.1 |
| SM-CSD N=4, gamma=4 | 1 | 3.7 s | 0.327 s | 0.327 s | 3.056 | 189.1 |

This only validates the harness. The trajectories used different numbers of
turns, and one task has no statistical meaning; do not cite it as a speed or
quality result.

## Correctness boundary

This integration supports the paper claim “SM-CSD as an inference backend for
terminal agents.” It is not trajectory-level SMC. The latter would require a
separate container snapshot, transcript, model KV lineage, and resampling
lifecycle for every particle.
