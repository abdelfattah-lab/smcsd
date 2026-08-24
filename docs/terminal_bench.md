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

If a larger particle population exhausts the frozen 0.4 KV-cache budget, use
`--mem-fraction-static` for a separately reported feasibility retry. The
effective value is stored in `experiment.json` and both summary CSVs.

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

## Semantic actionability checkpoint export

`configs/terminal_bench/semantic_actionability_smoke_v1.json` freezes the
semantic-only scaling axes and a two-task/two-trajectory artifact smoke. It
includes checkpoint intervals `{256,512,2048,8192}`, particle counts
`{4,8,16,32,64}`, verifier-call counts `{1,2,4,8}`, and one versus two
verifier models.

Export the smoke dataset from the completed local jobs:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python \
  scripts/terminal_bench/export_semantic_checkpoints.py \
  --jobs-dir ~/agentbench/jobs \
  --output-dir \
  ~/agentbench/semantic-actionability/tb21-semantic-actionability-smoke-v1
```

The exporter reconstructs exact model-output token IDs with the pinned Qwen
tokenizer. It stores a checkpoint every 256 cumulative completion tokens and
after every observed tool result. The larger token intervals are views over
the same 256-token records, so generation is not repeated.

The artifact has three deliberately separated data files:

- `checkpoints.jsonl`: semantic-verifier inputs and serialized model prefix
  state, with no reward or grader fields;
- `trajectories.jsonl`: source and generator/tool/verifier cost accounting;
- `labels.jsonl`: final Terminal-Bench rewards, which must never be supplied
  to a semantic verifier.

The 2026-08-24 smoke exported four trajectories across `fix-git` and
`query-optimize`. It produced 123 checkpoints: 39 exact token checkpoints and
84 post-tool checkpoints. Token reconstruction, stable IDs, label separation,
and serialization passed. Agent/trial wall times are exact. Because source
jobs serve concurrent trajectories on one shared GPU, accelerator allocation
is reported exactly for each complete source job and is deliberately not
attributed to individual trajectories.

Existing Harbor artifacts do not contain restorable terminal-container
snapshots. The stored language-model prefix is resumable by an engine, but the
terminal environment is not yet cloneable. This does not block the offline
semantic-actionability study. It must be solved before true trajectory-level
online SMC.
### Frozen development actionability screen

`semantic_actionability_generation_dev_v1.json` collects 12 tasks × 8 attempts
× 3 seeds with Qwen3.5-9B. Run one seed per GPU; the three commands can run in
parallel:

```bash
.venv/bin/python scripts/terminal_bench/run_likelihood_matrix.py \
  --manifest configs/terminal_bench/semantic_actionability_generation_dev_v1.json \
  --execute --methods ar --seeds 0 --run-tag pool --gpu 0 --port 30000 \
  --keep-going
```

Repeat with `(seed, gpu, port)` equal to `(1, 1, 30001)` and
`(2, 2, 30002)`. After all three jobs are complete, export the immutable
checkpoint artifact:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python \
  scripts/terminal_bench/export_semantic_checkpoints.py \
  --manifest configs/terminal_bench/semantic_actionability_dev_v1.json \
  --jobs-dir ~/agentbench/jobs \
  --output-dir \
  ~/agentbench/semantic-actionability/tb21-semantic-actionability-dev-v1
```

The dev manifest pins Qwen3.8-27B and Qwen3-32B as the two semantic verifiers,
including exact cached revisions. Score each verifier in a separate process
and GPU. The scorer reads only `checkpoints.jsonl`; it never opens labels:

```bash
CUDA_HOME=/data/home/yahya/cuda-13.0 \
PATH=/data/home/yahya/cuda-13.0/bin:$PATH \
LD_LIBRARY_PATH=/data/home/yahya/cuda-13.0/lib HF_HUB_OFFLINE=1 \
.venv/bin/python scripts/terminal_bench/score_semantic_checkpoints.py \
  --checkpoints ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/checkpoints.jsonl \
  --scorer Qwen/Qwen3.8-27B --base-gpu-id 3 \
  --max-mamba-cache-size 32 \
  --scores-output ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/qwen38_27b.scores.jsonl \
  --summary-output ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/qwen38_27b.summary.json
```

Use the same command with Qwen3-32B on GPU 4 and its own score/summary paths.
Scoring is append-only and resumes by stable verifier-call ID. A `.runs.jsonl`
sidecar records every scorer process's full GPU allocation window, including
model load and graph capture; the summary separately reports allocation and
inference time.

For multi-GPU scoring, partition deterministically with `--num-shards` and
`--shard-index`; a tail process can accept repeated `--completed-score`
paths without duplicating earlier calls. Merge finished parts with
`scripts/terminal_bench/merge_semantic_score_shards.py`. The merger rejects
duplicate IDs, missing checkpoints, mixed models/criteria, and absent cost
ledgers before atomically writing the canonical score and summary files.

Finally, run the label-aware analyzer only after both score files are sealed:

```bash
.venv/bin/python scripts/terminal_bench/analyze_semantic_actionability.py \
  --manifest configs/terminal_bench/semantic_actionability_dev_v1.json \
  --trajectories ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/trajectories.jsonl \
  --labels ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/labels.jsonl \
  --scores ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/qwen38_27b.scores.jsonl \
  --scores ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/qwen3_32b.scores.jsonl \
  --score-summary ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/qwen38_27b.summary.json \
  --score-summary ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/qwen3_32b.summary.json \
  --output ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/actionability.json \
  --markdown-output ~/agentbench/semantic-actionability/\
tb21-semantic-actionability-dev-v1/actionability.md
```

The analyzer reports group-balanced within-task/checkpoint ranking, top-half
correct-trajectory survival lift, task-bootstrap confidence intervals,
cross-verifier error correlation, and leakage-safe leave-one-task-out ensemble
weights. It enforces the frozen gate from the manifest before permitting the
online N=64 / eight-verifier-call sweep.

### Development result (2026-08-24)

The sealed pool contains 288 trajectories across 12 tasks: 134 successful and
154 failed. Export produced 22,258 reward-isolated checkpoints: 10,472 exact
256-token checkpoints and 11,786 post-tool checkpoints. Both verifier files
cover every checkpoint exactly once.

| scorer | checkpoint view | rank accuracy | task-bootstrap 95% CI | top-half survival lift |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-27B | 256 tokens | 0.612 | [0.569, 0.665] | 0.125 |
| Qwen3-32B | 256 tokens | 0.581 | [0.547, 0.622] | 0.082 |
| out-of-task ensemble | 256 tokens | 0.613 | [0.572, 0.664] | 0.121 |
| Qwen3.8-27B | post-tool | 0.630 | [0.583, 0.676] | 0.116 |

The frozen promotion gate passes all six clauses. At its selected 256-token
view, the ensemble covers 537 mixed groups in 11 mixed tasks and 16,226
successful-versus-failed comparisons. Its lower confidence bound exceeds 0.5,
and its 0.121 survival lift exceeds the predeclared 0.10 threshold. The best
single scorer is only 0.00092 behind, however, and cross-verifier error
correlation is 0.940. This promotes semantic online SMC, not unconditional
two-model scoring.

Verifier scoring used 9,590 allocated GPU-seconds for Qwen3.8-27B and 16,393
for Qwen3-32B, including every process startup and resumed shard, and consumed
370.1 million prompt tokens across 44,516 calls. Qwen3-32B put a mean 0.287
probability mass on the requested score-label tokens versus 0.968 for
Qwen3.8-27B, which is an additional reason to start with the latter.

The next prerequisite is a cloneable terminal-particle backend coupling
filesystem/process state, transcript, model-prefix/KV lineage, and cost state.
Only after fork/resume equivalence tests pass should the online one-factor
sweeps over particles through 64 and independent verifier calls through 8 run.
Until then, resampling saved trajectories would be an offline allocation
simulation rather than an online semantic-SMC result.
