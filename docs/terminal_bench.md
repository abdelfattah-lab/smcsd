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

The deterministic replay backend described below now satisfies the environment
cloneability prerequisite for the supported Pi tool subset. The remaining
prerequisite is a host-side online controller that advances and resamples the
environment, transcript, and model continuation as one particle.

### Deterministic terminal-particle backend (2026-08-24)

`scripts/terminal_bench/terminal_particle_backend.py` implements the fallback
selected for this host. Docker 29.6.2 is available, but daemon experimental
mode and CRIU checkpoint/restore are unavailable. The backend therefore starts
a content-addressed task image, replays completed Pi tool calls, and validates
semantic filesystem state plus surviving-process fingerprints. Its manifest
also binds the task, source transcript, checkpoint/model-prefix identity,
lineage, replay calls, and exact replay cost.

Useful entry points are:

```bash
python scripts/terminal_bench/terminal_particle_backend.py capabilities
python scripts/terminal_bench/terminal_particle_backend.py smoke \
  --image alexgshaw/fix-git:20260403 \
  --output ~/agentbench/semantic-actionability/terminal-particle-backend-smoke.json

python scripts/terminal_bench/terminal_particle_backend.py extract \
  --session /path/to/pi-session.jsonl \
  --image alexgshaw/fix-git:20260403 \
  --source-container quiescent-source-container \
  --task-id fix-git --workdir /app/personal-site \
  --state-root /app/personal-site --max-tool-calls 12 \
  --checkpoint-id trajectory-tool12 --output particle.json

python scripts/terminal_bench/terminal_particle_backend.py fork \
  --manifest particle.json --name semantic-particle-0 \
  --ledger-output semantic-particle-0.ledger.json
```

`--source-container` seals the expected state from a live, quiescent source and
causes every child to fail closed on filesystem or process divergence. When a
historical source container no longer exists, deterministic sibling equality
and the task verifier provide the independent checks.

Validation on the pinned `fix-git` image produced:

- synthetic replay: four mutating calls in 0.407 seconds, with identical
  filesystem digest and identical task-process fingerprint, including a
  surviving background process;
- full real replay: all 18 calls from a reward-1 frozen trajectory in 1.320
  seconds, reproducing Git HEAD `cecc2e5` and both exact benchmark verifier
  hashes (`027310...` and `0f8793...`);
- mid-trajectory fork: two independent replays of the first 12 calls produced
  the same semantic filesystem digest
  `24c15bb11ea776bdaf111bedb8825a4b516fb3fdc9ecc6d3a008e0190829baeb`
  and the same surviving-process fingerprint.

The filesystem digest hashes file bytes, executable bits, symlinks, and
semantic Git state while normalizing volatile `.git/index` stat caches and Git
reflog timestamps. Tool timestamps freeze Git author/committer time and
`SOURCE_DATE_EPOCH`, so replayed commits retain the source identity.

The frozen 288-trajectory pool audit finds 253 sessions (87.85%) immediately
replayable. The other 35 fail closed: 29 include a failed `edit`, two include a
failed `write`, and seven contain an unsupported/malformed tool name; three
sessions have overlapping reasons. Failed write/edit atomicity has not yet
been proven, so these events are deliberately not guessed through.

### Online two-sibling controller (2026-08-24)

`scripts/terminal_bench/terminal_particle_controller.py` now owns the coupled
online state. A shared OpenAI-compatible model server receives each particle's
exact serialized message prefix; the controller routes every returned
`bash`/`read`/`write`/`edit` call to that particle's Docker container,
seals the environment and transcript together, and resamples a survivor by
validated replay. The global ledger counts actual model calls/tokens, tool
calls, and replay work once rather than summing duplicated ancestral costs.

The real checkpoint test is:

```bash
python scripts/terminal_bench/terminal_particle_controller.py live-smoke \
  --manifest ~/agentbench/semantic-actionability/\
fix-git-prefix12-replay-manifest.json \
  --capture /path/to/requests-1224.jsonl \
  --after-tool-call-id call_c18c2fb811214ce7982ae505 \
  --base-url http://127.0.0.1:30000 \
  --pre-resample-turns 2 --post-resample-turns 1 \
  --output ~/agentbench/semantic-actionability/\
terminal-particle-controller-live-smoke.json

python scripts/terminal_bench/terminal_particle_controller.py restore \
  --checkpoint ~/agentbench/semantic-actionability/\
terminal-particle-controller-live-smoke.particles/slot-0.json \
  --name restored-particle --cleanup
```

With Qwen3.5-9B and the real reward-1 `fix-git` tool-12 checkpoint:

- both initial siblings had exactly the same filesystem/process state and the
  same serialized model-prefix hash;
- two independently seeded continuation rounds produced two distinct model
  prefixes and two distinct filesystems;
- resampling slot 0 over slot 1 copied filesystem, surviving processes,
  30-message prefix, and lineage exactly;
- both resampled siblings then made another live model/tool transition;
- the final sidecars each contain 15 replay events and 32 messages, with the
  message hash bound into the replay manifest;
- restoring one sidecar from the pinned image replayed all 15 events in 1.141
  seconds and reproduced its sealed filesystem/process state.

The final coupling run made six model calls and six tool calls in 5.259 seconds:
39,622 prompt tokens, 290 completion tokens, 1.742 aggregate model-request
seconds, 0.322 tool seconds, and 3.097 seconds for the three environment
materializations. These are active correctness-test costs. Model load/graph
capture and the server's full allocation window were shared with contract and
debug runs, so this artifact is not a serving-cost comparison.

Safe tool validation failures that provably occur before mutation are recorded
as replayable no-ops; failures after mutation begins still fail closed. The
controller stores exact OpenAI messages and benefits from server prefix caching,
but it does not yet export a persistent KV-cache handle.

This closes the two-sibling correctness milestone. Selection in the smoke test
is deliberately fixed to slot 0; no semantic verifier is used to choose it.

### Live semantic-only SMC pilot (2026-08-24)

`scripts/terminal_bench/terminal_semantic_smc.py` connects Qwen3.5-9B live
particles to the frozen, reward-isolated Qwen3.8-27B recoverability scorer.
Semantic score differences update log weights, `beta` controls their strength,
ESS triggers systematic resampling, and the best terminal semantic score selects
the returned particle. Identical cloned prefixes are scored once. Calls 2--8
use independently worded evaluation lenses and are averaged; this pilot uses
one verifier model, not a heterogeneous model ensemble.

The controller starts at the initial captured provider request and supports
generic `N`. Requested token interval `H` is aligned to the first completed
assistant/tool turn at or after the target. It is not an exact partial-assistant
checkpoint because the OpenAI-compatible generator exposes messages but no
persistent mid-assistant KV continuation handle.

The frozen single-task pilot and aggregation command are:

```bash
python scripts/terminal_bench/terminal_semantic_smc.py \
  --manifest ~/agentbench/semantic-actionability/fix-git-prefix12-replay-manifest.json \
  --capture /path/to/fix-git/agent/pi-capture/requests-1223.jsonl \
  --generator-base-url http://127.0.0.1:30000 \
  --verifier-base-url http://127.0.0.1:30001 \
  --axis particles --values 4,8,16,32,64 \
  --checkpoint-interval 256 --verifier-calls 1 \
  --beta 12 --ess-threshold 0.5 --max-rounds 24 \
  --output ~/agentbench/semantic-actionability/particle-sweep.json

python scripts/terminal_bench/analyze_terminal_semantic_smc.py \
  /path/to/particle-sweep.json /path/to/interval-sweep.json \
  /path/to/verifier-call-sweep.json /path/to/beta-sweep.json \
  /path/to/ess-sweep.json \
  --output /path/to/analysis.json --markdown-output /path/to/analysis.md
```

The exact protocol is frozen in
`configs/terminal_bench/online_semantic_smc_fix_git_pilot_v1.json`. Results:

| sweep | value | repeats | selected | mean population success | mean wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| particles | 4 / 8 / 16 / 32 / 64 | 1 | 100% each | 75.0 / 75.0 / 93.8 / 90.6 / 92.2% | 17.5 / 14.1 / 40.9 / 56.6 / 125.9 |
| interval | 256 / 512 / 2048 / 8192 | 3 | 100% each | 66.7 / 91.7 / 62.5 / 50.0% | 27.1 / 15.6 / 26.7 / 25.6 |
| verifier calls | 1 / 2 / 4 / 8 | 3 | 100% each | 95.8 / 83.3 / 87.5 / 79.2% | 23.0 / 19.5 / 31.3 / 38.2 |
| beta | 4 / 8 / 12 / 24 / 48 | 3 | 100% each | 79.2 / 79.2 / 100 / 91.7 / 100% | 24.3 / 22.5 / 24.2 / 20.2 / 26.3 |
| ESS fraction | .25 / .5 / .75 / .9 | 3 | 100% each | 95.8 / 75.0 / 91.7 / 100% | 22.7 / 23.5 / 23.8 / 28.1 |

Across all 56 runs, the terminal semantic selector chose a reward-1 state.
Whenever a run contained both correct and incorrect final particles, its
terminal semantic ranking AUC was 1.0. Thus this task clearly validates the
semantic verifier as a terminal selector. The repeated interval comparison also
shows preliminary middle-allocation value: `H=512` retains 91.7% correct
particles versus 50.0% for the terminal-only `H=8192` control. It is not yet a
benchmark claim: all points reuse one easy task, stochastic repetitions are not
independent tasks, and `H=256` once collapsed to 1/8 correct particles.

One verifier call is the promoted cost setting: 2/4/8 calls add no selected
accuracy here and eight calls raise mean wall time from 23.0 to 38.2 seconds.
`N=8` is the practical pilot size; `N=64` works correctly but takes 125.9
seconds. A clean restore of the selected N=64 sidecar replayed nine tool calls
in 0.847 seconds and reproduced its sealed model/environment state.

The five sweeps charged 3,154.3 allocated accelerator-seconds across the two
reserved GPUs while excluding external server startup. This is a correctness
and policy pilot, not yet a fast serving result. The next gate is a held-out,
multi-task Terminal-Bench comparison of `N=8,H=512,calls=1` against terminal
semantic Best-of-N and unresampled sampling at matched generator cost. Only
after that should the system add heterogeneous verifier endpoints, exact KV
forks, batching/overlap, conditional extra calls, and SM-CSD draft/target
integration. Speculative `gamma` is absent from this autoregressive controller
and must be swept after that integration rather than relabeled as semantic
`beta`.
