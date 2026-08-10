# Cross-tokenizer SMCSD

This document has two layers:

1. a model-agnostic onboarding procedure for a new target/draft tokenizer pair;
2. the validated Qwen3-4B/Llama-3.1-70B/GSM8K recipe as a worked example.

The algorithm is model-agnostic, but the current implementation is not
universal. A new pair must satisfy the compatibility requirements below and
pass the prescribed mapper and end-to-end gates before it is considered
supported.

## Support envelope

The current implementation is intended for two text-generation models that:

- are decoder-only autoregressive causal language models supported by the
  vendored SGLang runtime;
- can be loaded as Hugging Face tokenizers with deterministic text
  encode/decode behavior;
- use a dense autoregressive draft path;
- fit under the same SGLang tensor-parallel topology;
- can consume one shared serialized prompt string, even though each model
  tokenizes that string independently;
- expose unambiguous stop/EOS behavior that can be audited for the pair.

The current runtime entry point is the offline `SMCEngine`. Cross-tokenizer
arguments are not wired through `smcsd/http_server.py`.

The following are not currently claimed as generally supported:

- encoder-decoder, diffusion, or non-autoregressive models;
- multimodal prompt tokenization;
- independently rendered target and draft chat templates inside one request;
- arbitrary MTP/speculative draft architectures;
- model pairs whose decoded text cannot round-trip through both tokenizers;
- automatic LoRA-module discovery for every model architecture.

For chat models, choose one canonical prompt serialization and use it
consistently in teacher generation, draft training, standalone evaluation, and
SMC evaluation. `SMCEngine` tokenizes the same prompt string with both
tokenizers; it does not render separate chat templates for each model.

## Validated worked example

The best-validated contract from this repository is:

- target: `meta-llama/Llama-3.1-70B-Instruct`;
- draft base: `Qwen/Qwen3-4B-Instruct-2507`;
- draft adaptation: target sequence knowledge distillation (SeqKD), LoRA rank
  64, alpha 128, dropout 0;
- SMC: 8 particles, gamma 8, target temperature 0.7, draft temperature 0.9,
  resampling threshold 0.5;
- mapper: frozen `hybrid` artifact with boundary healing;
- terminal selection: posterior sampling.

These values are validated defaults for that model pair and workload, not
universal constants.

## Why cross-tokenizer decoding is different

Draft and target token IDs are meaningful only to their own models. Even when
two token sequences decode to the same text, their boundaries can differ:

```text
text                  "16"
Qwen draft tokens     "1", "6"
Llama target token    "16"
```

A per-token lookup is not enough. Tokenization is context-dependent, so
encoding `"1"` and `"6"` separately and concatenating the results need not
equal encoding `"16"` once. Scoring that non-canonical target sequence can
badly distort SMC importance weights.

SMCSD therefore maintains two token histories and two KV-cache lineages:

```text
prompt text
  ├─ draft tokenizer  → draft prefix → draft proposals + log q
  └─ target tokenizer → target prefix

draft proposal block
  → map its decoded bytes into target proxy tokens and heal unsafe seams
  → target scores mapped tokens
  → update particle weights and optionally resample
  → map/retokenize committed text back into draft state
```

Target and draft token IDs are never treated as interchangeable.

## Inference algorithm

For each SMC cycle:

1. Each particle uses the draft model to sample `gamma` draft tokens and records
   their autoregressive log probabilities.
2. `TokenMapper` converts each draft block into target-token space.
3. The target model scores the mapped target tokens in one verification pass.
4. SMC updates each particle with target score minus mapped draft proposal
   charge, normalizes weights, and resamples when ESS crosses the configured
   threshold.
5. The runtime updates the separate target and draft histories/KV tables.
6. At termination, `posterior_sample` draws the final particle according to its
   normalized posterior weight.

The mapper conserves the sampled draft path's total log probability across each
aligned segment. This is proposal accounting for the sampled native draft path;
it should not be interpreted as an exact marginal over every draft
tokenization that could emit the same byte string.

### Mapping modes

`live`
: Decode each special-free draft proposal block as text, encode that text once
  with the target tokenizer, and align draft and target spans. This is the
  correctness reference for a new pair, but it performs tokenizer work on every
  block.

`hybrid`
: Combine a frozen offline lookup artifact with deterministic live fallback and
  boundary repair. Common mappings avoid repeated tokenizer calls, while
  unresolved or context-sensitive runs are still decoded and re-encoded. This
  is the recommended mode only after it passes equivalence tests against
  `live`.

Hybrid mode requires a mapping artifact. The artifact stores tokenizer
fingerprints and is rejected when it does not match the loaded tokenizers.
The fingerprint includes the complete token-to-ID vocabulary, backend tokenizer
rules when available, tokenizer configuration, and special-token semantics.
Still pin tokenizer revisions: the fingerprint verifies the loaded object, not
the provenance of a mutable remote name. Build the artifact with the exact
tokenizer identifiers used at inference. If a merged checkpoint carries a
copied tokenizer under a different local identity, either use the original
pinned tokenizer explicitly at runtime or rebuild against that deployment
tokenizer; do not bypass validation.

“Hybrid” describes the combination of offline lookup and live repair. It does
not refer to a hybrid model architecture, and it is not Byte-Prefix
Marginalization (BPM). Runtime hybrid mapping transforms one sampled native
draft path; it does not marginalize over every draft-token sequence that could
spell the same text.

### What the mapping artifact contains

The builder scans the complete draft vocabulary, independently of any corpus:

1. If a normal draft token has a canonical decoded fragment equal to one target
   token, it records a one-to-one entry.
2. Otherwise it decodes the draft token and target-encodes that text. If the
   result is non-empty and no longer than `max_single_token_span`, it records a
   one-to-many entry.
3. Tokenizer-specific control-looking fragments are excluded from normal text
   entries. If both sides expose one unique equal-surface ID as a genuine
   special token, the builder records it in a separate special-token map;
   otherwise it is blocked. Runtime EOS handling still owns stop semantics.
4. Entries that exceed the span limit or cannot be encoded remain unmapped and
   use live fallback at runtime.

An optional corpus contributes only frequent draft n-gram entries. For each
selected draft tuple, the builder decodes the tuple together and target-encodes
the resulting text once. N-grams are a performance cache for common
context-sensitive runs, not a new correctness rule. They are created only when
`--max-ngram-entries` is greater than zero.

`SMC_CROSS_ONLINE_NGRAM=1` can also promote repeatedly healed runs into an
in-memory n-gram table. It is off by default because the table then depends on
the order of runtime traffic and is not represented by the frozen artifact.
Keep it off for reproducible evaluation.

The artifact also contains construction statistics, a canonical vocabulary
index for diagnostics, and tokenizer metadata. It contains no model weights,
logits, prompt-specific state, or learned probabilities. The same artifact can
therefore be reused for a base draft, LoRA adapter, and merged LoRA checkpoint
when—and only when—the draft tokenizer is unchanged.

### Exact hybrid decision path

For each particle and proposed draft block, `TokenMapper` applies this order:

1. **Identity-vocabulary bypass.** If `draft_tokenizer.get_vocab()` and
   `target_tokenizer.get_vocab()` are exactly equal, token IDs and draft log
   probabilities pass through one-to-one. The normal same-tokenizer SMC path is
   still preferable unless the cross-tokenizer machinery is being tested.
2. **Special/stop override.** An explicit equal-surface special-token artifact
   entry is applied before normal text mapping. A recognized draft EOS-family
   ID maps directly to the target EOS ID and takes priority over that artifact
   entry, so a stop is not scored as literal text such as `"<|im_end|>"`.
3. **Longest n-gram hit.** Hybrid mapping greedily takes the longest matching
   artifact n-gram beginning at the current draft position.
4. **Single-token hit.** If no n-gram matches, it checks the artifact's
   one-to-one or one-to-many entry for the current draft ID.
5. **Live fallback.** If the ID is absent, it decodes that draft token and
   target-encodes the text. An in-memory cache avoids repeating the same
   fallback. Artifact misses therefore normally affect latency, not byte
   correctness.
6. **Empty-fragment repair.** Draft byte-fallback fragments that temporarily
   produce no target token are merged into a neighboring visible event so
   their proposal charge is not silently dropped. If a fully flushed block has
   no visible target event at all, the row uses the same safe target-only
   fallback as an unusable overlength proposal.
7. **Boundary healing.** Artifact entries were encoded in isolation, so simply
   concatenating them can produce a poor target segmentation. Consecutive
   segments separated by an unsafe seam are decoded from the original draft
   tuple and target-encoded together. The healed result is cached.
8. **Proposal-charge placement.** The summed native-draft log probability for
   each aligned event is placed on the event's first target token; subsequent
   target subtokens receive zero proposal charge.

An artifact hit is not an acceptance decision and does not imply that the
target agrees with the draft. Static hits, healed runs, and live fallbacks all
produce target proxy tokens that are scored in the same way. Unlike
token-by-token accept/reject speculative decoding, this SMC path lets particle
weights and resampling determine which complete trajectories survive.

The runtime's current “safe seam” test examines the decoded first character of
the target token to the right of a seam after normalizing common SentencePiece
and byte-BPE markers. Empty fragments and fragments beginning with space,
newline, tab, or carriage return are treated as safe; other seams are healed.
This is a tested heuristic for the validated Qwen/Llama pair, not a theorem for
every tokenizer implementation. A new tokenizer family must be audited with
its own word-internal, whitespace, normalization, and byte-fallback cases.

Boundary healing is local to the currently unsettled proposal run. It does not
rewrite target tokens that have already been committed to the KV cache.
Consequently, “canonical” here means that unsafe internal seams in the mapped
continuation are re-encoded together. It does not promise that hybrid target
IDs are identical to `live` target IDs or to tokenizing the entire accumulated
document from scratch. The required invariants are decoded-byte equality,
proposal-charge conservation, valid stops, and acceptable end-to-end behavior.

### Why proposal charge moves between token boundaries

Suppose one sampled draft token has `log q = -1.25` and maps to two target
tokens:

```text
draft event       [d0]          log q = -1.25
target proxy      [t0, t1]      proxy log q = [-1.25, 0.0]
```

Charging the first target token makes the complete sampled event cost visible
even if decoding stops after `t0`. Conversely, if three draft tokens map to one
target token, their three log probabilities are summed on that target token.
For every non-empty mapped block:

```text
sum(mapped target proposal charges) == sum(sampled draft log probabilities)
```

The target contributes its own autoregressive score at every valid target
proxy position. SMC uses the sum of target log scores minus these conserved
draft proposal charges. The zeroes on later target subtokens do not make those
subtokens free; they only prevent charging the same sampled draft event twice.

### Behavior in different scenarios

**One draft token to one target token**
: The artifact supplies the target ID directly. This is the cheapest common
  path. Equal decoded text is required; equal numeric IDs are not.

**One draft token to several target tokens**
: The target verifies every mapped subtoken. The draft token's complete
  `log q` is charged on the first target subtoken. If the expansion still fits
  within `gamma`, the cycle proceeds normally.

**Several draft tokens to one or fewer target tokens**
: A static n-gram can map the run directly. Otherwise isolated single entries
  are joined and boundary healing re-encodes an unsafe word-internal run. All
  contributing draft log probabilities are summed.

**A token absent from the artifact**
: Hybrid mode performs decode-to-text then target-encode and caches the result.
  A high fallback count indicates a poor artifact or unusual workload, but is
  not by itself an accuracy failure. A fallback that cannot round-trip exactly
  is a compatibility failure.

**Whitespace, punctuation, and word-internal joins**
: A right-hand target token beginning with normalized whitespace ends the
  current heal group. Word-internal joins are re-encoded together. Include
  leading spaces, repeated whitespace, newlines, indentation, punctuation,
  numbers, and code delimiters in pair-specific tests.

**Unicode and byte-fallback tokens**
: Temporary zero-target fragments are attached to the next visible mapped
  event, or to the preceding event when they occur at the end. Test composed
  and decomposed Unicode, emoji, combining marks, and invalid/replacement-byte
  behavior. Reject a pair if decoded bytes are not stable.

**The entire flushed block maps to no target token**
: The row is marked as a discarded proxy, the sampled draft block is removed
  from draft lineage, and the target supplies the next bonus token. This avoids
  silently losing draft proposal charge. With validated boundary carry, an
  empty fragment can instead remain unsettled until a visible event or EOS
  arrives.

**Draft EOS or a common draft stop token**
: Runtime special handling maps recognized draft EOS-family IDs to the target
  EOS ID and protects that segment from boundary healing. Arbitrary control
  tokens are not generically translated, although the artifact can carry a
  unique equal-surface mapping when both IDs are declared special. Each
  configured stop sequence must be tested. Disabling
  `SMC_CROSS_MAP_SPECIALS` is diagnostic only.

**Mapped target length is shorter than `gamma`**
: The target verify batch is padded, but a validity mask excludes padding from
  scores and weights. The target bonus is sampled immediately after the last
  real proxy token.

**Mapped target length exceeds `gamma` with default flushing**
: The row is marked overlength. Its mapped proposal is discarded on both
  target and draft lineages, and the cycle falls back to a target-only bonus
  sampled at the current target prefix. That bonus is then retokenized into
  draft space. This preserves lineage correctness but wastes the draft block
  and reduces throughput. Persistent overlength means the pair's
  draft-to-target expansion exceeds the current verify budget. N-gram coverage
  or validated boundary carry may help, but increasing `gamma` alone is not
  guaranteed to help because it also lengthens the draft block. Otherwise the
  runtime needs an independently larger proxy budget or the pair/configuration
  should be rejected.

**Optional boundary carry**
: With `SMC_CROSS_BOUNDARY_CARRY=1`, the mapper may commit only the longest
  prefix ending at a safe seam and carry the remaining draft suffix into the
  next cycle. No target bonus is emitted while a suffix is unsettled, because
  emitting it first would reverse byte order. EOS flushes the remainder. This
  path is opt-in and must pass state-lineage and end-to-end tests for the exact
  model pair; the validated default uses full-block flushing.

**Target bonus expands to several draft tokens**
: After verification, one sampled target bonus token is decoded and re-encoded
  with the draft tokenizer. Its draft-token prefix is committed to the draft KV
  lineage and its final draft token becomes the next draft seed. The exact
retokenization is cached and is never truncated. The runtime reserves 32 extra
draft KV positions per cycle by default; increase
`SMC_CROSS_DRAFT_BONUS_HEADROOM` if a new tokenizer pair reports a larger
expansion.

**Base draft versus LoRA/merged draft**
: Mapping does not depend on draft weights. Reuse the artifact if the tokenizer
  and its special-token configuration are exactly unchanged. Rebuild and
  re-audit after adding tokens, changing tokenizer files, or changing stop IDs.

**Same family but different tokenizer revision**
: Family names do not activate the identity path. Any vocabulary, ID, added
  token, or special-token difference requires normal cross-tokenizer mapping
  and a pair-specific artifact.

### Reading mapper diagnostics

Set `SMC_CROSS_STATS_INTERVAL` to a positive number to print cumulative mapper
statistics during a run. The final report uses the same fields:

- `trie_hits`: longest n-gram entries used;
- `single_hits`: artifact single-token or explicit special mappings used;
- `cache_hits`: repeated live-fallback tuples served from the runtime cache;
- `fallback_hits` / `dtw_fallbacks`: unmapped events that required runtime
  fallback (`dtw_fallbacks` is a historical field name and does not mean every
  hybrid miss ran the live block aligner);
- `heal_groups`: unsafe multi-segment runs re-tokenized together;
- `heal_reencodes`: heal groups that missed the cache and called both
  tokenizers;
- `overlength_rows`: particle blocks sent through the discard/target-only
  fallback because the mapped proxy exceeded the verify budget or was a fully
  flushed empty proxy;
- `empty_proxy_rows`: the fully empty subset of those discarded rows;
- `carried_rows`: particle blocks with an unsettled suffix when boundary carry
  is enabled.

Nonzero fallback or healing counts are expected and can still be exact.
Correctness requires byte and charge invariants to hold. Performance tuning
then aims to reduce repeated fallback/heal tokenizer calls and overlength rows
without changing those invariants.

## Installation

Install the pinned SGLang submodule first, then SMCSD:

```bash
git submodule update --init --recursive
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e 3rdparty/sglang/python
uv pip install -e .
```

For LoRA training and dataset preparation:

```bash
uv pip install -e ".[train]"
```

Set `HF_TOKEN` in the environment when gated target weights require it. Keep
tokens, model caches, datasets, adapters, and experiment outputs outside Git.

## Model-agnostic workflow

Use these placeholders throughout the rest of the guide:

```bash
export TARGET_MODEL="<target Hugging Face ID or local merged checkpoint>"
export DRAFT_BASE_MODEL="<draft base-model ID>"
export DRAFT_MODEL="<draft base or merged SeqKD checkpoint>"
export TARGET_TOKENIZER="<target tokenizer ID>"
export DRAFT_TOKENIZER="<draft tokenizer ID>"
export DOMAIN_CORPUS="<representative UTF-8 text file>"
export MAPPING_ARTIFACT="artifacts/cross_tokenizer/draft_to_target.json"
```

Onboarding a new pair has five stages:

1. **Compatibility preflight:** load both models and tokenizers independently,
   verify prompt and stop behavior, and confirm the pair fits the runtime
   topology.
2. **Mapper validation:** build a pair-specific artifact and prove byte
   equality, proposal-charge conservation, boundary healing, and stop handling.
3. **Base-draft baseline:** compare `live` and `hybrid` mapping on identical
   prompts before any fine-tuning.
4. **Target SeqKD:** generate target response text, retokenize it with the
   draft tokenizer, and train response-only LoRA.
5. **Matched promotion gates:** compare base and trained drafts under identical
   prompts, seeds, target budget, SMC settings, and mapping artifact.

Do not skip directly to training. A mapper or prompt-serialization defect can
look like poor draft quality, and LoRA cannot repair incorrect runtime
accounting.

## Build and audit a mapping artifact

Artifacts are pair-specific, generated locally, and ignored by Git. Start with
a corpus-independent single-token artifact. This scans the complete draft
vocabulary and is the simplest correctness baseline:

```bash
python -m smcsd.cross_tokenizer.build \
  --draft-tokenizer "$DRAFT_TOKENIZER" \
  --target-tokenizer "$TARGET_TOKENIZER" \
  --output "$MAPPING_ARTIFACT" \
  --trust-remote-code
```

After this artifact matches `live` mode, a representative domain corpus can add
frequent n-gram fast paths:

```bash
python -m smcsd.cross_tokenizer.build \
  --draft-tokenizer "$DRAFT_TOKENIZER" \
  --target-tokenizer "$TARGET_TOKENIZER" \
  --corpus-file "$DOMAIN_CORPUS" \
  --max-ngram 8 \
  --max-ngram-entries 10000 \
  --max-ngram-target-span 16 \
  --output "$MAPPING_ARTIFACT" \
  --trust-remote-code
```

`--corpus-file` has no effect on the artifact when
`--max-ngram-entries=0`, which is the default. The value `10000` above is a
starting cache budget, not a validated universal optimum; compare artifact
size, hit rate, mapper latency, and end-to-end results. The corpus should cover
expected languages, whitespace, punctuation, numbers, code delimiters,
Unicode, and task-specific formatting. It must not contain secrets or
protected evaluation answers.

`--corpus-file` treats each nonempty physical line as one sample. Leading and
trailing whitespace on that line is preserved, but the line separator is not
part of the sample. Exercise actual newline transitions in mapper audit cases
and end-to-end prompts; corpus n-grams are not a substitute for correctness
coverage.

Validated Qwen/Llama example:

```bash
python -m smcsd.cross_tokenizer.build \
  --draft-tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --target-tokenizer meta-llama/Llama-3.1-70B-Instruct \
  --output artifacts/cross_tokenizer/qwen3-4b_to_llama3.1-70b.json \
  --trust-remote-code
```

Audit hybrid mapping against live mapping before a full model run. The audit
script itself is tokenizer-pair agnostic:

```bash
python scripts/audit_cross_tokenizer_mapper_modes.py \
  --draft-tokenizer "$DRAFT_TOKENIZER" \
  --target-tokenizer "$TARGET_TOKENIZER" \
  --artifact "$MAPPING_ARTIFACT" \
  --output /tmp/cross_tokenizer_mapper_audit.json
```

Validated Qwen/Llama example:

```bash
python scripts/audit_cross_tokenizer_mapper_modes.py \
  --draft-tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --target-tokenizer meta-llama/Llama-3.1-70B-Instruct \
  --artifact artifacts/cross_tokenizer/qwen3-4b_to_llama3.1-70b.json \
  --output /tmp/qwen3_llama70b_mapper_audit.json
```

The audit must report byte equality and proposal-charge conservation for every
case and exits nonzero if any built-in check fails, so it is safe to use as a
CI gate. Its built-in cases are only a smoke suite; add pair- and
domain-specific cases for whitespace, Unicode, code, stop strings, and
tokenizer boundary patterns observed in the intended workload.

This script maps each case as one complete block. It does not exercise actual
model-generated blocks, `gamma` overlength behavior, cross-cycle carry, target
bonus retokenization, KV lineage, resampling, or termination. Cover those in a
small end-to-end `live`/`hybrid` comparison before a production benchmark.

## Run cross-tokenizer SMCSD

Start a new pair in `live` mode so no offline artifact can hide a mapping
defect:

```python
import os

from smcsd import SMCEngine

engine = SMCEngine(
    model_path=os.environ["TARGET_MODEL"],
    draft_model_path=os.environ["DRAFT_MODEL"],
    tokenizer_path=os.environ["TARGET_TOKENIZER"],
    draft_tokenizer_path=os.environ["DRAFT_TOKENIZER"],
    cross_tokenizer=True,
    cross_tokenizer_mode="live",
    n_particles=4,
    gamma=4,
    final_selection="posterior_sample",
)
try:
    result = engine.generate(
        prompt="<the exact serialized prompt used by both models>",
        sampling_params={"temperature": 0.7, "max_new_tokens": 128},
    )
finally:
    engine.shutdown()
```

This is a correctness smoke test, not a promoted configuration. Once live-mode
outputs and diagnostics are sound, repeat the same prompts in `hybrid` mode
with `cross_tokenizer_artifact_path="$MAPPING_ARTIFACT"` and compare decoded
bytes, answers, and diagnostics.

Validated Qwen/Llama programmatic example:

```python
import os

from smcsd import SMCEngine

os.environ["SMC_CROSS_ONLINE_NGRAM"] = "0"

engine = SMCEngine(
    model_path="meta-llama/Llama-3.1-70B-Instruct",
    draft_model_path="/models/qwen3-4b-seqkd-merged",
    draft_tokenizer_path="Qwen/Qwen3-4B-Instruct-2507",
    cross_tokenizer=True,
    cross_tokenizer_mode="hybrid",
    cross_tokenizer_artifact_path=(
        "artifacts/cross_tokenizer/qwen3-4b_to_llama3.1-70b.json"
    ),
    n_particles=8,
    gamma=8,
    target_temperature=0.7,
    draft_temperature=0.9,
    resample_threshold=0.5,
    final_selection="posterior_sample",
    tp_size=2,
    trust_remote_code=True,
)
try:
    result = engine.generate(
        prompt="Solve: 17 * 23",
        sampling_params={"temperature": 0.7, "max_new_tokens": 512},
    )
finally:
    engine.shutdown()
```

Hardware-specific settings such as tensor parallelism, attention backend,
static memory fraction, and KV-token budget must be chosen for both checkpoints
and the GPUs. The target and draft currently share the engine's TP topology.

Validated GSM8K evaluation example:

```bash
SMC_CROSS_ONLINE_NGRAM=0 python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --draft-model /models/qwen3-4b-seqkd-merged \
  --draft-tokenizer-path Qwen/Qwen3-4B-Instruct-2507 \
  --prompt-tokenizer meta-llama/Llama-3.1-70B-Instruct \
  --cross-tokenizer \
  --cross-tokenizer-mode hybrid \
  --cross-tokenizer-artifact-path \
    artifacts/cross_tokenizer/qwen3-4b_to_llama3.1-70b.json \
  --particles 8 \
  --gamma 8 \
  --target-temperature 0.7 \
  --draft-temperature 0.9 \
  --resample-threshold 0.5 \
  --final-selection posterior_sample \
  --num-questions 400
```

This benchmark command renders prompts with the target tokenizer's chat
template. The retained SeqKD trainer renders training conversations with the
draft tokenizer's chat template. That is the measured historical contract, not
a general recommendation. For a new pair, choose the canonical serialization
deliberately—often the draft template because that is what the trainer uses—
and first confirm that the target remains coherent on those exact prompt bytes.
Treat alternative prompt templates as a matched ablation.

Freeze the exact evaluated model/tokenizer pairing before a promoted run:

```bash
python scripts/qwen3_smcsd_freeze_contract.py \
  --target-model meta-llama/Llama-3.1-70B-Instruct \
  --target-tokenizer meta-llama/Llama-3.1-70B-Instruct \
  --draft-model /models/qwen3-4b-seqkd-merged \
  --draft-base-model Qwen/Qwen3-4B-Instruct-2507 \
  --draft-tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --artifact artifacts/cross_tokenizer/qwen3-4b_to_llama3.1-70b.json \
  --output artifacts/contracts/qwen3-4b_llama70b.json
```

For a local merged draft, the freezer requires the base model, rejects an
unmerged adapter or incomplete shard index, checks architecture and tokenizer
lineage, and hashes every local weight shard. This can take time, but makes the
contract identify the actual checkpoint rather than only its directory name.

Use `live` mode as a correctness reference when validating a new tokenizer
pair. Do not enable experimental `SMC_CROSS_*` switches without repeating the
mapper, state-lineage, and end-to-end gates.

## Train the draft with target SeqKD

Target SeqKD is the portable training principle; the retained data-generation
and benchmark scripts are worked examples, not universal task adapters.

For a new model/task combination:

- replace the teacher generator with one that can call the selected target and
  preserve its exact model revision and sampling provenance;
- implement a task-appropriate verifier, or explicitly label unverified rows;
- split by source-prompt identity before creating multiple trajectories;
- ensure training uses the same canonical prompt serialization as draft
  inference;
- inspect the draft architecture's LoRA module names;
- provide a standalone evaluator and an end-to-end SMC evaluator for the task.

`qwen_lora_train_distill.py` currently targets the common
`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` layout used by the
validated Qwen model and many Llama-like models. It is not automatic for
architectures using names such as fused `c_attn`, MoE expert projections, or
other custom blocks. Such models require trainer support for their target
modules before using this recipe.

The trainer also requires a prefix-stable chat template: rendering the prompt
alone must produce the exact token prefix of the rendered prompt plus assistant
response. It fails rather than guessing a response-label boundary when that
invariant does not hold.

### Core idea

Generate response **text** from the target, then retokenize that text with the
draft tokenizer:

```text
target response text
  → render with the draft chat template
  → tokenize with the draft tokenizer
  → response-only cross-entropy
  → target-like draft policy
```

Do not align target token IDs to draft token IDs during SeqKD. The target
tokenizer is irrelevant to the training labels once target response text has
been produced.

SeqKD helps finite-particle SMC because target-like trajectories must first be
proposed by the draft before target weighting can select them.

### Teacher-data schema

The trainer accepts either `sft_messages`:

```json
{
  "sft_messages": [
    {"role": "user", "content": "Solve the problem..."},
    {"role": "assistant", "content": "Target-generated solution..."}
  ]
}
```

or `messages` plus `teacher_text`:

```json
{
  "messages": [{"role": "user", "content": "Solve the problem..."}],
  "teacher_text": "Target-generated solution..."
}
```

Recommended data rules:

1. Freeze teacher model/tokenizer revisions, chat template, sampling settings,
   stop conditions, and prompt-source revision.
2. Store raw messages and target response text with stable prompt IDs.
3. Verify task correctness when a reliable verifier exists.
4. Deduplicate responses.
5. Split by prompt identity before expanding multiple teacher trajectories.
6. Keep training, intrinsic-development, end-to-end-development, and protected
   final-test prompts disjoint.
7. Never train on benchmark test prompts.

For the validated GSM8K run, the provenance builder produced 25,008 training
responses from 5,881 GSM8K-train questions. It explicitly refused test-split
examples.

### Worked example: generate verified GSM8K teacher paths

For GSM8K, each input row must contain a user message and `gold_answer`.
Generate multiple target paths, retain only verifier-correct distinct
responses, and record the teacher sampling provenance:

```bash
python scripts/gen_multipath_target.py \
  --prompts /data/prompts/gsm8k_train.jsonl \
  --output /data/teacher/gsm8k_llama70b.jsonl \
  --target-model meta-llama/Llama-3.1-70B-Instruct \
  --tp-size 2 \
  --attention-backend torch_native \
  --n 8 \
  --max-per-problem 6 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-new-tokens 1024
```

`torch_native` runs eagerly in this script because it was the coherent
Llama-70B reference during validation. If another backend is selected, smoke
test its generated text before launching the full corpus.

### Worked example: build the validated GSM8K splits

`qwen3_build_seqkd_dataset.py` expects correctness-filtered teacher rows with
the exact GSM8K prompt/final-answer contract:

```bash
python scripts/qwen3_build_seqkd_dataset.py \
  --input /data/teacher/gsm8k_llama70b.jsonl \
  --output-dir /data/seqkd/qwen3_llama70b
```

The output manifest records source hashes and question-disjoint train,
development, intrinsic, and end-to-end-development splits.

### Worked example: train Qwen3 rank-64 LoRA

```bash
python scripts/qwen_lora_train_distill.py \
  --train-jsonl /data/seqkd/qwen3_llama70b/seqkd_train.jsonl \
  --eval-jsonl /data/seqkd/qwen3_llama70b/seqkd_dev.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir /models/qwen3-4b-seqkd-r64-adapter \
  --merged-output-dir /models/qwen3-4b-seqkd-merged \
  --lora-r 64 \
  --lora-alpha 128 \
  --lora-dropout 0 \
  --dtype bfloat16 \
  --learning-rate 1e-5 \
  --epochs 2 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0
```

The trainer:

- freezes the base model;
- adapts all attention and MLP projections;
- applies the exact draft chat template;
- masks prompt and padding tokens;
- trains only on assistant-response tokens;
- uses gradient checkpointing;
- saves the compact adapter and, when requested, a merged checkpoint.

### Evaluate before promotion

Validated Qwen/GSM8K standalone development command:

```bash
python scripts/qwen3_eval_seqkd_standalone.py \
  --model /models/qwen3-4b-seqkd-merged \
  --input /data/seqkd/qwen3_llama70b/seqkd_end_to_end_development.jsonl \
  --output /tmp/qwen3_seqkd_standalone.jsonl
```

Then compare base and SeqKD drafts under identical SMC prompts, seeds, target
budget, N, gamma, temperatures, mapper artifact, and terminal policy. Report:

- task accuracy and invalid-output rate;
- throughput and latency;
- target top-1/canonical score on fresh draft samples;
- normalized ESS, maximum particle weight, and log-weight range;
- mapper fallbacks, overlength rows, and non-finite weights.

The validated Qwen3 result was 354/372 standalone for SeqKD versus 352/372 for
the base draft. No tested post-SeqKD objective passed the live self-sampling
promotion gate, so the rank-64 SeqKD adapter remained the supported choice.

## New model-pair onboarding checklist

Treat each target/draft/tokenizer combination as a new system. A model family
name alone is insufficient because tokenizer revisions, added tokens, chat
templates, and stop IDs can change independently.

### 1. Freeze identities

- Record target and draft model IDs, revisions, and weight hashes where
  practical.
- Record both tokenizer IDs, revisions, chat templates, and fingerprints.
- Record SGLang, Transformers, PEFT, CUDA, and GPU versions.
- Record context lengths, dtype, tensor parallelism, and memory settings.

### 2. Validate prompt and tokenizer behavior

- Choose one canonical serialized prompt string.
- Encode that exact string with both tokenizers.
- Confirm each model produces coherent standalone output from it.
- Test empty strings, leading/trailing spaces, newlines, Unicode, numbers,
  code, all configured stop strings, EOS, and maximum-context boundaries.
- Confirm decoded generated text round-trips without silent cleanup or Unicode
  replacement.

If either model needs a mutually incompatible prompt serialization, the current
single-string `SMCEngine` interface is not sufficient for that pair.

### 3. Validate live mapping

- Start with a small prompt set and `cross_tokenizer_mode="live"`.
- Verify decoded target bytes equal decoded draft bytes for every mapped block.
- Verify each segment conserves draft proposal charge.
- Verify stop events terminate identically.
- Verify no non-finite weights or impossible sequence lengths occur.

Do not build conclusions from task accuracy until these invariants hold.

### 4. Validate hybrid mapping

- Start with the corpus-independent single-token artifact; add frozen
  domain-corpus n-grams only after it passes.
- Confirm tokenizer fingerprints match at load time.
- Compare live and hybrid modes on identical prompts and sampled draft blocks.
- Require byte equality and proposal-charge conservation.
- Review fallback, boundary-healing, overlength, carry, and special-token
  counts.
- Exercise one-to-many, many-to-one, unseen-token fallback, whitespace,
  Unicode/byte-fallback, EOS, short-proxy, overlength, and target-bonus
  expansion scenarios.

Keep live mode available as the reference implementation for regressions.

### 5. Establish matched baselines

Measure, on the same prompt IDs and seeds:

- target-only generation;
- standalone base draft;
- base-draft cross-tokenizer SMC in live mode;
- base-draft cross-tokenizer SMC in hybrid mode.

Freeze prompt serialization, output budget, particles, gamma, temperatures,
resampling threshold, terminal policy, runtime flags, and hardware placement.

### 6. Build task-specific SeqKD data

- Generate target response text from training prompts only.
- Preserve prompt IDs and teacher-generation provenance.
- Apply a reliable task verifier when available.
- Deduplicate trajectories within each prompt.
- Split by prompt identity before trajectory expansion.
- Convert accepted rows to `sft_messages` or `messages` plus `teacher_text`.
- Check that the draft chat/prompt serialization used by the trainer matches
  the runtime draft input contract.

The retained `gen_multipath_target.py` and
`qwen3_build_seqkd_dataset.py` implement this stage only for GSM8K.

### 7. Configure and train LoRA

- Confirm the draft is loadable through `AutoModelForCausalLM`.
- Identify architecture-appropriate LoRA target modules.
- Start from conservative rank, learning rate, and epoch count; rank 64,
  alpha 128, and two epochs are validated only for the Qwen3 worked example.
- Train with response-only labels and inspect the number of trainable
  parameters.
- Save the base model revision, adapter config, tokenizer, and training
  manifest with the adapter.

### 8. Apply promotion gates

Compare the base and trained drafts on fresh, self-sampled trajectories:

- standalone task quality must not materially regress;
- target agreement should improve on fresh draft samples;
- invalid-output and truncation rates must not increase;
- mapper invariants must remain exact;
- fixed-budget SMC quality should improve or remain within a declared
  non-inferiority bound;
- throughput and latency must be measured under identical runtime conditions;
- conclusions should hold across multiple seeds and more than one task domain
  before claiming general improvement.

Reject a candidate that improves only stale-trace loss or recorded importance
weights without improving fresh proposals or fixed-budget SMC behavior.

## What was deliberately not retained

The experiments tested BPM forward-KL/TV, clipped chi-square, depth-coupled
objectives, IW-MLE/Rényi-2, listwise outcome training, tail-only
specialization, adapter interpolation, and mapper/runtime sweeps. None
demonstrated a reliable improvement over SeqKD under fresh self-sampling and
fixed-budget SMC gates.

The recurring failure mode was denominator-only improvement on stale traces:
reducing current draft probability on old sampled paths increases recorded
`target_logp - draft_logq` without making newly sampled trajectories more
target-like. A lower trace loss or higher stale-trace weight is therefore not a
promotion criterion.

## Maintained implementation

Model-pair infrastructure:

- runtime entry point: `smcsd/engine.py`
- SMC integration: `smcsd/core/worker.py`, `scheduler.py`, `req_state.py`
- mapper: `smcsd/cross_tokenizer/`
- artifact builder: `python -m smcsd.cross_tokenizer.build`
- mapper audit: `scripts/audit_cross_tokenizer_mapper_modes.py`

Validated Qwen/GSM8K training example:

- end-to-end evaluation: `scripts/accuracy_test_gsm8k.py`
- verified teacher generation: `scripts/gen_multipath_target.py`
- SeqKD data validation: `smcsd/qwen3_seqkd.py`
- GSM8K SeqKD builder: `scripts/qwen3_build_seqkd_dataset.py`
- LoRA trainer: `scripts/qwen_lora_train_distill.py`
- standalone gate: `scripts/qwen3_eval_seqkd_standalone.py`
- contract freezer: `scripts/qwen3_smcsd_freeze_contract.py`
- adapter merger: `scripts/merge_lora_adapter.py`

Generated datasets, mapping JSON, model weights, checkpoints, logs, and dated
experiment directories are intentionally ignored by Git.
