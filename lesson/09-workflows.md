# 09 — Request-Processing Workflows (flow charts)

Side-by-side workflows of the three designs: SpecActor (#22520), SPECTRE
(#22272), and the SMCSD lockstep port (`smcsd/decoupled/`). All source-verified;
message names are exact. See [07](07-pr-landscape-and-roadmap.md) for the
axis-by-axis architecture comparison.

## 1. SpecActor (#22520) — autonomous drafter, streaming deltas

```
          VERIFIER (GPU0, target — authoritative)         DRAFTER (GPU1, draft model)
          ───────────────────────────────────────         ──────────────────────────────
client ─▶ prefill forward (extend)
          │
          ├─ DraftSync {prompt, committed_output} ──────▶ create mirror Req
          │                                               (max_new_tokens=1<<30,
          ▼                                                never self-finishes)
  ╔═ VERIFY LOOP — per decode step ════════════════╗  ╔═ DRAFT LOOP — free-running ═══╗
  ║                                                ║  ║                               ║
  ║ 1. snapshot tail from DraftTailBuffer          ║  ║ a. decode 1 token             ║
  ║    (allow_partial=True; drafter behind?        ║  ║ b. stream it ─────────────────╫─▶ async append to
  ║     → draft_buffer=None → plain 1-tok decode)  ║  ║    DraftTailStreamOutput      ║   DraftTailBuffer
  ║ 2. broadcast frozen snapshot to TP ranks       ║  ║    {base_committed_len,       ║   (proxy thread;
  ║ 3. build linear top-k=1 chain (EOS-pad + cut)  ║  ║     new_token_pos, token_id}  ║    stale-base /
  ║ 4. ONE target forward (TARGET_VERIFY)          ║  ║ c. ahead ≥ 2×window → SLEEP   ║    pending checks)
  ║ 5. EAGLE verify → longest prefix + 1 bonus     ║  ║    ahead < window   → WAKE    ║
  ║ 6. stream committed tokens ───▶ client         ║  ╚═══════════════╦═══════════════╝
  ║ 7. VerifyCommit {pre_verify_committed_len,     ║                  ▼
  ║    committed_token_ids} ───────────────────────╫─▶ apply commit segment:
  ╚════════════════════════════════════════════════╝   ├─ full match → advance
                                                        │  verifier_committed_prefix_len
                                                        │  (the ONLY hot path if no
                                                        │   rejections)
                                                        └─ MISMATCH → ROLLBACK:
                                                           truncate output_ids + KV
                                                           (token-granular), grammar
                                                           rollback, re-install bonus,
                                                           patch in-flight batch,
                                                           ECHO committed token ────▶
                                                           (drains verifier's
                                                            pending_expected queue)
finish/abort:
          ── DraftClose {reason} ───────────────────────▶ free mirror req + KV
```

Both loops run **concurrently and unsynchronized**; the `DraftTailBuffer` +
commit protocol re-establishes agreement ([03](03-draft-tail-buffer.md),
[05](05-drafter-runahead-and-rollback.md)).

## 2. SPECTRE (#22272) — target-driven rounds, snapshot protocol

```
          TARGET (binds Router)                           DRAFTER (dials in as Dealer;
          ─────────────────────                            shared server w/ own user traffic)
client ─▶ prefill (normal path)                           ──────────────────────────────────
          │
  ╔═ ROUND t — event_loop_normal_spectre_target ═══╗
  ║                                                ║
  ║ 0. adaptive gate: breaker OPEN / too many      ║
  ║    no-draft reqs / high overhead?              ║
  ║    → ntpb=1 plain decode, skip 1 & 3           ║
  ║                                                ║
  ║ 1. send DRAFT {rid, spec_cnt=t+1,              ║──▶ dedupe by spec_cnt, then
  ║    output_ids (FULL snapshot), cur_drafts}     ║    fork-point diff vs own tokens:
  ║                                                ║    ├ identical  → resume paused req
  ║ 2. verify forward on ROUND-t drafts            ║    ├ divergent  → local KV rollback
  ║    (EagleVerifyInput.verify → accept + bonus)  ║    │  (page_size==1) or full RE-PREFILL
  ║    ── drafter decodes round t+1 concurrently ──║    ▼
  ║                                                ║    decode exactly num_draft_tokens,
  ║ 3. torch.cuda.synchronize();                   ║    send response, then PAUSE
  ║    recv_draft_fn: wait ≤ 200 ms ◀──────────────║◀── DRAFT_RESPONSE {rid, spec_cnt=t+1,
  ║    take whatever arrived                       ║       draft_token_ids, draft_logprobs}
  ║                                                ║    (keeps KV + req_pool_idx while paused;
  ║ 4. _post_verify_update_drafts: fork-point      ║     overloaded → REJECT instead)
  ║    check committed vs (cur_drafts + new[0])    ║
  ║    ├ match → adopt as next cur_drafts          ║
  ║    └ stale → clear; sync retry if many failed  ║
  ╚════════════════════════════════════════════════╝
          round t+1 repeats…                              heartbeat ▲ every 1 s
finish/abort:                                             (3 s silence → unregistered,
          ── FINISH / ABORT {rid} ──────────────────────▶  breaker counts failures)
```

No commit/rollback messages exist — every round's full-snapshot `DRAFT` *is*
the commit; the drafter infers divergence by diffing.

## 3. SMCSD decoupled port (`smcsd/decoupled/`) — lockstep hybrid

```
          VERIFIER (GPU0: target 8B + SMC logic)          DRAFTER (GPU1: draft 1B + slot mirror)
          ──────────────────────────────────────          ─────────────────────────────────────
client ─▶ request enters DecoupledSMCScheduler
          │
          ├─ DraftPrefillReq {group_ids, input_ids,  ───▶ parent prefill forward (writes
          │     sampling}              (BLOCKING)          prompt draft-KV), sample x0
          │ ◀─ DraftPrefillResp {next_token_ids=x0} ────
          │  target prefill; materialize N particles
          ├─ DraftMaterializeGroup {group_id, slots, ───▶ alloc N slots; copy_block_table
          │     shared_seq_len}                            parent→particles (refcounted);
          ▼                                                release parent
  ╔═ LOCKSTEP ROUND — strictly sequential ══════════╗
  ║                                                 ║
  ║ 1. DraftStepReq {slots, verified_ids=bonus x0,  ║──▶ ASSERT seq_lens == mirror
  ║      seq_lens}                (BLOCKING)        ║    (fail-fast tripwire, replaces
  ║                                                 ║     upstream reconciliation)
  ║ 2. ◀─ DraftStepResp {tokens (bs,γ),             ║◀── AR loop γ+1 steps, all particle
  ║        logprobs (bs,γ)}                         ║    slots (step γ+1 only writes
  ║                                                 ║    x_γ's draft KV; sample discarded)
  ║ 3. target verify forward, γ+1 positions ×       ║
  ║    N particles — NO REJECTION: accept all γ,    ║
  ║    sample 1 bonus per particle                  ║
  ║ 4. SMC reweight (target vs draft logprobs)      ║
  ║ 5. resampling fired? ─ DraftCommitResample      ║──▶ batched_resample_kv on own pools
  ║      {dst_slots, src_slots}                     ║    (replay same plan; FIFO channel
  ╚═════════════════════════════════════════════════╝     ⇒ round t commit lands before
          next round (anchor = this round's bonus)         round t+1's step)
finish/EOS:
          ── DraftCloseGroup {group_id, slots} ─────────▶ free slots (idempotent)
```

## GPU occupancy over time (the one-glance comparison)

```
                    time ─────────────────────────────▶
SpecActor           GPU0 (target):  │VVVV│VVVV│VVVV│VVVV│     V = verify forward
(run-ahead)         GPU1 (draft):   │ddddddddddddddddddd│     d = draft decode
                                     drafter never stops; rollback repairs divergence

SPECTRE             GPU0 (target):  │VVVV│VVVV│VVVV│VVVV│
(1-round pipeline)  GPU1 (draft):     │dddd│dddd│dddd│        window t+1 drafted
                                     exactly one round of overlap                while verifying t

SMCSD port          GPU0 (target):  │    │VVVV│    │VVVV│
(lockstep, today)   GPU1 (draft):   │dddd│    │dddd│    │     strict alternation —
                                     no overlap yet; future = pipeline ACROSS groups
                                     (draft group B while verifying group A)
```

The last row is the roadmap in one picture: the lockstep port pays the
decoupling tax with zero overlap (each GPU idles ~half the time). The planned
speedup is SPECTRE-style **group pipelining** — fill GPU1's gaps with other
groups' draft rounds — since SMC's bonus-token anchor + per-round resampling
forbid SpecActor-style within-group run-ahead.

---
*Confidence:* High — SpecActor & SPECTRE flows from full diff reads at pinned
SHAs; SMCSD flow from `smcsd/decoupled/io_struct.py` + `tasks/decoupled_smc_design.md`.

## How overlap emerges in SpecActor's default mode (no pipeline controller)

Default = `SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL=true` (parallel mode). Client
requests only ever enter the verifier; the drafter's work is 100% verifier-created.
Overlap is **emergent from asynchrony**, regulated by three mechanisms:

```
                  ── time ──▶
VERIFIER (GPU0)  │■■■■ target verify fwd (t) ■■■■│■■■■ verify fwd (t+1) ■■■■│
                  ▲ snapshot: take whatever        ▲ snapshot again — buffer
                  │ tail is buffered NOW           │ refilled meanwhile
DraftTailBuffer ──┴────────────────────────────────┴────────────────────────
                   ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲             (tokens land async,
DRAFTER (GPU1)   │d│d│d│d│d│d│d│d│ …sleep if ≥2γ ahead… │d│d│d│d│
                          ◀── VerifyCommit(t) re-anchors the prefix
```

1. **The buffer is the clutch.** The transport thread appends draft tokens to the
   verifier's `DraftTailBuffer` *while* the verifier's GPU runs the big target
   forward (draft model is small → several tokens per verify forward). The next
   snapshot finds tokens already waiting — the verifier never blocks on drafting.
2. **`allow_partial` keeps the verifier non-blocking.** Snapshot takes whatever
   depth exists: full window → full verify; partial → shorter chain (EOS-pad +
   chain-cut); empty / drafter behind → plain 1-token decode for that request.
   A slow drafter degrades smoothly; it never stalls the target.
3. **The drafter speculates THROUGH the bonus token; rollback repairs.** Each
   round ends with a target-sampled bonus the drafter never produced. The drafter
   doesn't wait — it keeps greedy-decoding its own continuation. On commit:
   drafter's token at that position == bonus → full-match advance (nothing
   wasted); != bonus → rollback to divergence, install bonus, continue (in-flight
   stale tokens rejected by the `can_accept_prefix_len` floor). The 2×γ run-ahead
   window + sleep bound the waste of a wrong guess; wake resumes on catch-up.

Plus continuous batching on both engines: while request A sleeps or rolls back,
requests B/C/D keep the drafter GPU full.

Contrast of defaults: SPECTRE schedules overlap *explicitly*, exactly one round
deep (send DRAFT(t+1) before verify-forward(t), collect after, ≤200 ms wait).
The SMCSD port has **no overlap by default** (blocking `DraftStepReq`, GPUs
alternate) — deliberate, because SMC's bonus-anchor + per-round resampling make
speculating through the round boundary algorithmically invalid; future overlap
is cross-group pipelining.

## How overlap works in SPECTRE's default mode (explicit one-round pipeline)

Requests only enter the target; the drafter learns about a request lazily via
the first `DRAFT` message (`spec_cnt==0` carries full `input_ids` +
`sampling_params`). Overlap is **explicitly scheduled** — send-before-forward
inside the target's event loop:

```
              ── time ──▶
TARGET (GPU0)  │send DRAFT(t+1)│■■■■ verify fwd on drafts(t) ■■■■│sync+recv ≤200ms│fork-pt: adopt/clear│ …
                      └──────────────┐                                  ▲
DRAFTER (GPU1)                       ▼                                  │
               …PAUSED… │reconcile fork point; decode window(t+1)│──────┘ respond; PAUSE
                         (gap until DRAFT(t+2) — filled by drafter's OWN user traffic)
```

1. **Send-before-forward hides draft latency**: DRAFT(t+1) is dispatched before
   the round-t verify forward launches; if the window finishes within the
   forward, the post-forward wait (`SPECTRE_RECV_TIMEOUT_MS`=200 ms cap) is ~0.
2. **Round mailbox, not stream buffer**: a bg receive thread fills pre-registered
   `req_to_draft_token[(rid, spec_cnt)]` slots — one window per request per
   round; stale `spec_cnt`/epoch responses are dropped. (The DraftTailBuffer
   analog, but bounded to one round.)
3. **Chained speculation through the bonus**: drafter drafts from
   `output_ids + cur_drafts`, so `new_draft[0]` lands on the bonus position;
   full accept + bonus match → `new_draft[1:]` becomes next `cur_drafts`.
   Any rejection breaks the chain at the fork point — max waste = ONE window
   (vs SpecActor's up-to-2×γ). Drafter self-repairs on the next snapshot
   (local KV rollback or re-prefill); no rollback messages exist.
4. **Drafter GPU filled by multi-tenancy, not run-ahead**: between windows the
   drafter serves its own user traffic; `REJECT`s drafting under load.
5. **First-class degradation**: per-batch flip to plain decode (own ntpb=1
   graph); circuit breaker OPEN after 30 empty rounds → no draft requests for
   100 rounds → HALF_OPEN probe with full context. Target never depends on
   the drafter for progress.

Contrast: SpecActor = continuous push, opportunistic consume, 2×γ run-ahead,
commit/rollback repair. SPECTRE = one pulled window per round, send-early to
hide, snapshot-diff repair, spare drafter capacity → own tenants. The SMC
port's future cross-group pipelining should reuse SPECTRE's send-before-forward
loop shape applied across groups.
