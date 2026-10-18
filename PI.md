# pi as the client: how it shapes load on this cluster

Client-side findings from the 2026-10-15 stall/speed investigation. The server turned out to be fine (no regression,
no memory pressure, no NCCL problem); the load shape that produced the stalls is produced by the **client**, and most
of it is tunable there without touching vLLM. Server-side detail: `QWEN.md` ("The stall, caught end to end",
"When does KV actually get flushed?", "Speed re-check"). Tooling: `scripts/prefix_probe.py`, `scripts/watch_stalls.sh`.

## What pi puts on the wire (verified against pi 1.2.0 source, this machine)

| Behaviour | Where | Consequence for this cluster |
|---|---|---|
| Streams every request; sends `temperature` only if configured | provider layer | Server-side sampling defaults apply; for MiMo this is a storm mitigation, see `MIMO.md` |
| Injects project instructions: `<agent-dir>/AGENTS.md` first, then walks **up from cwd** collecting `AGENTS.override.md`/`AGENTS.md`/`AGENTS.MD`/`CLAUDE.md`/`CLAUDE.MD`, nearest wins | `core/resource-loader.js:165` | Every session carries its own instructions; there is no shared file unless a user-level or ancestor `AGENTS.md` exists. On this box neither exists, so a session rooted in `~/spark-tuning` injects exactly one file |
| Skills, system-prompt overrides and the tool list are also in the system prompt | `core/agent-session.js:1272` | Changing tools/skills/settings mid-session rewrites the prompt prefix → full prefix-cache miss |
| Context files are read inside `resourceLoader.reload()`; the only call site is the explicit settings/extension reload | `resource-loader.js:468`, `agent-session.js:2980` | **Mid-session file edits change nothing live.** Verified empirically: a session running since 10-07 still carried the 10-07 `AGENTS.md` after several same-day edits. The re-read happens at session start/resume |
| Compaction triggers when `contextTokens > contextWindow − reserveTokens` | `core/compaction/compaction.js:173` | With the 262,144-token model and pi's defaults (`reserveTokens` 16,384) sessions ride to ~246K tokens before compacting |

## Why that shape hurts here

pi re-sends the entire conversation on every turn. That is free when the prefix cache hits and brutal when it misses:

| Path | Measured cost |
|---|---|
| Warm hit, steady state (any size) | ~2,500–3,000 tokens recomputed, roughly **constant** — 2,995 of 7,939 at 8K, 2,535 of 23,959 at 24K, 2,623 of 96,559 at 96K (`results/prefix-probe-tail-*-2026-10-15.json`). ~0.8 s regardless of context |
| Cold miss | 182,012 tokens = **50 s** of exclusive prefill at 3,640 tok/s; captured with a second request blocked on scheduler capacity and every other stream at zero tokens (`results/stall-watch/stall-2026-10-15T1411.csv`) |
| Concurrency tax, thinking on, T=0.6 | 1 stream 52.8 tok/s → 3 streams 32.0 → 4 streams 26.9–29.3 at 1K, 22.7–26.1 at 32K (aggregate 98–112) |

Actual traffic profile since the 10-13 launch: of 1,601 requests, 386 had 100K–200K prompt tokens and 97 exceeded
200K — about a third of turns carry >100K tokens, with a cumulative prefix-cache hit rate of 95.3%. Big contexts do
**not** slow the hit path; they only scale the cost of a miss from annoying to catastrophic.

## What actually diverges a prefix

Anything that changes the prompt from token 0: **session start or resume** (context files are re-read then, so an
edited `AGENTS.md` costs exactly one full re-crunch at the next session boundary), model change, thinking-level
change, tool/skill set changes, `/reload` or settings changes, and compaction (which rewrites history). What does
**not** diverge it, all measured: idle time (89.4% of a 24K prompt still served after 900 s of nothing), competing
traffic from other agents, and editing the instruction files mid-session.

One nuance found while banking the raw data: the **first** repeat after a cold prefill recomputes more than steady
state — 9,215 tokens on a 96K prompt on the first warm pass, 2,623 on the next. So a turn immediately after a miss
costs a little extra; from the turn after that, the tail is flat. Earlier drafts of this doc quoted the 9,215 figure
as the steady-state cost; corrected 2026-10-15.

## What changed when nvidia FP4-Local became the live model (2026-10-16)

The measurements above are **config C** (the deployment that was live during the investigation). The cluster now serves
FP4-Local K=2 (`FP4LOCAL.md`), which changes the balance for everything in this doc:

| | config C | FP4-Local (live) |
|---|---|---|
| warm hit, 96K prompt | 2,623 tokens recomputed, 3.92 s | **0 recomputed, 0.35 s** (independently re-measured, `results/fp4local/fp4local-k2-mycheck-prefix-96000-2026-10-16.json`) |
| cold prefill, 96K prompt | 26.5 s (3,640 tok/s) | 36.2 s (2,668 tok/s) — **~36% slower per token** |
| recurring cause of the stalls | every hit re-crunched ~2.5-3K tokens; divergence re-crunched all 182K | hits are free; only genuine divergence costs |

Practical consequences for the client, in order:
- The dominant stall driver is gone: the recurring re-crunch was the mamba state granularity that FP4-Local doesn't have.
- **Misses got more expensive, not cheaper.** A 180K-token cold prompt now monopolises the engine for ~68 s rather than
  ~50 s. So "don't create a giant cold prompt" matters *more* on the live model: don't resume a huge session across a
  restart, don't change model/thinking/tools on a long session, and keep the compaction staging below.
- The staging is still the right client change: it caps how large a history (and therefore a miss) can get, and
  compaction itself rewrites history, so it should happen at a natural break rather than at 246K.

## Tuning the client (no server restart needed)

Staged on 2026-10-15 in `~/.pi/agent/settings.json` (backup `settings.json.bak-2026-10-15`):

```json
"compaction": { "enabled": true, "reserveTokens": 65536, "keepRecentTokens": 20000 }
```

Effect: compaction now fires above **196,608** context tokens instead of 245,760, capping the worst-case prompt (and
therefore the worst-case miss) by ~20%, and `keepRecentTokens` 30,000 → 20,000 keeps post-compaction prompts smaller.
Revert by restoring the backup, or with `scripts/stage_pi_compaction.py --revert`. That script also does `--check`
and a fresh `--reserve N` staging (idempotent, verifies after writing, refuses a file it cannot parse; apply/revert/
idempotency/refusal exercised on a copy 2026-10-15). The values above were first applied by hand, so the file and the
script agree. Deliberately not tighter: too-small sessions lose work context, and the warm-hit cost is flat in prompt
size, so this is about tail risk only.

Free behaviour changes, in rough order of value:
- Don't change model, thinking level or tool set on a long session — each is a full miss at that size.
- Prefer a fresh session over resuming a giant one across a restart; a resume re-reads and rebuilds the prompt anyway.
- Turn thinking off where reasoning isn't needed: +20% decode (63.9 vs 52.8 tok/s) and better MTP acceptance (2.60 vs 2.05).
- Batch instruction-file edits and do them at a session boundary, since the next resume pays one re-crunch.
- Keep this host free of other heavy disk/IO jobs while decoding — the KV reuse that makes a warm turn ~0.8 s is unified memory.

## Why the status-line token speed reads ~half the benchmark

The user's status line comes from the **`pi-token-speed`** package (`packages: ["npm:pi-token-speed"]` in
`~/.pi/agent/settings.json`). It is a UI element, not a measurement, and three things in it under-report, biggest first:

1. **It counts one token per streamed delta.** `recordDelta()` calls `recordTokens(1)` in the default `direct` strategy
   (`src/core/engine.ts`). But with MTP the server hands the client a whole *accepted draft block* per SSE chunk, so a
   chunk carries ~the mean acceptance length, not one token. Measured on the live server: **0.38-0.55 chunks per
   completion token** (1.8-2.6 tokens/chunk). The display therefore reads out roughly the engine's **step rate**
   (~25/s), not its token rate - which is why it looks stuck around 24-25 no matter what the benchmark says.
2. **Tool-call argument deltas are only counted for `edit` and `write`** (`TOKEN_GENERATION_TOOLS`,
   `src/config/constants.ts:9`). `bash`, `read` and MCP tool arguments stream by uncounted while the clock keeps
   running - a serious deflator for agent work, where much of the output is tool arguments.
3. **The clock starts at stream start**, so prefill/TTFT sits in the denominator. Negligible on warm hits (0.03-0.28 s
   measured), but a cold 50K prompt (~19 s at the live model's 2,668 tok/s) would swamp it.

Measured three ways on the same requests (true = `usage.completion_tokens` / decode wall time; the other two are this
extension's own arithmetic, replicated from its handlers):

| stream | true tok/s | display, default `direct` | display, `estimate` |
|---|---|---|---|
| plain text, template default (thinking on) | 33.4 | 18.4 | 33.3 |
| `bash` tool-call arguments | 52.6 | 17.8 | 48.9 |
| `edit` tool-call arguments | 38.6 | 14.1 | 34.7 |

Not the cause, despite being the obvious guess: **thinking tokens are counted** - the delta handlers cover
`thinking_delta` as well as `text_delta` (`src/core/message-handlers.ts:33`), and this build streams reasoning as
`delta.reasoning`.

Fix for the live number: **staged 2026-10-16** — `"tokenSpeed": { "countStrategy": "estimate" }` is now in
`~/.pi/agent/settings.json` (backup `settings.json.bak-*-tokspeed` beside it). It tracks the true rate within ~8% on all
three rows above. Notes for whoever revisits this:

- The value must be the **key** (`"estimate"`), not the UI label; `ScalarSettingsItem.validate()` checks
  `Object.keys(labels)` and silently substitutes the default (`direct`) for anything else.
- It is read in `settings.initialize()`, which runs from the extension's `session_start` handler — so it applies at the
  **next session start**, no need to restart pi just for it. Invalid values surface as a
  `[pi-token-speed] Invalid countStrategy …` notification, which is the fastest confirmation that it took.
- `/tps` opens the extension's own settings menu (and writes the same block).
- `useProviderTokens: true` does **not** help the streaming figure: pi does send `stream_options.include_usage`, but
  vLLM emits usage only in the **final** chunk, so it can only snap the total at the end - which the extension already
  does via `reconcileTotal()` on message end.

Corollary worth remembering: because the divisor is the acceptance length, **better speculative acceptance makes the
display read worse** while the engine actually gets faster. For any speed question use
`scripts/bench_decode.py`, which divides the server's own `usage.completion_tokens` by wall time.

## Why restarting the client re-crunches the whole prompt, and what it costs

The KV cache does **not** expire: 900 s of idle and three competing prompts of the same size class both left a cached
prompt fully served (`results/prefix-probe-idle-1007.json`, `results/prefix-probe-2026-10-15.json`). The question is
never "did it expire" but "did the prompt change, and where". The cache is **content-addressed from token 0**: a block
is reusable only if every token before it is identical, because in causal attention the K/V of position *i* is a
function of all preceding tokens (and the Gated DeltaNet state encodes the exact sequence it was computed from). A
change near the front does not invalidate a little - it makes every cached entry after it *wrong*, so there is nothing
left to reuse. Measured on the live server, one prompt of ~30K tokens, one token added (`scripts/prefix_probe.py
--steps cold,warm,edit_early,edit_late`, new steps added for exactly this; `results/prefix-probe-edit-position-30000-2026-10-16.json`):

| step | recomputed | cached | end to end |
|---|---|---|---|
| cold (first sight) | 29,959 | 0 | 11.08 s |
| identical resend | **0** | 29,959 | **0.24 s** |
| **1 token changed at the START** | **29,960** | 0 | **11.04 s** |
| 1 token changed at the END | 5,386 | 24,576 | 2.28 s |

**pi's system prompt is stable across restarts.** Read out of a real session file, it is assembled from fixed sections -
`preamble` (169 chars), `tools` (339), `rules` (839), `docs` (1406), `addendum` (8,906), `cwd` (36), `project_context`
(16,950) - with no date, no session id and no UUID anywhere in it. So a restart is normally a perfect hit, and pays a
full prefill only when one of those sections changes: the injected `AGENTS.md`/`CLAUDE.md` (re-read at session start),
the tool/skill set, the model, or the cwd. **Which is what happened today**: several commits edited this repo's
`AGENTS.md` while sessions were live, so the first restart of each session after those edits re-read the file, diverged
near token 0, and re-crunched the entire history. That cost is ours, not the cluster's.

**The 4,096 floor.** An *identical* prompt resumes at the exact end (0 recompute) because vLLM publishes a partial tail
entry for it. A *modified* prompt does not get that treatment: its hit floors to a 4,096-token boundary, so an append
costs `(prompt mod 4096) + 4096` tokens - exact at both sizes measured (29,962 -> 1,290+4,096 = 5,386; 120,962 ->
2,178+4,096 = 6,274). That is the per-turn cost of pi appending a turn: ~6K tokens on average, ~2.3 s at the live
prefill rate. Why 4,096 is **untested**; it equals `--max-num-batched-tokens`, which makes it a candidate lever, nothing more.

**Why a resume feels slower than it used to.** The live build prefills ~27% slower per token than config C (measured
today: 11,959 tokens in 4.49 s = 2,664 tok/s, and 96,559 in 36.2 s = 2,668; config C was 3,640). A 180K-token resume is
therefore ~68 s now against ~50 s on config C. Separately, a **server** restart wipes the KV cache outright - it lives
in GPU memory - so the FP4-Local swap on 2026-10-16 made every session's next resume cold regardless of client.

**Measured end to end on 2026-10-18 with real pi resumes.** A copy of the pi config pointed pi at a logging proxy in
front of the live server, which saved every request body and its time to first token. The session first read
`FP4LOCAL.md` and `QWEN.md`, so the history was ~87K characters (`results/pi-resume-2026-10-18.tsv`):

| resume (`pi -c`) | TTFT | what the request body showed |
|---|---|---|
| nothing changed | **0.34 s** | all prior messages, tools and params byte-identical: warm |
| one line appended to `AGENTS.md` | **11.9 s** | first difference in message 0 (system) at char 22,751, the end of the injected `AGENTS.md`: cold |
| thinking level changed (`--model qwen/qwen3.8-flash-next:low`, default is medium) | **12.2 s** | `reasoning_effort` the only difference, messages identical: cold |
| back to the default level | **0.50 s** | warm again: the medium-rendered prefix was still cached |

So a restart is warm by default. The two measured causes of a cold resume are an edited context file and a **changed
thinking level**. The second one is the chat template: with thinking on, `low` and `xhigh` each write a different
"Reasoning effort is set to ..." sentence as the very first line of the system prompt, and `medium` writes none, so
switching levels diverges at about token 0. The template accepts only `low`, `medium` and `xhigh`. **`high` returns HTTP
400** ("Unexpected reasoning effort high"), which pi shows as "Connection error." TTFT here only separates cold from
warm; it is not a speed number (single run each, other traffic on the server not ruled out).

Mitigations, cheapest first: batch injected-file edits to session boundaries, since one edit is charged to every
session that restarts afterwards; keep one thinking level per session (switching back to an earlier level is warm only
while that prefix is still cached); keep injected files short (this repo's `AGENTS.md` is ~17K chars and sits *ahead of
the whole conversation*, so it is paid on every cold prefill); prefer continuing a session over restarting one; leave
`compaction.reserveTokens` at 65,536 so a resume's prompt has a bound.

Probe caveat learned the hard way: the probe's nonce is deterministic (`--seed 7`) so saved results can be re-measured,
but that also means two agents probing concurrently **share prefixes** - one of my "cold" rows came back 74.5% cached
because a parallel run had already cached the common filler. `--seed` now lets you isolate a measurement, and a cold
row that finds anything cached says so out loud.

## Provider config and model switching

The local pi config (outside this repo) has a `qwen` provider at `<head>:8000/v1`, plus `mimo` at
`<head>:8000/mimo/v1` (inert — MiMo is not installed) and a GLM entry when provisioned. Only the provider matching the
model currently served works; `mimo/up.sh` / `glm/up.sh GLM_VARIANT=E224` / `mimo/down.sh --qwen` change which one
answers. Qwen's template defaults to thinking **on**, which is the slower path above.

## Open / unverified

- The `reserveTokens` staging is applied and verified in the config file, but has not yet been observed firing in a
  real session — nobody has crossed 196K since it was set.
- Session-level traces of the concurrent agents live on another host, so per-session prompt growth was inferred from
  server-side histograms, not from their session logs.
- Resume behaviour is now measured end to end (2026-10-18, above) for no change, a context-file edit and a thinking-level
  change. Tool/skill/extension set, model, cwd and pi-version changes are still inferred, not measured.
- The FP4-Local figures in this doc: the warm-hit zero is my own re-measurement of what `FP4LOCAL.md` already
  reported (single run each); the 36.2 s / 2,668 tok/s cold number is mine, single run, 96K prompt. No repeat spread,
  and nothing else was on the server at the time.
