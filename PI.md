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
- Whether a *resume* costs a full re-crunch in practice is inferred from the code path plus the observed miss sizes,
  not measured end to end.
