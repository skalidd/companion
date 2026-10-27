# Architecture

Design decisions, in the order they hardened. Most of this exists because
something failed in production first.

## The jars

- **Jar 1 — raw record.** Every chat message, copied from OWU via its read
  API into `record.sqlite` (`messages` table). Immutable log.
- **Jar 2 — the index.** `index_entries` table in the same SQLite file:
  one labelled memory line + keywords + salience + a 384-d float32 vector
  BLOB per indexed message. One file, one backup.
- **Jar 3 — the self.** `data/profile.md`, curated by hand, injected
  verbatim on every turn.

Write path: OWU -> housekeeping -> summarizer -> Jar 2.
Read path: Jar 2 -> pipe -> OWU prompt. The pipe never writes; the
housekeeper never talks to OWU's chat.

## Why a pinned summarizer

Early builds reused whatever model was loaded as the compactor. Each model
answers the summarizer prompt differently, and a general-purpose chat model
will happily write chain-of-thought into memory rows. Result: hundreds of
junk index rows. The pin (one port, one alias, one small model, thinking
off) plus a hard failure when it's missing is deliberate: loud and
recoverable beats silent and polluted.

The model itself: `schneewolflabs/Compactor-Qwen3.5-4B` (Q4_K_M GGUF),
fine-tuned for context compaction. Benchmarked against the
served chat model on the same prompts: higher parse rate on the labelled
format, fewer output tokens, and CPU-only at zero VRAM versus the chat
model's GPU footprint. A task-tuned small model is not a downgrade here;
it is the point.

Keep it on CPU even if you have VRAM to spare. A Q4 4B is a few GB; on a
24 GB card that is a few GB your main model would rather spend on context, and
a 4B prompt pass of this size finishes fast enough on 8 threads that the
difference doesn't matter. VRAM is the constrained resource in nearly every
local stack; system RAM almost never is. Offloading the summarizer is the
wrong direction to optimize in.

`resolve_summarizer()` validates `model_path` against `GET /models`. That
comparison is why the yaml holds an *alias string*, not a `.gguf` path —
a path can never match, and the mismatch degrades silently.

## Retrieval (Brick 4)

No decay curve, no half-life. Recency gate (7-day window; the user's
trigger phrase widens it to 90 for one turn) + score:

    score = 0.7 * (0.5 * cosine + 0.5 * keyword overlap) + 0.3 * salience

Embeddings: bge-small-en-v1.5, local, cached per process. The weights
(0.7/0.3, 0.5/0.5) were tuned against retrieval probes, not theory.

## Compaction (OWU built-in, tuned)

OWU's context compaction folds old turns into a rolling snapshot. The
stock template broke on this stack in two ways, both fixed in
`config/compaction-prompt.txt`:
1. templates without real message placeholders produce empty summaries;
2. thinking-capable models leak their reasoning trace into the summary
   unless `enable_thinking: false` is set in the request.

Verify compaction against the wire (packet capture or the `ctx` viewer),
never against the UI.

## Honesty rules (Brick 5)

`data/prompt.md` bounds confabulation: the model may answer only from the
conversation in front of it or the Remember block; anything else is "I
don't have that in front of me." This is a prompt, not a hack — edit it to
your voice, keep the three-source enumeration.

## Cost reality

In measured requests, tool schemas can be ~77% of the outgoing prompt body
while retained history is small. If you hit context limits, profile the
request (Content-Length + a capture) before blaming the memory system.

## Failure modes and guards

| Failure | Guard |
|---|---|
| summarizer down -> silent fallback -> junk rows | pinned port/alias, hard raise, row stays `indexed=0` and retries |
| reasoning trace in summaries | `summarize_thinking: false` + `--jinja` |
| empty compaction summary | template with placeholders, verified on the wire |
| profile/prompt drift between disk and OWU function | pipe reads files at request time; one deployed copy, hashed |
| DR amnesia (pipe lost with OWU reinstall) | pipe kept as a file; README says re-register |
