# Companion — a local memory system for Open WebUI

Persistent, private memory for a self-hosted Open WebUI + llama.cpp stack.
No cloud, no accounts. One SQLite file is the whole database.

This is a working personal system, not a maintained product. It runs on one
machine against one Open WebUI version, and it is not going to grow a test
suite. Fork it, break it, take what's useful — the architecture notes in
`docs/ARCHITECTURE.md` are probably the most portable part.

Your chat model stays dumb as a post about the past; a small pinned
summarizer model quietly compacts it into a searchable index; a pipe
function re-injects the right pieces at every turn.

## How it works

Four moving parts:

1. **The pipe** (`pipe/companion_pipe.py`) — an Open WebUI *Pipe* function,
   registered in OWU's function table. On every request it:
   - injects `data/profile.md` (always-on self profile) and
     `data/prompt.md` (honesty rules) as system blocks;
   - retrieves the most relevant recent memory rows from the index and
     injects them as a `Remember` block;
   - records nothing itself — writes are the housekeeper's job.

2. **Housekeeping** (`housekeeping.py`) — a systemd user service. Idle-gated:
   when the chat goes quiet, it takes unindexed messages, asks the
   summarizer to turn each into a labelled memory line, embeds it, and
   stores it in `index_entries` inside `data/record.sqlite`.

3. **The summarizer** (`systemd/companion-summarizer.service`) — a second,
   small llama-server, CPU-only, pinned to one port and one model alias.
   It costs zero VRAM. Hard-fails by design when down; never silently
   falls back to another model.

4. **Context compaction** (OWU built-in, tuned via `config/compaction-prompt.txt`)
   — when a chat exceeds the token budget, OWU folds old turns into a
   rolling state snapshot. The pipe passes the previous snapshot forward.

Retrieval is deliberately boring: recency-gated window (7 days, 90 when you
say a trigger phrase like "think back"), 70% relevance (half cosine, half
keyword overlap) + 30% salience. No decay curves.

See `docs/ARCHITECTURE.md` for the design history and the failure modes
each guard exists to prevent.

## Requirements

- Linux, systemd user sessions
- Open WebUI (pip or container) with API access
- llama.cpp (`llama-server`) for the chat model **and** a second instance
  for the summarizer (any small instruct model works; ~4B is the sweet
  spot on CPU). What we actually run:
  [`schneewolflabs/Compactor-Qwen3.5-4B`](https://huggingface.co/schneewolflabs/Compactor-Qwen3.5-4B),
  Q4_K\_M GGUF — fine-tuned for this compaction task, which is why it
  follows the labelled-line format reliably where general chat models
  did not.
- Python 3.11+

## Install

    ./install.sh

It creates a venv, installs deps, downloads the embedding model, and prints
the manual steps (OWU pipe registration, summarizer model path, systemd
units). The manual steps are intentionally manual: OWU's function API
varies by version, and a DR-ready setup should be verifiable by eye.

Layout after install:

    ~/companion/
      housekeeping.py
      pipe/companion_pipe.py     <- paste into OWU Admin > Functions
      tools/                     <- read-only CLI viewers (ctx, summaries)
      config/companion.yaml      <- from config/companion.example.yaml
      config/compaction-prompt.txt
      data/record.sqlite         <- created empty; this is your memory
      data/profile.md            <- from config/profile.example.md
      data/prompt.md             <- honesty rules, edit to taste
      data/hfcache/              <- embedding model (auto-downloaded)
      logs/events.log

## What actually needs backing up

`data/record.sqlite` + `config/companion.yaml` + `data/profile.md`.
Everything else is code and re-downloadable models.

## Gotchas learned the hard way

- OWU stores pipes in its own `webui.db`, not on disk — back up OWU too,
  or keep the pipe file and re-register after a restore.
- `model_path` in the yaml must be the llama-server `--alias` string, not
  a `.gguf` path. A path can never validate against `GET /models`, and the
  failure is silent.
- Thinking-capable summarizer models emit a reasoning trace instead of the
  labelled memory line unless thinking is off (`summarize_thinking: false`).
- The compaction template must include the message placeholders; OWU's
  defaults changed between versions. Verify with a packet capture or the
  `ctx` viewer, not by trusting the UI.
- Tool schemas in OWU requests can dwarf chat history; if you hit context
  limits, check tool payload size before blaming memory.

## License

MIT — see LICENSE.
