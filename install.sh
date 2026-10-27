#!/usr/bin/env bash
# Companion installer. Idempotent-ish; prints the manual steps at the end.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${COMPL_HOME:-$HOME}"
DEST="$ROOT/companion"

echo "== staging into $DEST"
mkdir -p "$DEST"/{config,data,logs,tools,pipe}
cp -n housekeeping.py "$DEST/"
cp -n pipe/companion_pipe.py "$DEST/pipe/"
cp -n tools/*.py "$DEST/tools/"
[ -f "$DEST/config/companion.yaml" ] || cp config/companion.example.yaml "$DEST/config/companion.yaml"
cp -n config/compaction-prompt.txt "$DEST/config/"
[ -f "$DEST/data/profile.md" ] || cp config/profile.example.md "$DEST/data/profile.md"
cp -n config/prompt.md "$DEST/data/prompt.md"

echo "== venv + deps"
[ -d "$DEST/.venv" ] || python3 -m venv "$DEST/.venv"
"$DEST/.venv/bin/pip" install -q -U sentence-transformers PyYAML

echo "== embedding model (bge-small-en-v1.5 -> data/hfcache)"
HF_HOME="$DEST/data/hfcache" "$DEST/.venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-small-en-v1.5")
PY

echo
echo "==================== MANUAL STEPS ===================="
cat <<'STEPS'
1. Edit $DEST/config/companion.yaml  (open_webui_token, paths, alias)
2. Summarizer: put your compact gguf at $DEST/data/models/compactor.gguf
   (recommended: schneewolflabs/Compactor-Qwen3.5-4B, Q4_K_M quant),
   copy systemd/companion-summarizer.service to
   ~/.config/systemd/user/, edit paths, then:
     systemctl --user daemon-reload && systemctl --user enable --now companion-summarizer
3. Housekeeping: same with systemd/companion-housekeeping.service.
4. Open WebUI: Admin > Functions > Add Function, paste the FULL contents of
   $DEST/pipe/companion_pipe.py, save, ensure it is Active and
   "Control chat history" is on. (The pipe lives in OWU's webui.db, not on
   disk — if OWU is rebuilt/reinstalled, re-paste it.)
5. OWU admin > Settings > Chat > Context Compaction: enable, and set the
   prompt template to the contents of $DEST/config/compaction-prompt.txt.
   token_cap ~65536, retention ~40%. Verify it produced a real summary
   after a long chat (the `ctx` tool shows the live stack).
6. Optional CLI viewers:
     ln -s $DEST/tools/context.py ~/.local/bin/ctx
     ln -s $DEST/tools/summaries.py ~/.local/bin/summaries
STEPS
