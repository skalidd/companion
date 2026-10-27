# companion_pipe.py  —  Pipe Function (Open WebUI "functions" API) for the
# Personal AI Companion (v2).
#
# Brick 1 — "The Record" (Jar 1): diff incoming messages vs record.sqlite,
#   INSERT OR IGNORE any whose id is new. No AI. Just a write. Lossless.
# Brick 2 — "The Self" (Jar 3): prepend profile.md as a leading
#   system message every request so the companion knows who you are from
#   line one. Plain text, editable.
# Brick 5 — Honesty (Jar 3b): prepend data/prompt.md as a leading system
#   message (right after the profile) carrying the guard against confabulation.
#   Plain text, editable; never written to Jar 1.
# Brick 4 — "The Search" (Jar 2, retrieval): on every inlet, take the latest
#   user message as the query, scan the indexed past (Jar 2), and inject a
#   short "Remember" system message carrying the top recency-gated,
#   keyword+meaning (hybrid) hits. This is the only brick that reads Jar 2.
#
# Idempotent: every request re-diffs the WHOLE history; a repeat never
# re-writes old turns; a transient write self-heals on the next request.
#
# Safety: capture (Brick 1), profile injection (Brick 2) and retrieval
# (Brick 4) live in separate try/excepts. A failure in ANY brick NEVER
# breaks the chat — the pipe always returns `body` unchanged. Logs go to
# docker logs (open-webui).
#
# NOTE (v2 layout): the pipe's runtime host is the open-webui container. It
# has only `companion/data` bind-mounted (NOT `companion/config`), so it
# CANNOT read companion.yaml. The retrieval defaults are therefore mirrored
# here as the RETRIEVAL dict; keep them in sync with the yaml.

import json
import os
import re
import sqlite3

DATA_DIR = os.environ.get("COMPANION_DATA",
                          os.path.expanduser("~/companion/data"))
SQLITE   = os.path.join(DATA_DIR, "record.sqlite")
PROFILE  = os.path.join(DATA_DIR, "profile.md")
PROFILE_ID = "companion-self-profile"    # stable id, idempotent, never written to Jar 1
MEMORY_ID  = "companion-memory-retrieved" # stable id, refreshed each turn, never written to Jar 1
PROMPT     = os.path.join(DATA_DIR, "prompt.md")
HONEST_ID  = "companion-honesty-rules"    # stable id, injected every turn, never written to Jar 1

# --- Retrieval config (Brick 4) — mirror of companion.yaml `retrieval:` ---
# (open-webui sees only data/, not config/, so this is the source of truth
#  that runs. relevance = 0.5*meaning(cosine) + 0.5*keyword(overlap).
# NOTE (v2 layout): the pipe's runtime host is the open-webui container. It
# has only `companion/data` bind-mounted (NOT `companion/config`), so it
# CANNOT read companion.yaml. The retrieval defaults are therefore mirrored
# here as the RETRIEVAL dict; keep them in sync with the yaml.
RETRIEVAL = {
    "default_window_days": 7,           # recency window (days)
    "expanded_window_days": 90,         # widened window when user triggers it
    "expand_trigger": "think back",     # phrase in the user message that widens the window
    "max_context_tokens_for_retrieval": 4000,
    "relevance_weight": 0.7,
    "salience_weight": 0.3,
    "relevance_blend_meaning": 0.5,     # weight of semantic (bge cosine) in relevance
    "relevance_blend_keyword": 0.5,     # weight of lexical keyword overlap in relevance
    "salience": {"high": 1.0, "medium": 0.5, "low": 0.2},
    "max_results": 10,
    "summary_max_chars": 220,
    "keyword_min_chars": 4,
}

_EMBEDDER = None  # module-level bge cache (set once per WebUI process)


def _conn():
    return sqlite3.connect(SQLITE, timeout=5.0)


def _model_short(model):
    """Reduce the full model object the WebUI hands back to a short, stable
    label. We never need the whole object to identify which model spoke."""
    if not model:
        return None
    if isinstance(model, dict):
        return model.get("name") or model.get("id")
    return model


def _load_profile():
    """Read profile.md (Jar 3). Missing/empty -> empty string. Never throws."""
    try:
        with open(PROFILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _inject_profile(body):
    """Prepend the profile as a leading system message so the model sees it
    every request. Idempotent (stable id) and never written to Jar 1."""
    text = _load_profile()
    if not text:
        return
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    if any((m or {}).get("id") == PROFILE_ID for m in msgs):
        return
    msgs.insert(0, {
        "id": PROFILE_ID,
        "role": "system",
        "content": text,
        "children_ids": [],
    })


def _inject_honesty(body):
    """Prepend the honesty/voice prompt (data/prompt.md) as a leading system
    message, right after the profile, so the model knows how to answer before
    it sees any context. Idempotent (stable id), never written to Jar 1."""
    try:
        with open(PROMPT, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception:
        return ""
    if not text:
        return
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    if any((m or {}).get("id") == HONEST_ID for m in msgs):
        return
    msgs.insert(1, {
        "id": HONEST_ID,
        "role": "system",
        "content": text,
        "children_ids": [],
    })


def _ensure_table():
    con = _conn()
    con.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        " id           TEXT PRIMARY KEY,"
        " chat_id      TEXT NOT NULL,"
        " chat_title   TEXT,"
        " parent_id    TEXT,"
        " children_ids TEXT,"
        " role         TEXT,"
        " content      TEXT,"
        " models       TEXT,"
        " done         INTEGER,"
        " ts           INTEGER,"
        " captured_at  TEXT DEFAULT (datetime('now'))"
        " ) "
    )
    con.commit()
    con.close()


def _content_of(msg):
    """Extract human-readable text from whatever shape Open WebUI hands back
    (str, block dict, or list of blocks). A user message must never drop."""
    c = msg.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):                       # {'type':'output_text'|..., 'text':...}
        return c.get("text", "")
    if isinstance(c, list):                       # list of blocks
        out = []
        for block in c:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t in ("output_text", "input_text", "text"):
                out.append(block.get("text", ""))
            elif isinstance(block.get("text"), str):
                out.append(block["text"])
        return "\n".join(out).strip()
    return str(c)


def _content_words(text):
    """Cheap lexical word set (lower-cased, alphanumeric + apostrophe) for
    keyword overlap. Pure stdlib, never raises."""
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _append_messages(messages, chat_id, model):
    """Diff `messages` vs the DB; insert only rows whose id is new.
    Returns (inserted).  Only real turns are recorded — the synthetic
    profile/memory system messages are never written to Jar 1."""
    con = _conn()
    mshort = _model_short(model)
    ins = 0
    cur = con.cursor()
    for msg in messages or []:
        mid = msg.get("id")
        if mid == PROFILE_ID or mid == MEMORY_ID or mid == HONEST_ID or not mid:
            continue                               # skip synthetic system lines
        role = (msg.get("role") or "unknown").lower()
        if role not in ("user", "assistant"):
            continue                               # only record real turns
        ins += cur.execute(
            "INSERT OR IGNORE INTO messages ("
            " id, chat_id, chat_title, parent_id, children_ids, role, "
            " content, models, done, ts) VALUES (?,?,?,?,?,?,?,?,?,?)"
            ,
            (
                mid,
                chat_id,
                msg.get("chat_title"),
                msg.get("parent_id"),
                msg.get("children_ids"),
                role,
                _content_of(msg),
                json.dumps([mshort]) if mshort else None,   # short name, not the ~2KB object
                1 if role == "assistant" else 0,
                msg.get("ts"),
            ),
        ).rowcount
    con.commit()
    con.close()
    return ins


def _embedder():
    """Load bge-small OFFLINE from the persistent hfcache (the same model and
    the same load path as housekeeping.py). Returns (model, dim) or (None, None).
    Loaded once per process (module-level cache); a load failure means
    retrieval simply returns nothing rather than breaking the chat."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        import glob
        # Pin offline + local cache so the model never touches the network.
        os.environ.setdefault("HF_HOME", os.path.join(DATA_DIR, "hfcache"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer
        repo_dir = os.path.join(DATA_DIR, "hfcache", "models--BAAI--bge-small-en-v1.5")
        snap_root = os.path.join(repo_dir, "snapshots")
        cands = []
        if os.path.isdir(snap_root):
            cands.extend(sorted(glob.glob(os.path.join(snap_root, "*"))))
        cands.append(repo_dir)                          # full cached dir
        cands.append("BAAI/bge-small-en-v1.5")          # last resort: repo id (offline)
        last = None
        for c in cands:
            try:
                m = SentenceTransformer(c, local_files_only=(c != "BAAI/bge-small-en-v1.5"))
                try:
                    dim = m.get_embedding_dimension()
                except Exception:
                    dim = m.get_sentence_embedding_dimension()
                _EMBEDDER = (m, dim)
                print("[companion:retrieve] bge loaded (dim=%s)" % dim)
                return (m, dim)
            except Exception as e:
                last = e
                print("[companion:retrieve] bge candidate %r failed: %s" % (c, e))
        raise last
    except Exception as e:
        print("[companion:retrieve] bge unavailable; retrieval disabled: %s" % e)
        _EMBEDDER = (None, None)
        return _EMBEDDER


def _retrieve(body):
    """Inject a recency-gated hybrid memory (keyword + meaning) as a single
    'Remember' system message. Query = latest user message; hits already
    present in the context window are excluded. Never written to Jar 1 (role
    == 'system'). Any failure is swallowed so the chat always proceeds."""
    try:
        msgs = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(msgs, list):
            return

        # Message ids already in the context window -> never re-inject them.
        in_ctx = set()
        for m in msgs:
            mid = m.get("id") if isinstance(m, dict) else None
            if mid:
                in_ctx.add(mid)

        # Query = latest user message.
        query = ""
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            if (m.get("role") or "").lower() != "user":
                continue
            q = _content_of(m)
            if q and q.strip():
                query = q.strip()
                break
        if not query:
            return

        # Recency gate + "think back" expand.
        trigger = RETRIEVAL["expand_trigger"].lower()
        expanded = trigger in query.lower()
        days = RETRIEVAL["expanded_window_days"] if expanded else RETRIEVAL["default_window_days"]

        # index_entries has no captured_at column, so join messages.captured_at
        # (same clock as datetime('now') in this container).
        con = _conn()
        cur = con.cursor()
        sql = (
            "SELECT ie.message_id, ie.summary, ie.keywords, ie.confidence, "
            "       ie.salience, ie.vectors, m.captured_at "
            "FROM index_entries ie "
            "JOIN messages m ON m.id = ie.message_id "
            "WHERE ie.indexed = 1 "
            "  AND m.captured_at IS NOT NULL "
            "  AND m.captured_at >= datetime('now', '-' || ? || ' days')"
        )
        params = [days]
        if in_ctx:
            sql += "  AND m.id NOT IN (" + ",".join("?" * len(in_ctx)) + ")"
            params.extend(sorted(in_ctx))
        rows = cur.execute(sql, params).fetchall()
        con.close()
        if not rows:
            print("[companion:retrieve] no candidates in window(%d d)" % days)
            return

        model, dim = _embedder()
        if model is None or dim is None:
            return
        import numpy as np
        qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        qw = _content_words(query)

        results = []
        for row in rows:
            _mid, summary, keywords, _conf, sal, blob, cap = row
            vec = None
            try:
                if isinstance(blob, (bytes, bytearray)) and len(blob) == dim * 4:
                    vec = np.frombuffer(blob, dtype=np.float32)
            except Exception:
                vec = None
            if vec is None or vec.size != dim:
                continue
            cosine = float(np.dot(qv, vec))
            cos = max(0.0, min(1.0, cosine))
            # Keyword overlap (lexical), against the summary + keywords blob.
            hay = _content_words(" ".join(x for x in (summary, keywords) if x))
            if qw and hay:
                kw = len([w for w in qw if w in hay]) / len(qw)
            else:
                kw = 0.0
            kw = max(0.0, min(1.0, kw))
            relevance = (RETRIEVAL["relevance_blend_meaning"] * cos +
                         RETRIEVAL["relevance_blend_keyword"] * kw)
            sal_val = RETRIEVAL["salience"].get(sal, RETRIEVAL["salience"].get("low", 0.2))
            score = (RETRIEVAL["relevance_weight"] * relevance +
                     RETRIEVAL["salience_weight"] * sal_val)
            results.append((score, cap, summary, keywords, kw, cos))

        results.sort(key=lambda t: (t[0], t[1]), reverse=True)
        results = results[:RETRIEVAL["max_results"]]
        if not results:
            return

        # Render the memory, respecting the token budget.
        budget_chars = int(RETRIEVAL["max_context_tokens_for_retrieval"]) * 4
        smax = RETRIEVAL["summary_max_chars"]
        lines = []
        window_note = ("last %d days" % days) + ("" if not expanded else " (think-back, 90d)")
        used = 0
        for i, (score, cap, summary, keywords, kw, cos) in enumerate(results, 1):
            s = (summary or "").strip()
            if len(s) > smax:
                s = s[:smax].rstrip() + "…"
            kw_txt = "kw:" + (" " + ",".join(sorted(_content_words(keywords))[:5])) if keywords else ""
            line = "%d. [%s] %s  (%.2f meaning, %.2f keyword) %s" % (
                i, cap, s, cos, kw, kw_txt)
            if used + len(line) + 2 > budget_chars and len(lines):
                break
            lines.append(line)
            used += len(line) + 2
        if not lines:
            return
        header = "## Remember (retrieved, %s)" % window_note
        text = header + "\n" + "\n".join(lines)

        # Place the 'Remember' block at the END of the messages list, every
        # turn. The conversation is append-only, so [profile][honesty][user..ass
        # ..user] is a byte-stable prefix across turns. A Remember that sits
        # mid-conversation and changes content each turn breaks the llama.cpp
        # prefix cache (that was the 7 s / f_keep ~0.24 turn). Retrieval content
        # is unchanged — only the block's position moves to the end, which is
        # also the better recency spot for the model.
        # Drop any previously-placed Remember so it cannot linger mid-conversation.
        msgs[:] = [m for m in msgs
                   if not (isinstance(m, dict) and m.get("id") == MEMORY_ID)]
        msgs.append({
            "id": MEMORY_ID,
            "role": "system",
            "content": text,
            "children_ids": [],
        })
        print("[companion:retrieve] %d hit(s) | window=%dd | top=%s" % (
            len(results), days, ", ".join(("%.2f" % t[0]) for t in results[:3])))
    except Exception:
        import traceback
        print("[companion:retrieve ERROR]\n" + traceback.format_exc())


class Filter:
    async def inlet(self, body, __user__=None, __metadata__=None):
        md = __metadata__ or {}
        # Brick 1 — capture (must never break)
        try:
            _ensure_table()
            chat_id = md.get("chat_id") or body.get("chat_id") or "unknown-chat"
            model = md.get("model")
            ins = _append_messages(body.get("messages"), chat_id, model)
        except Exception:
            import traceback
            print(f"[companion:capture ERROR]\n{traceback.format_exc()}")
        # Brick 2 — self (separate try so it can never hide capture errors)
        try:
            _inject_profile(body)
        except Exception:
            import traceback
            print(f"[companion:profile ERROR]\n{traceback.format_exc()}")
        # Brick 5 — honesty (separate try; leading system msg right after profile)
        try:
            _inject_honesty(body)
        except Exception:
            import traceback
            print(f"[companion:honesty ERROR]\n{traceback.format_exc()}")
        # Brick 4 — search (separate try; runs after profile so it lands at idx 1)
        try:
            _retrieve(body)
        except Exception:
            import traceback
            print(f"[companion:retrieve ERROR]\n{traceback.format_exc()}")
        try:
            print(f"[companion:inlet] chat={md.get('chat_id')} "
                  f"model={_model_short(md.get('model'))} new={ins}")
        except Exception:
            pass
        return body

    async def outlet(self, body, __user__=None, __metadata__=None):
        md = __metadata__ or {}
        # Outlet captures the assistant reply (inlet may miss it on streaming);
        # same idempotent diff, so it is harmless to run twice. No retrieval
        # here: the model has already answered, so a memory injected now would
        # only affect a later request — and inlet already handles that turn.
        try:
            _ensure_table()
            chat_id = md.get("chat_id") or body.get("chat_id") or "unknown-chat"
            model = md.get("model")
            ins = _append_messages(body.get("messages"), chat_id, model)
        except Exception:
            import traceback
            print(f"[companion:capture ERROR]\n{traceback.format_exc()}")
        try:
            print(f"[companion:outlet] chat={md.get('chat_id')} "
                  f"model={_model_short(md.get('model'))} new={ins}")
        except Exception:
            pass
        return body
