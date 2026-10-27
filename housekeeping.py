#!/usr/bin/env python3
"""
Companion housekeeping — writes Jar 2 (the index).

Long-running process. Under Option A it runs as a systemd --user unit on the
the real host. It is idle-gated: when the
record has had no new messages for more than idle_threshold_seconds it claims
up to batch_size messages that are not yet *successfully indexed* and writes
one index_entries row per message: a short SUMMARY, KEYWORDS, a CONFIDENCE
score, a SALIENCE label (high/medium/low), and a bge-small vector (BLOB).
The row is the LABEL on the drawer, not the contents — the contents stay in
Jar 1 (record.sqlite messages), which this process NEVER reads, writes, or
deletes.

Design rules (from the v2 blueprint):
  * index_entries has NO UNIQUE constraint on message_id, so dedupe is done
    by the terminal indexed=1 flag. A row is (re)claimed while indexed=0.
  * On a failed 27B call the row stays indexed=0 and its retries counter
    is bumped; after max_summarization_retries it is marked indexed=1 with a
    "(skipped...)" summary so it stops re-processing.
  * Idle check uses captured_at. The ts column is mostly NULL and is NOT
    trusted.
  * No external deps beyond the .venv (sentence_transformers, numpy, pyyaml).
    The 27B is called with stdlib urllib (no 'requests' installed).
  * bge-small loads OFFLINE from the persistent local cache
    companion/data/hfcache, so it survives container restarts and is
    backup-able with the rest of the folder.
  * Never writes Jar 1 (record.sqlite messages) or profile.md.
  * Only localhost / LAN calls.
"""
import os
import re
import sys
import json
import glob
import time
import sqlite3
import logging
import urllib.request
import datetime

# Resolve the companion root. Prefer the HOME env set by the systemd unit.
def find_root():
    for h in (os.environ.get("COMPL_HOME"), os.path.expanduser("~")):
        if h and os.path.isdir(os.path.join(h, "companion")):
            return os.path.abspath(h)
    return os.path.expanduser("~")

ROOT = find_root()
BASE = os.path.join(ROOT, "companion")
CFG_PATH = os.path.join(BASE, "config", "companion.yaml")
LOG_PATH = os.path.join(BASE, "logs", "events.log")
DB_PATH = os.path.join(BASE, "data", "record.sqlite")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = logging.getLogger("housekeeping")
log.setLevel(logging.INFO)
_h = logging.FileHandler(LOG_PATH)
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_h)

NP = None  # numpy (lazy, to keep top-level light)


def cfg_get(cfg, key, default=None):
    # cfg_get(cfg, key) and cfg_get(cfg, key, default): the third argument is
    # the default value (a normal positional), NOT a second path component.
    # This is what fixed the startup TypeError where %d got a None.
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    return default


def load_yaml(path):
    import yaml
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise SystemExit("cannot load config %s: %s" % (path, e))


def load_config():
    return load_yaml(CFG_PATH)


# ------------------------------------------------------------- bge ----
class Embedder:
    def __init__(self):
        self.model = None
        self.dim = None
        self._load()

    def _load(self):
        cfg = load_config()
        dbpath = os.path.expanduser(cfg_get(cfg, "record_db") or DB_PATH)
        data_dir = os.path.dirname(os.path.abspath(dbpath))
        self.hf_home = os.path.join(data_dir, "hfcache")
        # Pin offline + local cache so the model never touches the network.
        for k, v in {"HF_HOME": self.hf_home, "HF_HUB_OFFLINE": "1",
                     "TRANSFORMERS_OFFLINE": "1",
                     "TOKENIZERS_PARALLELISM": "false"}.items():
            if k not in os.environ:
                os.environ[k] = v
        repo_dir = os.path.join(self.hf_home, "models--BAAI--bge-small-en-v1.5")
        snap_root = os.path.join(repo_dir, "snapshots")
        cands = []
        if os.path.isdir(snap_root):
            cands.extend(sorted(glob.glob(os.path.join(snap_root, "*"))))
        cands.append(repo_dir)  # full cached dir
        cands.append("BAAI/bge-small-en-v1.5")  # last resort: repo id (offline)
        from sentence_transformers import SentenceTransformer
        last = None
        for c in cands:
            try:
                m = SentenceTransformer(c, local_files_only=(c != "BAAI/bge-small-en-v1.5"))
                self.model = m
                try:
                    self.dim = m.get_embedding_dimension()
                except Exception:
                    self.dim = m.get_sentence_embedding_dimension()
                log.info("bge loaded from %r (dim=%s)", c, self.dim)
                return
            except Exception as e:
                last = e
                log.warning("bge candidate %r failed: %s", c, str(e)[:140])
        raise SystemExit("bge load failed from all candidates; last error: %s" %
                         str(last)[:220])

    def encode(self, texts):
        """Return a numpy array, shape (n, dim), float32, L2-normalised."""
        import numpy as np
        global NP
        if NP is None:
            NP = np
        v = self.model.encode(list(texts), convert_to_numpy=True,
                              normalize_embeddings=True)
        return v.astype(NP.float32)

    def vector_to_blob(self, vec):
        return vec.tobytes()  # float32 little-endian (native on x86_64)


# -------------------------------------------- summarizer endpoint discovery ---
# The summarizer must follow whichever model is ACTIVE, not a hardcoded
# port/model name. If the current model gets relegated, a pinned
# llama_cpp_api + model_path would silently break and need a code edit.
#
# Resolution order, cached for resolve_cache_seconds and re-checked whenever a
# call fails:
#   1. config llama_cpp_api + model_path, but ONLY if that endpoint is alive
#      and actually serving that model. This keeps an explicit override working
#      while never trusting a dead one.
#   2. probe the candidate localhost ports for GET /models and take the first
#      one that lists a loaded model id. /models is required deliberately:
#      unrelated services answer /health fine but 404 /models (:8000 does),
#      so /health alone would produce a false positive.
#   3. no live endpoint -> raise. The message stays indexed=0 and retries,
#      which is correct: there is nothing to summarize with yet.
_DISCOVERY_CACHE = {"base": None, "model": None, "until": 0.0}


def _probe_models(base, timeout):
    """Return the first loaded model id advertised at base, else None."""
    try:
        req = urllib.request.Request(base.rstrip("/") + "/models",
                                     method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception:
        return None
    items = data.get("data") or []
    ids = []
    for it in items:
        if isinstance(it, dict) and it.get("id"):
            st = it.get("status")
            if isinstance(st, dict):
                st = st.get("value")
            # llama-server reports status 'loaded'; accept a missing status too
            # since not every OpenAI-compatible server reports one.
            if st in (None, "loaded"):
                ids.append(it["id"])
    return ids[0] if ids else None


def resolve_summarizer(cfg):
    """Return (base_url, model_id) for the ACTIVE model. Raises if none live."""
    now = time.time()
    if _DISCOVERY_CACHE["base"] and now < _DISCOVERY_CACHE["until"]:
        return _DISCOVERY_CACHE["base"], _DISCOVERY_CACHE["model"]

    cache_secs = float(cfg_get(cfg, "resolve_cache_seconds", 60) or 60)
    probe_to = float(cfg_get(cfg, "resolve_probe_timeout_seconds", 3) or 3)

    # 1. explicit config, validated
    api = cfg_get(cfg, "llama_cpp_api")
    model = cfg_get(cfg, "model_path")
    if api and model:
        ids = _probe_models(api, probe_to)
        if ids and ids[0] == model:
            _DISCOVERY_CACHE.update(base=api.rstrip("/"), model=model,
                                    until=now + cache_secs)
            return api.rstrip("/"), model

    # 2. discover: any localhost port serving an OpenAI-compatible /models.
    ports = cfg_get(cfg, "candidate_ports") or [8096, 8081, 8080]
    for p in ports:
        base = "http://127.0.0.1:%s" % p
        found = _probe_models(base, probe_to)
        if found:
            log.info("summarizer resolved by discovery: %s (%s)", base, found)
            _DISCOVERY_CACHE.update(base=base, model=found,
                                    until=now + cache_secs)
            return base, found

    raise RuntimeError("no live OpenAI-compatible summarizer endpoint found "
                       "(checked: %s)" % ", ".join(str(p) for p in ports))


def invalidate_summarizer():
    """Drop the cache so the next call re-discovers the active model."""
    _DISCOVERY_CACHE.update(base=None, model=None, until=0.0)


# --------------------------------------------------------- 27B call ---
# The summarizer prompt lives here, not inline in ask_27b(), so the bench tool
# and production send byte-identical messages. Only the model varies between runs.
# BUGFIX: the original prompt (no example) made the 27B drop the SUMMARY line
# and sometimes return empty output. A strict, short prompt with one concrete
# example anchors the format reliably (tested 5/5 vs 2/5 without it).
# BUGFIX: attribution. Every turn was previously handed over as an anonymous
# user-role message, and the example below literally said "User requested...",
# so the summarizer credited YOU with everything the agent did. It now gets an
# explicit USER TURN / ASSISTANT TURN tag plus an instruction to name the actor.
# The tag lives in the content, NOT in the message role: ending the array with
# an assistant-role message would make a chat model continue that turn instead
# of answering the label contract, which is the format we fought for elsewhere.
PROMPT_SYS = ("You are a summarizer. The final message contains one chat turn to "
              "summarize, tagged USER TURN or ASSISTANT TURN. Earlier turns are context only; "
              "do not reply to them.\n"
              "SUMMARY must describe ONLY the tagged turn: what the speaker of that turn said, "
              "asked, decided, or did. Do not describe earlier turns, and do not describe what "
              "happened in response to the tagged turn.\n"
              "Output ONLY four lines in this exact format, no extra text:\n"
              "SUMMARY: <sentence>\n"
              "KEYWORDS: <words>\n"
              "CONFIDENCE: <0.0-1.0>\n"
              "SALIENCE: <high|medium|low>\n"
              "An ASSISTANT TURN is written in first person by the assistant, so 'I' and 'my' inside "
              "it refer to the assistant, never the user.\n"
              "In SUMMARY, name whoever actually did the thing: 'the user' for a USER TURN, "
              "'the assistant' for an ASSISTANT TURN. Never credit the user with an action the "
              "assistant took, or the reverse.\n"
              "Example for 'ASSISTANT TURN: I restarted the service, it is indexing now':\n"
              "SUMMARY: The assistant restarted the service, which began indexing messages.\n"
              "KEYWORDS: restart, service, indexing\n"
              "CONFIDENCE: 0.9\n"
              "SALIENCE: medium")


def build_messages(target_text, preceding, target_role="user"):
    """target_role is the role of the turn being summarized ('user'|'assistant').
    Defaults to 'user' so the bench tools keep working unchanged."""
    msgs = [{"role": "system", "content": PROMPT_SYS}]
    for p in preceding:
        msgs.append({"role": p["role"], "content": (p["content"] or "")[:1000]})
    tag = "ASSISTANT TURN" if (target_role or "").lower() == "assistant" else "USER TURN"
    msgs.append({"role": "user",
                 "content": "%s to summarize:\n%s" % (tag, (target_text or "")[:1000])})
    return msgs


def ask_27b(cfg, target_text, preceding, timeout, target_role="user"):
    base, model = resolve_summarizer(cfg)
    msgs = build_messages(target_text, preceding, target_role)
    # max_tokens is configurable: reasoning models (qwen3.8-flash-next) spend
    # part of the budget on a separate reasoning channel, and at 200 the four
    # labelled lines got truncated mid-answer (finish_reason=length, no
    # CONFIDENCE/SALIENCE). 600 completes reliably.
    max_tokens = int(cfg_get(cfg, "summarize_max_tokens", 600) or 600)
    payload = {"model": model, "messages": msgs,
               "temperature": 0.1, "max_tokens": max_tokens}
    # FIX: the summarizer model (Compactor-Qwen3.5-4B) is chat-tpl thinking-capable.
    # With --jinja and no explicit override it emits a reasoning trace instead of the
    # four labelled lines: bench measured 0/10 parseable + 3/10 keywords there vs
    # 10/10 + 10/10 with thinking off. cfg_get() is flat (top-level keys only), so
    # summarize_thinking is a top-level config key. Default False: the labelled-line
    # contract needs the answer channel, not the reasoning channel. Set True only for
    # a model that puts the label block in content anyway (flash-next does).
    if bool(cfg_get(cfg, "summarize_thinking", False)) is False:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    msg = data["choices"][0]["message"]
    content = msg.get("content")
    if not content:
        # Reasoning backends can return content=None with the answer parked in
        # reasoning_content. Only accept that fallback if the label block is
        # actually present; otherwise let it fail and retry.
        content = msg.get("reasoning_content") or ""
        if "SUMMARY:" not in content:
            raise RuntimeError("empty completion (reasoning model truncated?)")

    def grab(label):
        m = re.search(r"%s:\s*(.+)" % label, content)
        return m.group(1).strip() if m else ""

    cf = grab("CONFIDENCE")
    try:
        conf = max(0.0, min(1.0, float(cf)))
    except Exception:
        conf = 0.0
    sal = grab("SALIENCE").lower()
    sal = sal if sal in ("high", "medium", "low") else "low"
    # BUGFIX: a missing SUMMARY left the index entry with an empty summary.
    # Fall back to a short excerpt of the actual turn so a summary always exists.
    summary = grab("SUMMARY")
    if not summary:
        summary = re.sub(r"\s+", " ", (target_text or "").strip())[:120]
    return {
        "summary": summary,
        "keywords": grab("KEYWORDS"),
        "confidence": conf,
        "salience": sal,
    }


def parse_dt(s):
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:
        return None


def _now_utc():
    # Jar 1's captured_at is written by the pipe inside the open-webui
    # container, whose clock is UTC (measured offset == 0). This service runs
    # on the real desktop host in a local zone (CDT, UTC-5). A naive
    # datetime.now() is therefore ~5h behind every captured_at, so (now-last)
    # is always negative and the idle gate never opens. Always compare in UTC
    # so the gate is correct on any host timezone.
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

def is_idle(db, cfg):
    thresh = float(cfg_get(cfg, "idle_threshold_seconds", 30) or 30)
    row = db.execute(
        "SELECT MAX(captured_at) FROM messages WHERE captured_at IS NOT NULL").fetchone()
    last = parse_dt(row[0]) if row else None
    if last is None:
        return False
    return (_now_utc() - last).total_seconds() > thresh


def batch_id():
    return "b-%s-%s" % (datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                         str(os.getpid())[-4:])


def open_db():
    return sqlite3.connect(DB_PATH, timeout=30)


def claim_batch(db, cfg):
    n = int(cfg_get(cfg, "batch_size", 20) or 20)
    return db.execute("""
        SELECT m.id, m.chat_id, m.role, m.content, m.captured_at
        FROM messages m
        WHERE m.role IN ('user','assistant')
          AND m.content IS NOT NULL AND length(m.content) >= 3
          AND m.id NOT IN (SELECT message_id FROM index_entries WHERE indexed = 1)
        ORDER BY m.captured_at ASC, m.id
        LIMIT ?""", (n,)).fetchall()


def preceding_for(db, chat_id, mid, limit=4):
    # BUGFIX: returned raw SQLite tuples; ask_27b indexes them as dicts (p["role"]),
    # so raise "tuple indices must be integers". Return dicts instead.
    rows = db.execute("""
        SELECT role, content FROM messages
        WHERE chat_id = ? AND role IN ('user','assistant')
          AND content IS NOT NULL AND length(content) >= 3 AND id != ?
        ORDER BY captured_at DESC LIMIT ?""", (chat_id, mid, limit)).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]


def process_batch(db, cfg, rows, emb):
    """Return (indexed_ok, skipped). Writes index_entries rows for `rows`."""
    batch = batch_id()
    max_retry = int(cfg_get(cfg, "max_summarization_retries", 5) or 5)
    n_ok = n_skipped = 0

    # If nothing is serving a model right now, do NOT claim-and-burn: retry
    # counters are what eventually mark a row terminally "skipped", and a
    # skipped row never retries again. A server being down is a temporary
    # condition, not a per-message failure, so leave the rows untouched.
    try:
        resolve_summarizer(cfg)
    except Exception as e:
        log.info("skipping batch, no summarizer yet: %s", str(e)[:140])
        return 0, 0

    # Embed all messages in one vectorised call (cheap, CPU, bge-small).
    vecs = emb.encode([r[3] for r in rows])

    for i, (mid, chat_id, role, content, cap) in enumerate(rows):
        pre = preceding_for(db, chat_id, mid)
        try:
            res = ask_27b(cfg, content, pre,
                          timeout=int(cfg_get(cfg, "summarize_timeout_seconds", 180) or 180),
                          target_role=role)
            if res is None:
                raise RuntimeError("27B returned None")
        except Exception as e:
            log.warning("msg %s: 27B call failed: %s", mid, str(e)[:140])
            # The endpoint may have gone away (model relegated / server
            # restarted on another port). Force a re-discovery before the next
            # message so the batch follows the new active model.
            invalidate_summarizer()
            _bump_and_maybe_skip(db, mid, chat_id, batch, max_retry)
            continue

        # Success: replace any placeholder row with the final indexed row.
        db.execute("DELETE FROM index_entries WHERE message_id=?", (mid,))
        db.execute("""
            INSERT INTO index_entries
                (message_id, batch_id, chat_id, summary, keywords, confidence,
                 salience, vectors, indexed, retries, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, datetime('now'))""",
                   (mid, batch, chat_id, res["summary"], res["keywords"],
                    res["confidence"], res["salience"],
                    emb.vector_to_blob(vecs[i])))
        log.info("msg %s indexed | chat %s | salience %s | conf %.2f",
                 mid, chat_id, res["salience"], res["confidence"])
        n_ok += 1

    db.commit()
    return n_ok, n_skipped


def _bump_and_maybe_skip(db, mid, chat_id, batch, max_retry):
    """Bump retries; once max_retry is hit, mark the row terminal (skipped)."""
    cur = db.execute(
        "SELECT message_id FROM index_entries WHERE message_id=?", (mid,)).fetchone()
    if cur is not None:
        db.execute("UPDATE index_entries SET retries = retries + 1 WHERE message_id=?", (mid,))
    else:
        # BUGFIX: placeholder row omitted the NOT NULL chat_id column.
        db.execute("INSERT INTO index_entries(message_id, chat_id, batch_id, "
                   "indexed, retries) VALUES (?, ?, ?, 0, 1)", (mid, chat_id, batch))
    ret = db.execute(
        "SELECT retries FROM index_entries WHERE message_id=?", (mid,)).fetchone()
    retries = ret[0] if ret else 0
    if retries >= max_retry:
        db.execute("DELETE FROM index_entries WHERE message_id=? AND indexed = 0", (mid,))
        # chat_id is NOT NULL in the schema. Omitting it here raised
        # IntegrityError, which aborted the whole batch transaction — so the
        # retry bumps never committed, retries stayed pinned below max_retry,
        # and the batch re-claimed the same rows forever (crash loop every 10s).
        db.execute("""
            INSERT INTO index_entries
                (message_id, chat_id, batch_id, summary, keywords, confidence,
                 salience, vectors, indexed, retries, created_at)
            VALUES (?, ?, ?, ?, ?, 0.0, 'low', ?, 1, ?, datetime('now'))""",
                   (mid, chat_id, batch, "(skipped: model unavailable)", "",
                    b"", max_retry))
        log.info("msg %s: retries exhausted (%s/%s) -> marked skipped (terminal)",
                 mid, retries, max_retry)


# ------------------------------------------------------------- main ---
EMBED = None


def main():
    global EMBED
    cfg = load_config()
    try:
        EMBED = Embedder()
    except SystemExit:
        raise
    log.info(
        "housekeeping started | idle=%ss batch=%d poll=%ss max_retries=%d db=%s",
        cfg_get(cfg, "idle_threshold_seconds", 30),
        cfg_get(cfg, "batch_size", 20),
        cfg_get(cfg, "poll_interval_seconds", 10),
        int(cfg_get(cfg, "max_summarization_retries", 5) or 5),
        DB_PATH)
    try:
        import signal
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
        signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
    except Exception:
        pass
    poll = float(cfg_get(cfg, "poll_interval_seconds", 10) or 10)
    while True:
        # BUGFIX: the loop previously held a fresh connection each cycle with no
        # close(), so an open read-transaction (is_idle's SELECT) leaked every
        # iteration and eventually raised 'database is locked'.
        db = None
        try:
            db = open_db()
            if is_idle(db, cfg):
                rows = claim_batch(db, cfg)
                if rows:
                    log.info("idle; claiming %d messages", len(rows))
                    n_ok, _ = process_batch(db, cfg, rows, EMBED)
                    log.info("batch done: indexed=%d of claims=%d", n_ok, len(rows))
                else:
                    log.info("idle; nothing to index")
            # When not idle we just sleep; no DB work.
        except Exception as e:
            log.error("loop error: %s", repr(e)[:300])
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass
        time.sleep(poll)


if __name__ == "__main__":
    main()
