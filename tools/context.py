#!/usr/bin/env python3
"""context.py — what is actually live in the model's context, human-legible.

Read-only. Reconstructs the prompt stack the way Open WebUI + companion_pipe.py
will assemble it for the next turn, and prints it layer by layer:

  Brick 2  profile.md      (data/profile.md, prepended every request)
  Brick 5  prompt.md       (data/prompt.md,  prepended every request)
  chat     the message chain Open WebUI holds for this chat
  Brick 4  the Remember block the pipe would inject (recomputed live,
           same RETRIEVAL dict and same scoring as the deployed pipe)

Tokens are ESTIMATES (chars/4) except the total, which is taken from Open
WebUI's own context_usage when it reports one. Nothing here is written; the
DB is opened read-only and the OWU API is only GET-ed.

Flags (console shim: ~/.local/bin/ctx)
  ctx                 newest chat: stack + what retrieval would inject
  ctx -c 42c6         pick a chat by id prefix
  ctx --list          all chats, fill % and whether OWU ever compacted them
  ctx --stack         layers + totals only. Fast: skips the embedding model
  ctx --render        the Remember block byte-for-byte as the model gets it
  ctx --summary       OWU's compaction events for this chat, full text
  ctx --full          injected summaries untruncated
  ctx --tail N        the N newest messages actually inside the window
  ctx --log N         tail the pipe's own [companion:retrieve] log lines

Retrieval replay reads the config out of data/_deployed_dump/companion_pipe.py,
i.e. what is REALLY deployed, not what config/companion.yaml says should be.
A mismatch between the two is printed as CONFIG DRIFT.
"""
import argparse
import ast
import datetime as _dt
import json
import os
import re
import sqlite3
import subprocess
import sys
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
# OWU's own SQLite, host-side path for the container volume. Opened read-only.
WEBUI_DB = os.path.expanduser(
    "~/.local/share/containers/storage/volumes/open-webui/_data/webui.db")


def _settings():
    """Read OWU's context_compaction settings. GET-only; the /configs REST route
    is an SPA catch-all and cannot be read over the API, so this is the only leg
    that can see them."""
    if not os.path.exists(WEBUI_DB):
        return None
    con = sqlite3.connect("file:%s?mode=ro" % WEBUI_DB, uri=True)
    try:
        out = {}
        for k, v in con.execute("select key,value from config"):
            if k.startswith("chat.context_compaction"):
                key = k.split("context_compaction.")[1]
            elif k == "task.model.params":
                key = "task_model_params"
            else:
                continue
            try:
                out[key] = json.loads(v)
            except Exception:
                out[key] = v
        out.setdefault("task_model_params", {})
        return out
    finally:
        con.close()
CFG = os.path.join(BASE, "config", "companion.yaml")
DUMP = os.path.join(DATA, "_deployed_dump", "companion_pipe.py")

# ids the pipe injects synthetically; never recorded in Jar 1
SYNTH = ("companion-self-profile", "companion-memory-retrieved",
         "companion-honesty-rules")


def _w(s):
    """Rough token estimate. Labelled as such in output; do not treat as exact."""
    return len(s or "") // 4


def _cfg():
    """Pull token + base url out of companion.yaml without needing PyYAML."""
    tok, base = None, "http://127.0.0.1:3000"
    try:
        txt = open(CFG, encoding="utf-8").read()
        m = re.search(r'open_webui_token["\']?\s*:\s*["\']([^"\']+)', txt)
        if m:
            tok = m.group(1)
        m = re.search(r'open_webui_base(?:_url)?["\']?\s*:\s*["\']?([\w./:-]+)', txt)
        if m:
            base = m.group(1).rstrip("/")
    except OSError:
        pass
    return tok, base


def _get(url, tok, timeout=25):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer %s" % tok})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    if body.lstrip().startswith("<"):
        raise SystemExit("OWU returned HTML, not JSON (bad route or bad token?) for %s" % url)
    return json.loads(body)


def _retrieval_dict():
    """Read the RETRIEVAL dict literal out of the DEPLOYED dump, so the viewer
    reports the config that actually runs rather than a second copy."""
    try:
        tree = ast.parse(open(DUMP, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "RETRIEVAL":
                return ast.literal_eval(node.value)
    except Exception as e:
        print("  (could not read RETRIEVAL from %s: %s)" % (DUMP, e))
    return None


def _compaction(hist):
    """OWU's live context compaction, as exposed by the API.

    NOT chat.summary -- that column is NULL for every chat on this install.
    The real record is per-message:
      - output items of type "open_webui:context_compaction" (the live ones)
      - a contextSummary field on a checkpoint message (the stored one)
    Returns (events, latest) where events is a chronological list of dicts
    and latest is the newest non-empty summary text (or None).
    """
    msgs = hist.get("messages") or {}
    events = []
    for mid, m in msgs.items():
        out = m.get("output")
        if isinstance(out, list):
            for it in out:
                if isinstance(it, dict) and it.get("type") == "open_webui:context_compaction":
                    events.append({"mid": mid, "role": m.get("role"),
                                   "ts": m.get("timestamp") or 0,
                                   "src": "output", "text": it.get("summary") or ""})
        cs = m.get("contextSummary") or m.get("context_summary")
        if cs:
            events.append({"mid": mid, "role": m.get("role"),
                           "ts": m.get("timestamp") or 0,
                           "src": "contextSummary", "text": cs})
    events.sort(key=lambda e: (e["ts"], e["mid"]))
    latest = None
    for e in events:
        if e["text"].strip():
            latest = e
    return events, latest


def _chain(history):
    """Walk parentId from currentId -> linear oldest-first transcript.
    Returns (chain, offchain) where offchain are nodes on no branch of current."""
    msgs = history.get("messages") or {}
    cur = history.get("currentId")
    chain, seen = [], set()
    while cur and cur in msgs and cur not in seen:
        seen.add(cur)
        chain.append(msgs[cur])
        cur = msgs[cur].get("parentId")
    chain.reverse()
    return chain, [m for mid, m in msgs.items() if mid not in seen]


def _retrieve_live(cfg, query, in_ids, limit):
    """Replicate the deployed pipe's scoring. Returns (rows, note)."""
    days = cfg["default_window_days"]
    con = sqlite3.connect("file:%s?mode=ro" % os.path.join(DATA, "record.sqlite"), uri=True)
    try:
        sql = ("SELECT ie.message_id, ie.summary, ie.keywords, ie.salience, ie.vectors, "
               "       m.captured_at FROM index_entries ie "
               "JOIN messages m ON m.id = ie.message_id "
               "WHERE ie.indexed=1 AND m.captured_at IS NOT NULL "
               "  AND m.captured_at >= datetime('now','-' || ? || ' days')")
        rows = con.execute(sql, (days,)).fetchall()
    finally:
        con.close()
    if not rows:
        return [], "no indexed rows inside the %d-day window" % days

    try:
        import glob
        import numpy as np
        os.environ.setdefault("HF_HOME", os.path.join(DATA, "hfcache"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        repo = os.path.join(DATA, "hfcache", "models--BAAI--bge-small-en-v1.5")
        snap = sorted(glob.glob(os.path.join(repo, "snapshots", "*")))
        if not snap:
            return [], "bge snapshot missing in data/hfcache"
        model = SentenceTransformer(snap[-1], local_files_only=True)
        qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    except Exception as e:
        return [], "embedding unavailable (%s) — use --stack" % type(e).__name__

    kmin = cfg["keyword_min_chars"]

    def words(t):
        return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower())
                if len(w) >= kmin}

    qw = words(query)
    out = []
    for mid, summ, kws, sal, blob, cap in rows:
        if not isinstance(blob, (bytes, bytearray)) or len(blob) != 384 * 4:
            continue
        vec = __import__("numpy").frombuffer(blob, dtype=np.float32)
        cos = max(0.0, min(1.0, float(__import__("numpy").dot(qv, vec))))
        hay = words(" ".join(x for x in (summ, kws) if x))
        kw = (len([w for w in qw if w in hay]) / len(qw)) if qw and hay else 0.0
        rel = (cfg["relevance_blend_meaning"] * cos +
               cfg["relevance_blend_keyword"] * kw)
        score = (cfg["relevance_weight"] * rel +
                 cfg["salience_weight"] * cfg["salience"].get(sal, 0.2))
        out.append({"id": mid, "score": score, "cos": cos, "kw": kw,
                    "sal": sal, "cap": cap, "summary": (summ or "").strip(),
                    "keywords": kws or "", "in_ctx": mid in in_ids})
    out.sort(key=lambda r: (r["score"], r["cap"]), reverse=True)
    return out[:limit], "window=%dd pool=%d" % (days, len(rows))


def main():
    ap = argparse.ArgumentParser(description="Human-legible view of the live context.")
    ap.add_argument("-c", "--chat", help="chat id or prefix (default: newest chat)")
    ap.add_argument("-n", "--max-results", type=int, help="override max_results for display")
    ap.add_argument("--stack", action="store_true", help="layers + totals only; skip embedding")
    ap.add_argument("--full", action="store_true", help="do not truncate injected summaries")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--log", type=int, default=0, metavar="N",
                    help="also tail N of the pipe's [companion:retrieve] log lines")
    ap.add_argument("--list", action="store_true", help="list chats with their context fill")
    ap.add_argument("--render", action="store_true",
                    help="print the Remember block exactly as the model would receive it")
    ap.add_argument("--summary", action="store_true",
                    help="dump OWU's compaction events for this chat, full text")
    ap.add_argument("--tail", type=int, default=0, metavar="N",
                    help="print the last N chat messages the model actually sees")
    args = ap.parse_args()

    tok, base = _cfg()
    if not tok:
        raise SystemExit("no open_webui_token in %s" % CFG)

    if args.list:
        lst = _get(base + "/api/v1/chats/", tok)
        lst = lst if isinstance(lst, list) else lst.get("chats", [])
        print("%-9s %-6s %-4s %-26s %-16s %s"
              % ("chat", "tok", "pct", "title", "updated", "OWU compaction"))
        for c in sorted(lst, key=lambda x: x.get("updated_at") or 0, reverse=True):
            cid = c.get("id") or ""
            try:
                d = _get(base + "/api/v1/chats/%s" % cid, tok)
                ch = d.get("chat")
                if isinstance(ch, str):
                    ch = json.loads(ch)
            except Exception:
                ch = {}
            hist = ch.get("history") or {}
            cu = ch.get("context_usage") or d.get("context_usage") or c.get("context_usage") or {}
            ev, latest = _compaction(hist)
            ua = c.get("updated_at")
            when = _dt.datetime.fromtimestamp(ua).strftime("%Y-%m-%d %H:%M") if ua else "?"
            note = "never compacted"
            if latest:
                low = latest["text"].lower()
                bad = any(k in low for k in ("no prior", "no current", "not provided",
                                             "no conversation", "nothing to summar",
                                             "no working-session", "ungrounded"))
                note = "live %d chars%s" % (len(latest["text"]), "  <-- EMPTY-STATE" if bad else "")
            print("%-9s %-6s %-4s %-26s %-16s %d events, %s" % (
                cid[:8], cu.get("tokens") or "-", cu.get("percent") or "-",
                (ch.get("title") or c.get("title") or "")[:26], when, len(ev), note))
        print("\n  context_usage is only populated for a chat OWU has recently rendered;")
        print("  '-' there is normal. 'never compacted' means OWU's own summarizer has")
        print("  never run for that chat, so the whole older past falls on Brick 4.")
        print("  'EMPTY-STATE' = OWU compacted but wrote a summary saying there was nothing")
        print("  to compact. That is the clobber signature. Inspect one with: ctx -c <id> --summary")
        return

    # resolve chat
    if args.chat:
        lst = _get(base + "/api/v1/chats/", tok)
        lst = lst if isinstance(lst, list) else lst.get("chats", [])
        hit = [c for c in lst if (c.get("id") or "").startswith(args.chat)]
        if not hit:
            raise SystemExit("no chat starting with %r (try: ctx --list)" % args.chat)
        chat_id = hit[0]["id"]
    else:
        lst = _get(base + "/api/v1/chats/", tok)
        lst = lst if isinstance(lst, list) else lst.get("chats", [])
        if not lst:
            raise SystemExit("no chats visible to this token")
        chat_id = max(lst, key=lambda c: c.get("updated_at") or 0)["id"]

    # NOTE: /chats/id/<uuid> serves the SPA HTML on this build; /chats/<uuid>
    # is the route that returns JSON.
    d = _get(base + "/api/v1/chats/%s" % chat_id, tok)
    chat = d.get("chat") or {}
    if isinstance(chat, str):
        chat = json.loads(chat)
    hist = chat.get("history") or {}
    chain, offchain = _chain(hist)

    W = args.width
    print("=" * W)
    print("LIVE CONTEXT — %s" % (d.get("title") or "(untitled)"))
    print("  chat %s   nodes %d   on current branch %d   off-branch %d"
          % (chat_id[:8], len(hist.get("messages") or {}), len(chain), len(offchain)))
    cu = d.get("context_usage") or {}
    if cu:
        print("  OWU context_usage: %s tokens of %s  (%s%%)  source=%s"
              % (cu.get("tokens"), cu.get("threshold"), cu.get("percent"), cu.get("source")))

    ev, latest = _compaction(hist)
    print("  OWU compaction events: %d  (per-message; chat.summary column is NULL on "
          "this install and is NOT the live source)" % len(ev))

    st = _settings()
    if st:
        eff = min(st.get("token_threshold") or 0, st.get("token_cap") or 0)
        tp = st.get("task_model_params") or {}
        print("  OWU settings: enable=%s trigger>%d tok  cap=%s  keeps %s%% of msgs  summarizer=%s"
              % (st.get("enable"), eff, st.get("token_cap"),
                 st.get("retention_percentage"),
                 st.get("model") or "<the chat's own model>"))
        print("  task.model.params: %s -> summary max_tokens=%s. Empty means OWU uses the "
              "model's own preset, NOT the template's budget."
              % (json.dumps(tp), tp.get("max_tokens", "unset")))
        tpl = st.get("prompt_template") or ""
        if tpl and "{{COMPACTED_MESSAGES}}" not in tpl:
            print("  !! prompt_template has NO {{COMPACTED_MESSAGES}} / {{RECENT_MESSAGES}} / "
                  "{{PREVIOUS_SUMMARY}}\n"
                  "     variables, so OWU sends the summarizer a prompt with NO conversation "
                  "text in it.\n"
                  "     Every compaction on this install is therefore blind. This is the "
                  "clobber root cause.")
    else:
        print("  OWU settings: unreadable (%s not found on this host)" % WEBUI_DB)
    if latest:
        when = (_dt.datetime.fromtimestamp(latest["ts"]).strftime("%Y-%m-%d %H:%M")
                if latest["ts"] else "undated")
        print("  live summary on msg %s [%s] via %s : %d chars ~%d tok   written %s"
              % (latest["mid"][:8], latest["role"], latest["src"],
                 len(latest["text"]), _w(latest["text"]), when))
        head = latest["text"].strip().split("\n")[0][:70]
        print("  starts: %r" % head)
    else:
        print("  live summary: none. OWU has never compacted this chat.")
    # Two distinct failure shapes, and they must not be confused:
    #  (a) EMPTY-STATE -- summariser ran, saw no conversation text, wrote a stub.
    #  (b) NO-SNAPSHOT -- summariser saw the conversation, but the stored text is
    #      its own reasoning. OWU stores message.content OR message.reasoning_content
    #      (utils/context_compaction.py _response_text), and the compaction payload
    #      sets no reasoning/thinking toggle -- only max_tokens. So a reasoning-on
    #      model whose content channel came back empty has its chain of thought saved
    #      as Brick 0. The tell: our template demands a "1. STATE" heading and
    #      reasoning never contains one.
    EMPTY = ("no prior", "no current", "not provided", "no conversation",
             "nothing to summar", "no working-session", "ungrounded")
    if latest:
        low = latest["text"].lower()
        if not re.search(r"(?m)^\s*#?\s*1[.)]?\s*state", latest["text"]):
            print("\n  !! NO-SNAPSHOT: the stored summary has no '1. STATE' heading, so\n"
                  "     it is not the snapshot the template asked for -- it reads as the\n"
                  "     summariser's own reasoning. OWU saves content OR reasoning_content,\n"
                  "     and the compaction payload sends no thinking toggle, so a\n"
                  "     reasoning-on model leaks its chain of thought into Brick 0.\n"
                  "     Fix: give the summariser its own model with thinking off, or set\n"
                  "     reasoning_effort=none in task.model.params.\n"
                  "     task.model.params is %s, so summary length is whatever the model\n"
                  "     preset allows -- not the template's 1800-token cap.\n"
                  % json.dumps((st or {}).get("task_model_params")))
            # Keyword sniffing is meaningless over reasoning prose, so skip it here.
            low = ""
        if any(k in low for k in EMPTY):
            print("\n  !! The newest summary says there was NOTHING to compact, yet this chat\n"
                  "     holds %d nodes. Empty-state snapshot -- the clobber shape, not a\n"
                  "     healthy rollup. It was written %s. If that predates the template fix,\n"
                  "     this is stale damage, not a live fault: it clears on the next\n"
                  "     compaction, once the chat crosses %d tokens. Older past that was\n"
                  "     already discarded is NOT recoverable from OWU; it survives only in\n"
                  "     the Brick 4 index."
                  % (len(chain), when, (st or {}).get("token_cap") or 65536))
    if ev:
        print("  event sizes:", ", ".join(str(len(e["text"])) for e in ev[-12:]))

    if args.summary:
        print("=" * W)
        print("BRICK 0 — every OWU compaction event for this chat (%d)" % len(ev))
        print("  These are what OWU hands the model as '[CONVERSATION SUMMARY]'. The pipe's")
        print("  inlet runs AFTER OWU compacts (middleware.py:2145 then :2315), so this text")
        print("  is already in the body your inlet receives -- it is not something you add.")
        for i, e in enumerate(ev, 1):
            when = _dt.datetime.fromtimestamp(e["ts"]).strftime("%m-%d %H:%M") if e["ts"] else "?"
            print("\n--- %d/%d  msg %s [%s] via %s  %s  %d chars ---"
                  % (i, len(ev), e["mid"][:8], e["role"], e["src"], when, len(e["text"])))
            print(e["text"] if args.full else e["text"][:1200])
            if not args.full and len(e["text"]) > 1200:
                print("  …(truncated; --full for all %d chars)" % len(e["text"]))
        return

    # --- the stack, in the order the model receives it ---
    prof = open(os.path.join(DATA, "profile.md"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(DATA, "profile.md")) else ""
    hon = open(os.path.join(DATA, "prompt.md"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(DATA, "prompt.md")) else ""

    conv_txt = "\n".join("[%s] %s" % (m.get("role"), m.get("content")) for m in chain)

    rc = _retrieval_dict()
    mem_rows, note = [], ""
    if rc and not args.stack:
        q = ""
        for m in reversed(chain):
            if (m.get("role") or "").lower() == "user" and (m.get("content") or "").strip():
                q = m["content"].strip()
                break
        in_ids = {m.get("id") for m in chain if m.get("id")} | set(SYNTH)
        mem_rows, note = _retrieve_live(rc, q, in_ids, args.max_results or rc["max_results"])
    mem_txt = ""
    if mem_rows:
        lines = []
        for i, r in enumerate(mem_rows, 1):
            s = r["summary"]
            if not args.full and len(s) > rc["summary_max_chars"]:
                s = s[:rc["summary_max_chars"]].rstrip() + "…"
            lines.append("%d. [%s] %s  (%.2f meaning, %.2f keyword)" % (i, r["cap"], s, r["cos"], r["kw"]))
        mem_txt = "\n".join(lines)

    rows = [
        ("1", "Brick 2  profile.md", "system, id=%s" % SYNTH[0], prof),
        ("2", "Brick 5  prompt.md", "system, id=%s" % SYNTH[2], hon),
        ("3", "chat history (post-compact)", "%d msgs, OWU-held" % len(chain), conv_txt),
        ("3b", "Brick 0  OWU summary", "system, inserted by OWU at idx 1",
         (latest["text"] if latest else "")),
        ("4", "Brick 4  Remember", "system, appended last, id=%s" % SYNTH[1], mem_txt),
    ]
    print("-" * W)
    print("%-3s %-22s %-34s %8s" % ("#", "layer", "how it gets there", "est tok"))
    tot = 0
    for n, name, how, txt in rows:
        t = _w(txt)
        tot += t
        print("%-3s %-22s %-34s %8d" % (n, name, how[:34], t))
    print("%-3s %-22s %-34s %8d" % ("", "(sum of estimates)", "", tot))
    if cu.get("tokens"):
        print("  note: OWU says %s, my layers sum to %d (chars/4). The gap is OWU's own\n"
              "        system/template overhead plus reasoning output, which my transcript\n"
              "        count skips. Treat my per-layer numbers as relative, not exact."
              % (cu["tokens"], tot))

    if args.stack:
        print("-" * W)
        print("(--stack: skipped retrieval replay)")
        _tail(chain, W, args.tail)
        return

    # --- what the memory layer would actually say ---
    print("=" * W)
    print("BRICK 4 — what would be injected this turn")
    print("  config read from the DEPLOYED dump (%s)" % os.path.relpath(DUMP, BASE))
    if rc:
        print("  window %dd | expand on %r -> %dd | max_results %d | budget %d tok"
              % (rc["default_window_days"], rc["expand_trigger"],
                 rc["expanded_window_days"], rc["max_results"],
                 rc["max_context_tokens_for_retrieval"]))
        print("  scoring: %.2f*relevance + %.2f*salience, relevance = %.2f*meaning + %.2f*keyword"
              % (rc["relevance_weight"], rc["salience_weight"],
                 rc["relevance_blend_meaning"], rc["relevance_blend_keyword"]))
    print("  %s  (pool shown here is NOT in-context-filtered, so the dedupe check"
          % note)
    print("   below can run; the live pipe excludes those rows in SQL)")
    if not mem_rows:
        print("  (no hits replayed: %s)" % (note or "no query"))
    else:
        for i, r in enumerate(mem_rows, 1):
            flag = "  <-- id present in OWU's stored chain" if r["in_ctx"] else ""
            s = r["summary"]
            if not args.full:
                s = re.sub(r"\s+", " ", s)[:W - 14]
            print("\n  %2d. score %.3f  cos %.2f  kw %.2f  %s  %s%s"
                  % (i, r["score"], r["cos"], r["kw"], r["sal"],
                     (r["cap"] or "?"), flag))
            if args.full:
                for ln in textwrap_lines(s, W - 6):
                    print("      " + ln)
            else:
                print("      " + s)
        if args.render:
            print("=" * W)
            print("REMEMBER BLOCK — literal text the model receives (query = last user msg)")
            days = rc["default_window_days"]
            print("## Remember (retrieved, last %d days)" % days)
            smax = rc["summary_max_chars"]
            for i, r in enumerate(mem_rows, 1):
                s = re.sub(r"\s+", " ", r["summary"])
                if len(s) > smax:
                    s = s[:smax].rstrip() + "\u2026"
                kwj = ",".join(sorted({w for w in re.findall(r"[a-z0-9']+",
                                (r["keywords"] or "").lower()) if len(w) >= 4})[:5])
                print("%d. [%s] %s  (%.2f meaning, %.2f keyword) %s"
                      % (i, r["cap"], s, r["cos"], r["kw"], "kw:" + kwj if kwj else ""))

        dupes = [r for r in mem_rows if r["in_ctx"]]
        if dupes:
            print("\n  %d hit(s) whose id is in OWU's STORED chain. The live pipe dedupes"
                  % len(dupes))
            print("   against the request PAYLOAD, which this tool cannot see; if OWU")
            print("   hands the pipe a truncated window, re-injecting these is correct.")
        else:
            print("\n  dedupe check: no injected hit's id appears in the stored chain.")

    _tail(chain, W, args.tail)

    if args.log:
        print("=" * W)
        print("PIPE LOG (docker logs open-webui, last %s retrieve lines)" % args.log)
        try:
            p = subprocess.run(["docker", "logs", "--since", "12h", "open-webui"],
                               capture_output=True, text=True, timeout=60)
            both = (p.stdout or "") + "\n" + (p.stderr or "")
            hits = [ln.strip() for ln in both.splitlines() if "companion:retrieve" in ln]
            for ln in hits[-args.log:]:
                print("  " + ln)
            if not hits:
                print("  (none in the last 12h)")
        except Exception as e:
            print("  (log read failed: %s)" % e)


def _tail(chain, W, n):
    """Print the N most recent messages sitting in the live window."""
    if not n or not chain:
        return
    print("=" * W)
    print("CHAT TAIL — the last %d message(s) in the live window" % n)
    for m in chain[-n:]:
        when = _dt.datetime.fromtimestamp(m.get("timestamp") or 0).strftime("%m-%d %H:%M")
        body = re.sub(r"\s+", " ", str(m.get("content") or ""))
        print("\n  [%s] %-9s %s  (%d est tok)" % (
            when, m.get("role") or "?", (m.get("id") or "?")[:8], _w(body)))
        for ln in textwrap_lines(body, W - 4):
            print("    " + ln)
    print("\n  window spans: %s -> %s" % (
        _dt.datetime.fromtimestamp(chain[0].get("timestamp") or 0),
        _dt.datetime.fromtimestamp(chain[-1].get("timestamp") or 0)))
    print("  %d nodes on this branch; anything older than the first node is only in "
          "Jar 1 + the index." % len(chain))


def textwrap_lines(s, w):
    out, cur = [], ""
    for word in s.split():
        if len(cur) + len(word) + 1 > w and cur:
            out.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        out.append(cur)
    return out


if __name__ == "__main__":
    main()
