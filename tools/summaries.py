#!/usr/bin/env python3
"""
Read-only viewer for the companion memory index (tools/summaries.py).

Every row in index_entries is a summarizer's paraphrase of one message:
  summary    - what the exchange was about, one or two sentences
  keywords   - comma-separated tags the retriever uses for keyword scoring
  confidence - 0..1, how sure the summarizer claimed to be
  salience   - high | medium | low, how worth-remembering it judged the message
  created_at - UTC timestamp of when the summary was written

This tool never writes. It only formats.

Usage
  summaries.py                    # newest 25, newest first
  summaries.py -n 60              # newest 60
  summaries.py --chat <id>        # one chat, oldest first (reads like a transcript)
  summaries.py --search "phrase"  # rows whose summary/keywords match
  summaries.py --bad              # only rows that look broken (conf 0 / no keywords)
  summaries.py --stats            # rollup: counts by salience, confidence, batch
  summaries.py --md out.md        # dump to a Markdown file instead of the terminal
  summaries.py --all              # everything (pager-free; cap with -n)

Column layout: time is shown in LOCAL time, converted from the UTC value stored
in the DB (the log file is local, the DB is UTC, and comparing the two raw is a
trap that has already caused one false alarm).
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(BASE_DIR, "data", "record.sqlite")

# salience first: the point of the index is "what mattered", so high sorts up
ORDER = """
  CASE salience WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2
                ELSE 3 END,
  confidence DESC, created_at DESC
"""


def local(ts):
    """DB stores UTC as TEXT 'YYYY-MM-DD HH:MM:SS'; render in local time."""
    if not ts:
        return "?"
    try:
        dt = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return ts
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def short_chat(cid):
    return (cid or "?")[:8]


def rows_for(db, a):
    w, p = [], []
    if a.chat:
        w.append("(e.chat_id = ? OR e.chat_id LIKE ?)")
        p += [a.chat, a.chat + "%"]
    if a.search:
        w.append("(COALESCE(e.summary,'') LIKE ? OR COALESCE(e.keywords,'') LIKE ?)")
        p += ["%" + a.search + "%"] * 2
    if a.bad:
        w.append("(COALESCE(e.confidence,0) = 0 OR COALESCE(e.keywords,'') = '' "
                 "OR COALESCE(e.summary,'') = '')")
    if a.fresh:
        w.append(f"e.created_at >= datetime('now', '-{int(a.fresh)} hours')")
    where = ("WHERE " + " AND ".join(w)) if w else ""
    limit = "" if a.all else "LIMIT ?"
    order = "e.created_at ASC" if a.chat else ORDER
    sql = f"""
      SELECT e.created_at, e.chat_id, e.message_id, e.summary, e.keywords,
             e.confidence, e.salience, e.batch_id,
             substr(COALESCE(m.content,''), 1, 120)
      FROM index_entries e
      LEFT JOIN messages m ON m.id = e.message_id
      {where}
      ORDER BY {order}
      {limit}
    """
    args = p + ([] if a.all else [a.n])
    return db.execute(sql, args).fetchall()


def wrap(t, width, indent):
    """Wrap text, hanging the continuation lines under the field."""
    out, line, words = [], " " * indent, (t or "").split()
    for wd in words:
        if len(line) + len(wd) + 1 > width and line.strip():
            out.append(line.rstrip())
            line = " " * indent
        line += " " + wd
    if line.strip():
        out.append(line.rstrip())
    return out or [" " * indent]


def render_terminal(rows):
    W = 78
    for (ts, chat, mid, summ, kw, conf, sal, batch, orig) in rows:
        print(f"{local(ts)}  chat {short_chat(chat)}  {sal or '-':<6}  "
              f"conf {0.0 if conf is None else conf:.2f}")
        if orig:
            o = orig.replace("\n", " ").strip()
            print(f"  msg   {o[:W - 8]}" + ("…" if len(o) > W - 8 else ""))
        lines = wrap(summ or "(no summary)", W, 8)
        print("  sum  " + lines[0].strip())
        for ln in lines[1:]:
            print("       " + ln.strip())
        if kw:
            print(f"  kw    {kw}")
        print()


def render_markdown(rows, path):
    with open(path, "w") as f:
        f.write("# Companion memory index\n\n")
        f.write(f"Generated {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')} "
                f"— {len(rows)} rows. Times are local.\n\n")
        cur = None
        for (ts, chat, mid, summ, kw, conf, sal, batch, orig) in rows:
            if chat != cur:
                cur = chat
                f.write(f"\n## chat {short_chat(chat)}\n\n")
            f.write(f"### {local(ts)} — {sal or '-'} (conf "
                    f"{0.0 if conf is None else conf:.2f})\n\n")
            if orig:
                f.write(f"> {orig.replace(chr(10), ' ').strip()}\n\n")
            f.write(f"**summary** {summ or '(none)'}\n\n")
            if kw:
                f.write(f"**kw** `{kw}`\n\n")
    print(f"wrote {path}  ({len(rows)} rows)")


def stats(db):
    q = lambda s: db.execute(s).fetchall()
    print("=== index rollup")
    for k, v in q("""SELECT 'rows indexed=1', COUNT(*) FROM index_entries WHERE indexed=1
                     UNION ALL SELECT 'rows indexed=0', COUNT(*) FROM index_entries WHERE indexed=0
                     UNION ALL SELECT 'distinct chats', COUNT(DISTINCT chat_id) FROM index_entries
                     UNION ALL SELECT 'batches', COUNT(DISTINCT batch_id) FROM index_entries""" ):
        print(f"  {k:<18} {v}")
    print("\n=== by salience")
    for sal, n, conf in q("""SELECT COALESCE(salience,'(null)'), COUNT(*),
                                    ROUND(AVG(confidence),3)
                             FROM index_entries WHERE indexed=1 GROUP BY 1 ORDER BY 2 DESC"""):
        print(f"  {sal:<8} {n:>5}  avg conf {conf}")
    print("\n=== confidence buckets (indexed rows)")
    for lo, hi in [(0, 0), (0.01, 0.5), (0.51, 0.8), (0.81, 0.95), (0.96, 1)]:
        n = db.execute("SELECT COUNT(*) FROM index_entries WHERE indexed=1 "
                       "AND confidence BETWEEN ? AND ?", (lo, hi)).fetchone()[0]
        print(f"  {lo:>4}-{hi:<4} {n:>5}")
    print("\n=== empty keywords:",
          db.execute("SELECT COUNT(*) FROM index_entries WHERE indexed=1 "
                     "AND COALESCE(keywords,'')=''").fetchone()[0])
    print("=== newest batch")
    for b, n, oldest, newest in q("""SELECT substr(batch_id,1,14), COUNT(*),
                                            MIN(created_at), MAX(created_at)
                                     FROM index_entries GROUP BY batch_id
                                     ORDER BY MAX(created_at) DESC LIMIT 3"""):
        print(f"  {b}  {n:>3} rows  {oldest} .. {newest} (UTC)")


def main():
    ap = argparse.ArgumentParser(description="Human-readable view of the memory index.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("-n", type=int, default=25, help="row cap (default 25)")
    ap.add_argument("--all", action="store_true", help="no row cap")
    ap.add_argument("--chat", help="filter to one chat id (prefix ok); sorts oldest-first")
    ap.add_argument("--search", help="substring match on summary/keywords")
    ap.add_argument("--bad", action="store_true", help="only rows that look broken")
    ap.add_argument("--fresh", type=int, metavar="H", help="only rows from the last H hours")
    ap.add_argument("--stats", action="store_true", help="print the rollup and exit")
    ap.add_argument("--md", metavar="PATH", help="write Markdown to PATH instead of terminal")
    a = ap.parse_args()

    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    if a.stats:
        stats(db)
        return
    rows = rows_for(db, a)
    if not rows:
        print("no rows matched"); return
    if a.md:
        render_markdown(rows, a.md)
    else:
        render_terminal(rows)
        print(f"({len(rows)} rows"
              + ("" if a.all else f", newest {a.n}; -n/--all for more") + ")")


if __name__ == "__main__":
    main()
