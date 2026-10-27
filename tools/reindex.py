#!/usr/bin/env python3
"""Re-index bad index_entries rows by clearing them so housekeeping re-claims them.

Why this works: housekeeping.claim_batch() selects messages that are NOT in
(SELECT message_id FROM index_entries WHERE indexed = 1). Deleting an indexed
row therefore puts its message back in the queue. process_batch() re-embeds and
re-inserts in the same call, so there is no half-state.

Bad-row definition (from the Oct 24 audit, verified against data/record.sqlite):
  keywords = ''  OR  confidence = 0
Those are the fingerprints of two failure modes, both fixed now:
  * the old general-purpose fallback that ignored the four-line label protocol
  * rows written by _bump_and_maybe_skip() as "(skipped: model unavailable)"
Real summaries from the current protocol carry keywords and non-zero confidence.

Read-only unless --apply is passed. Nothing here touches the messages table.

  .venv/bin/python tools/reindex.py                 # dry run: what would clear
  .venv/bin/python tools/reindex.py --stats         # quality breakdown
  .venv/bin/python tools/reindex.py --apply         # clear + audit log
  .venv/bin/python tools/reindex.py --apply --limit 40   # staged pilot
"""
import argparse
import datetime
import os
import sqlite3
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "data", "record.sqlite")
LOG = os.path.join(BASE, "logs", "reindex.log")

BAD = "(keywords IS NULL OR keywords = '' OR confidence = 0)"


def stats(db):
    print("id column is NULL in %d/%d rows -> deleting by rowid, not id" %
          (db.execute("select count(*) from index_entries where id is null").fetchone()[0],
           db.execute("select count(*) from index_entries").fetchone()[0]))
    q = lambda s: db.execute(s).fetchone()[0]
    total = q("select count(*) from index_entries where indexed=1")
    bad = q("select count(*) from index_entries where indexed=1 and %s" % BAD)
    print("indexed rows ......... %d" % total)
    print("  bad (to re-index) .. %d" % bad)
    print("  good ............... %d" % (total - bad))
    print("  empty keywords ..... %d" %
          q("select count(*) from index_entries where indexed=1 and (keywords is null or keywords='')"))
    print("  conf = 0 ........... %d" %
          q("select count(*) from index_entries where indexed=1 and confidence=0"))
    print("  skipped sentinel ... %d" %
          q("select count(*) from index_entries where indexed=1 and summary like '%skipped: model unavailable%'"))
    # Attribution audit: the summarizer used to get no role tag, so every turn read
    # as if the human did it. A row is suspect when the summarized turn was the
    # assistant's but the summary credits the user (or the reverse).
    # Two different bugs, counted separately. The error only ever runs one way per
    # bug, which is what makes them separable:
    #   wrong actor   - assistant turn credited to 'the user' (no role tag was sent)
    #   wrong subject - user turn summarized as what the *reply* did (scope drift)
    join = ("select count(*) from index_entries e "
            "join messages m on m.id = e.message_id "
            "where e.indexed=1 and m.role=")
    print("  mis-attributed ..... %d" % (
        q(join + "'assistant' and (e.summary like 'The user%' "
                "or e.summary like 'User %')")
        + q(join + "'user' and (e.summary like 'The assistant%' "
                  "or e.summary like 'Assistant%')")))
    print("    wrong actor ...... %d" % q(
        join + "'assistant' and (e.summary like 'The user%' "
               "or e.summary like 'User %')"))
    print("    wrong subject .... %d" % q(
        join + "'user' and (e.summary like 'The assistant%' "
               "or e.summary like 'Assistant%')"))
    print("  messages never indexed %d" %
          q("select count(*) from messages m where m.role in ('user','assistant') "
            "and m.content is not null and length(m.content)>=3 "
            "and m.id not in (select message_id from index_entries where indexed=1)"))
    print("\nbad rows by month:")
    for m, n in db.execute("select substr(created_at,1,7), count(*) from index_entries "
                           "where indexed=1 and %s group by 1 order by 1" % BAD):
        print("   %s  %d" % (m, n))
    print("\nsample bad summaries:")
    for s in db.execute("select summary from index_entries where indexed=1 and %s "
                        "order by created_at desc limit 4" % BAD):
        print("   %r" % (s[0][:90],))
    print("\nsample good summaries:")
    for s in db.execute("select summary from index_entries where indexed=1 and not %s "
                        "order by created_at desc limit 3" % BAD):
        print("   %r" % (s[0][:90],))


def apply_clear(db, limit, all_rows=False):
    # all_rows=True clears every indexed row regardless of quality. Used when the
    # summarizer prompt itself changed, so old output is stale even when it parsed
    # cleanly (e.g. the role-attribution fix: good format, wrong actor).
    # NOTE: the id column is NULL in every row (it is never populated on insert),
    # so id=? matches nothing. rowid is the only reliable key here.
    where = "indexed=1" if all_rows else "indexed=1 and %s" % BAD
    rows = db.execute("select rowid, message_id, chat_id, substr(summary,1,120), confidence, "
                      "created_at from index_entries where %s "
                      "order by created_at asc limit ?" % where, (limit,)).fetchall()
    if not rows:
        print("no bad rows to clear")
        return 0
    dupes = len(rows) - len({r[1] for r in rows})
    if dupes:
        print("note: %d message_ids appear more than once" % dupes)
    ids = [r[0] for r in rows]
    db.executemany("delete from index_entries where rowid=?", [(i,) for i in ids])
    db.commit()

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a") as fh:
        fh.write("%s cleared %d rows (reason: %s)\n"
                 % (stamp, len(rows),
                    "full rebuild after prompt change" if all_rows else "bad-summary audit"))
        for r in rows:
            fh.write("  id=%s msg=%s chat=%s conf=%s created=%s | %s\n"
                     % (r[0], r[1], r[2], r[4], r[5], (r[3] or "").replace("\n", " ")))

    left = db.execute("select count(*) from index_entries where indexed=1 and %s" % BAD).fetchone()[0]
    todo = db.execute("select count(*) from messages m where m.role in ('user','assistant') "
                      "and m.content is not null and length(m.content)>=3 "
                      "and m.id not in (select message_id from index_entries where indexed=1)").fetchone()[0]
    print("cleared %d rows -> %d queued for housekeeping, %d bad rows remain" % (len(rows), todo, left))
    print("audit log: %s" % LOG)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default is dry run)")
    ap.add_argument("--stats", action="store_true", help="print quality breakdown")
    ap.add_argument("--limit", type=int, default=100000, help="max rows to clear")
    ap.add_argument("--all", action="store_true",
                    help="clear ALL indexed rows, not only malformed ones "
                         "(use after a summarizer-prompt change)")
    ap.add_argument("--db", default=DB)
    a = ap.parse_args()

    if not os.path.exists(a.db):
        sys.exit("no db at %s" % a.db)
    db = sqlite3.connect(a.db, timeout=20)
    try:
        if a.stats or not a.apply:
            stats(db)
            if not a.apply:
                print("\ndry run. pass --apply to clear these rows.")
        else:
            apply_clear(db, a.limit, all_rows=a.all)
    finally:
        db.close()


if __name__ == "__main__":
    main()
