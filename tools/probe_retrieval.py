"""Offline probe of the Brick 4 retrieval path.

Replicates _retrieve() scoring from data/_deployed_dump/companion_pipe.py
(same RETRIEVAL weights, same hybrid blend) and runs hand-picked queries
against the live index, so ranking quality can be judged without going
through Open WebUI. Read-only.
"""
import glob
import os, sys, sqlite3
import numpy as np

DATA = os.environ.get(
    "COMPANION_DATA", os.path.expanduser("~/companion/data"))
R = {"default_window_days": 7, "max_context_tokens_for_retrieval": 4000,
     "relevance_weight": 0.7, "salience_weight": 0.3,
     "relevance_blend_meaning": 0.5, "relevance_blend_keyword": 0.5,
     "salience": {"high": 1.0, "medium": 0.5, "low": 0.2},
     "max_results": 10, "summary_max_chars": 220, "keyword_min_chars": 4}

os.environ.setdefault("HF_HOME", os.path.join(DATA, "hfcache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from sentence_transformers import SentenceTransformer
_repo = os.path.join(DATA, "hfcache", "models--BAAI--bge-small-en-v1.5")
_snap = sorted(glob.glob(os.path.join(_repo, "snapshots", "*")))
if not _snap:
    sys.exit("no bge snapshot in cache")
m = SentenceTransformer(_snap[-1], local_files_only=True)

STOP = set("""the a an and or but if then than that this these those of to in on for with
without is are was were be been being do does did doing have has had having i you your our
their it its as at by from so not no yes can could should would will just about how what
which who whom when where why all any some more most other such into over under again""".split())

def words(t):
    return {w for w in __import__("re").findall(r"[a-z0-9]+", (t or "").lower())
            if len(w) >= R["keyword_min_chars"] and w not in STOP}

con = sqlite3.connect("file:%s?mode=ro" % os.path.join(DATA, "record.sqlite"), uri=True)
rows = con.execute(
    "SELECT ie.summary, ie.keywords, ie.salience, ie.vectors, m.captured_at, m.role "
    "FROM index_entries ie JOIN messages m ON m.id = ie.message_id "
    "WHERE ie.indexed=1 AND m.captured_at >= datetime('now','-7 days')").fetchall()
con.close()

MATS = np.stack([np.frombuffer(b, dtype=np.float32) for (_s, _k, _sal, b, _c, _r) in rows
                 if isinstance(b, (bytes, bytearray)) and len(b) == 384 * 4])
META = [r for r in rows if isinstance(r[3], (bytes, bytearray)) and len(r[3]) == 384 * 4]
print("pool: %d rows (embedded %d)" % (len(rows), MATS.shape[0]))

QUERIES = [
    ("attribution",  "the summarizer kept attributing my messages to the assistant"),
    ("deploy-drift", "which copy of the pipe is actually deployed in open webui"),
    ("compactor",    "the 4B compactor model pinned on port 8097 cpu only"),
    ("orphans",      "move relegated files to the Orphaned folder"),
    ("budget",       "is the retrieval token budget too large for the context window"),
]

for name, q in QUERIES:
    qv = m.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
    qw = words(q)
    scored = []
    for vec, (summ, kws, sal, _b, cap, role) in zip(MATS, META):
        cos = float(np.clip(np.dot(qv, vec), 0.0, 1.0))
        hay = words(" ".join(x for x in (summ, kws) if x))
        kw = len([w for w in qw if w in hay]) / len(qw) if qw and hay else 0.0
        rel = R["relevance_blend_meaning"] * cos + R["relevance_blend_keyword"] * kw
        sc = R["relevance_weight"] * rel + R["salience_weight"] * R["salience"].get(sal, 0.2)
        scored.append((sc, cap, cos, kw, sal, role, summ))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    print("\n### %s\n    query: %s" % (name, q))
    for i, (sc, cap, cos, kw, sal, role, summ) in enumerate(scored[:5], 1):
        print("  %d. %.3f (m%.2f k%.2f %s/%s) [%s] %s" % (
            i, sc, cos, kw, sal, role[0], cap, (summ or "").strip()[:110].replace("\n", " ")))
