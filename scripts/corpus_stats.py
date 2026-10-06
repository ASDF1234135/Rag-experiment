# usage: python -m scripts.corpus_stats <dataset>   (run from repo root; chunks data/<dataset> into work/<dataset>)
import sys, json, time, statistics as st
from pathlib import Path
from ragexp.config import Config
from ragexp.chunking import get_encoding
from ragexp.datasets import load_docs, load_questions
from ragexp.pipeline import build_store
from ragexp.runner import Workspace
from ragexp.store import ChunkStore

ds = sys.argv[1]
cfg = Config.load("configs/experiment.yaml")
ws = Workspace(cfg, ds)
enc = get_encoding(cfg.chunk.encoding)
store = ChunkStore(ws.chunks)
t = time.time()
n = build_store(load_docs(Path("data")/ds/"docs.jsonl"), store, enc, cfg.chunk)
print(f"{ds}: chunked {n} new docs in {time.time()-t:.0f}s; chunks={len(store)}")
tot = 0; ser = 0; cnt = 0
for b in store.iter_chunks(batch=20000):
    for c in b:
        tot += c.n_tokens; cnt += 1
        ser += c.n_tokens + len(enc.encode(c.title, disallowed_special=())) + 8
print(f"body tokens {tot:,}; embed tokens (title+body) ~{ser:,}; mean/chunk {ser/cnt:.0f}")
print(f"embedding cost @ $0.02/M ~ ${ser*0.02/1e6:.2f}")
print(f"G5 full-scan per query-round @ $0.042/M ~ ${(ser+cnt*60)*0.042/1e6:,.2f}, >= {(ser+cnt*60)/1e5/3600:.1f} h at 100K tok/s")
qs = load_questions(Path("data")/ds/"questions.jsonl")
print("questions", len(qs), "gold/q", round(st.mean(len(q.gold_doc_ids) for q in qs),2), "evidence/q", round(st.mean(len(q.evidence_doc_ids) for q in qs),2), "neg/q", round(st.mean(len(q.negative_doc_ids) for q in qs),2))
pool = [len({c for d in set(q.gold_doc_ids+q.evidence_doc_ids+q.negative_doc_ids) for c in store.doc_chunk_ids(d)}) for q in qs[:100]]
print("chunks in gold+evidence+negative docs per q (first 100): mean", round(st.mean(pool)), "max", max(pool))
