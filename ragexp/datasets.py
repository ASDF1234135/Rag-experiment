"""Normalised dataset format + converters.

Normalised layout (what the pipeline reads):
    <dir>/docs.jsonl       {"doc_id", "title", "text"}
    <dir>/questions.jsonl  {"qid", "question", "answer", "gold_doc_ids", "evidence_doc_ids",
                            "negative_doc_ids", "meta"}

Converters take the *already downloaded and decrypted* raw files, so nothing here touches the
network. See README for the download steps.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

from .types import Doc, Question


def _read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _read_table(path: Path) -> Iterator[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):  # stream: corpus is ~1.7 GB
            yield from batch.to_pylist()
    else:
        yield from _read_jsonl(path)


def write_normalised(out_dir: str | Path, docs: Iterable[Doc], questions: Iterable[Question]) -> tuple[int, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nd = nq = 0
    with open(out / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"doc_id": d.doc_id, "title": d.title, "text": d.text}, ensure_ascii=False) + "\n")
            nd += 1
    with open(out / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.__dict__, ensure_ascii=False) + "\n")
            nq += 1
    return nd, nq


def load_docs(path: str | Path) -> Iterator[Doc]:
    for r in _read_jsonl(Path(path)):
        yield Doc(str(r["doc_id"]), r.get("title", ""), r["text"])


def load_questions(path: str | Path) -> list[Question]:
    return [Question(**r) for r in _read_jsonl(Path(path))]


# ---------------------------------------------------------------- BrowseComp-Plus

def split_front_matter(text: str) -> tuple[str | None, str]:
    """BrowseComp-Plus documents start with a YAML-ish block (--- / title: ... / date: ... / ---).
    Returns (title, body-without-front-matter); title is None if there is no front matter."""
    if not text.startswith("---"):
        return None, text
    end = text.find("\n---", 3)
    if end < 0:
        return None, text
    title = None
    for line in text[3:end].splitlines():
        if line.startswith("title:"):
            title = line[len("title:"):].strip().strip('"')
    nl = text.find("\n", end + 1)
    return title, (text[nl + 1:] if nl >= 0 else "")


def derive_title(text: str, url: str | None, max_len: int = 150) -> str:
    """Title from the front matter; else the first short line; else URL host + path."""
    t, body = split_front_matter(text)
    if t:
        return t[:max_len]
    for line in body.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            if len(line) <= max_len:
                return line
            break
    if url:
        p = urlparse(url)
        return (p.netloc + p.path).strip("/")[:max_len]
    return ""


def convert_browsecomp_plus(queries_path: str | Path, corpus_path: str | Path, out_dir: str | Path) -> tuple[int, int]:
    """queries_path: the decrypted queries file from the official repo's decrypt script (jsonl with
    query_id, query, answer, gold_docs, evidence_docs, negative_docs; each doc {docid, url, text}).
    corpus_path: Tevatron/browsecomp-plus-corpus exported as .parquet/.jsonl (docid, url, text)."""

    def docs() -> Iterator[Doc]:
        for r in _read_table(Path(corpus_path)):
            _, body = split_front_matter(r["text"])
            yield Doc(str(r["docid"]), derive_title(r["text"], r.get("url")), body)

    def ids(lst) -> list[str]:
        return [str(d["docid"] if isinstance(d, dict) else d) for d in (lst or [])]

    qs = [
        Question(
            qid=str(r["query_id"]), question=r["query"], answer=r["answer"],
            gold_doc_ids=ids(r.get("gold_docs")), evidence_doc_ids=ids(r.get("evidence_docs")),
            negative_doc_ids=ids(r.get("negative_docs")),
        )
        for r in _read_table(Path(queries_path))
    ]
    return write_normalised(out_dir, docs(), qs)


# ---------------------------------------------------------------- RAGBench

_TITLE_PASSAGE = re.compile(r"^\s*Title:\s*(.*?)\s*\n\s*Passage:\s*(.*)$", re.S)


def split_title_passage(raw: str) -> tuple[str, str]:
    m = _TITLE_PASSAGE.match(raw)
    return (m.group(1), m.group(2)) if m else ("", raw)


def relevant_sentences(row: dict) -> list[str]:
    """Texts of the sentences RAGBench annotated as relevant (`all_relevant_sentence_keys`, e.g. "0c"),
    looked up in `documents_sentences` ([[key, text], ...] per document). These are the reference
    for completeness: RAGBench defines completeness as the share of relevant sentences a response uses."""
    by_key = {k: t for doc in (row.get("documents_sentences") or []) for k, t in doc}
    out = []
    for k in row.get("all_relevant_sentence_keys") or []:
        m = re.match(r"\s*(\d+[a-z]+)", str(k))  # some annotations carry stray punctuation, e.g. "0b."
        t = by_key.get(m.group(1)) if m else None
        if t:
            t = re.sub(r"^\s*(Passage|Title):\s*", "", t).strip()
            if t:
                out.append(t)
    return out


def convert_ragbench(files: list[str | Path], out_dir: str | Path, subset: str,
                     max_questions: int | None = None, seed: int = 0) -> dict:
    """Merge every split of one RAGBench subset into one corpus.

    RAGBench `documents` is a list of plain strings per sample with NO document id, so ids are
    content hashes (identical text -> same doc, which is the de-duplication the spec asks for).
    Gold docs = documents that hold at least one sentence in `all_relevant_sentence_keys`
    (keys look like "0a", "1c": leading digits = index into the sample's documents).
    Samples without any relevant-sentence key are dropped (no gold -> no document-level recall).
    `answer` is RAGBench's `response`, which is LLM-generated, not a human gold answer.
    """
    docs: dict[str, Doc] = {}
    qs: list[Question] = []
    dropped = dup_ids = 0
    seen_ids: set[str] = set()
    for f in files:
        for r in _read_table(Path(f)):
            raw_docs = list(r["documents"])
            ids = []
            for raw in raw_docs:
                did = "d" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
                if did not in docs:
                    title, text = split_title_passage(raw)
                    docs[did] = Doc(did, title, text)
                ids.append(did)
            gold_idx = {int(m.group(1)) for k in (r.get("all_relevant_sentence_keys") or [])
                        if (m := re.match(r"(\d+)", str(k)))}
            gold = [ids[i] for i in sorted(gold_idx) if i < len(ids)]
            if not gold:
                dropped += 1
                continue
            if r["id"] in seen_ids:  # RAGBench techqa lists every question twice; ids must stay unique
                dup_ids += 1
                continue
            seen_ids.add(r["id"])
            qs.append(Question(
                qid=f"{subset}-{r['id']}", question=r["question"], answer=r.get("response", ""),
                gold_doc_ids=list(dict.fromkeys(gold)), evidence_doc_ids=list(dict.fromkeys(gold)),
                negative_doc_ids=[i for i in dict.fromkeys(ids) if i not in gold],
                meta={"subset": subset, "reference_is_llm_generated": True,
                      "relevant_sentences": relevant_sentences(r)},
            ))
    total = len(qs)
    if max_questions and total > max_questions:  # seeded subsample; the corpus keeps every document
        import random

        qs = sorted(random.Random(seed).sample(sorted(qs, key=lambda q: q.qid), max_questions), key=lambda q: q.qid)
    nd, nq = write_normalised(out_dir, docs.values(), qs)
    return {"docs": nd, "questions": nq, "questions_before_sampling": total, "dropped_no_gold": dropped,
            "duplicate_id_rows_skipped": dup_ids}


def subset_by_pools(src_dir: str | Path, out_dir: str | Path, n_questions: int, seed: int) -> dict:
    """Seeded random sample of `n_questions`, with a corpus made of every document those questions name
    (gold + evidence + hard negatives). This is the reduced-pool variant of spec section 9 made physical:
    only that corpus needs embeddings. The result is a self-contained normalised dataset."""
    import random

    qs = load_questions(Path(src_dir) / "questions.jsonl")
    if n_questions < len(qs):
        qs = random.Random(seed).sample(sorted(qs, key=lambda q: q.qid), n_questions)
    qs = sorted(qs, key=lambda q: q.qid)
    keep = {d for q in qs for d in q.gold_doc_ids + q.evidence_doc_ids + q.negative_doc_ids}
    docs = (d for d in load_docs(Path(src_dir) / "docs.jsonl") if d.doc_id in keep)
    nd, nq = write_normalised(out_dir, docs, qs)
    missing = len(keep) - nd
    return {"questions": nq, "docs": nd, "referenced_docs_missing_from_corpus": missing}
