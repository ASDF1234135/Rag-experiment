import json

from ragexp import datasets
from ragexp.cli import main
from ragexp.report import load_results


def test_ragbench_converter(tmp_path):
    rows = [
        {"id": "1", "question": "Q1?", "response": "R1",
         "documents": ["Title: A\nPassage: alpha text", "Title: B\nPassage: beta text", "no title here"],
         "all_relevant_sentence_keys": ["0a", "0b", "2c"]},
        {"id": "2", "question": "Q2?", "response": "R2",
         "documents": ["Title: A\nPassage: alpha text", "gamma"], "all_relevant_sentence_keys": ["0a"]},
        {"id": "3", "question": "Q3?", "response": "R3", "documents": ["x"], "all_relevant_sentence_keys": []},
    ]
    f = tmp_path / "raw.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    stats = datasets.convert_ragbench([f], tmp_path / "out", "covidqa")
    assert stats == {"docs": 5, "questions": 2, "questions_before_sampling": 2, "dropped_no_gold": 1,
                     "duplicate_id_rows_skipped": 0}  # dup "alpha text" merged; dropped sample keeps its doc as distractor
    qs = datasets.load_questions(tmp_path / "out" / "questions.jsonl")
    docs = {d.doc_id: d for d in datasets.load_docs(tmp_path / "out" / "docs.jsonl")}
    assert len(qs[0].gold_doc_ids) == 2 and qs[0].gold_doc_ids[0] == qs[1].gold_doc_ids[0]
    assert docs[qs[0].gold_doc_ids[0]].title == "A" and docs[qs[0].gold_doc_ids[0]].text == "alpha text"
    assert qs[0].meta["reference_is_llm_generated"]


def test_browsecomp_converter_and_titles(tmp_path):
    corpus = [{"docid": "1", "url": "https://en.example.org/wiki/Foo", "text": "---\ntitle: Foo Title\ndate: 2002-05-13\n---\nbody body"},
              {"docid": "2", "url": "https://x.org/a/b", "text": "x" * 400}]
    queries = [{"query_id": "7", "query": "q", "answer": "a", "gold_docs": [{"docid": "1"}],
                "evidence_docs": [{"docid": "1"}, {"docid": "2"}], "negative_docs": []}]
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(r) for r in corpus), encoding="utf-8")
    (tmp_path / "q.jsonl").write_text("\n".join(json.dumps(r) for r in queries), encoding="utf-8")
    assert datasets.convert_browsecomp_plus(tmp_path / "q.jsonl", tmp_path / "c.jsonl", tmp_path / "o") == (2, 1)
    docs = {d.doc_id: d for d in datasets.load_docs(tmp_path / "o" / "docs.jsonl")}
    assert docs["1"].title == "Foo Title" and docs["1"].text == "body body" and docs["2"].title == "x.org/a/b"
    q = datasets.load_questions(tmp_path / "o" / "questions.jsonl")[0]
    assert q.gold_doc_ids == ["1"] and q.evidence_doc_ids == ["1", "2"]


def test_demo_runs_end_to_end(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    assert "| G5 |" in out and "artifacts in" in out


def test_subset_by_pools(tmp_path):
    docs = [datasets.Doc(str(i), "", f"text {i}") for i in range(10)]
    qs = [datasets.Question(f"q{i}", f"Q{i}?", "a", gold_doc_ids=[str(i)], evidence_doc_ids=[str(i)],
                            negative_doc_ids=[str(5 + i)]) for i in range(5)]
    datasets.write_normalised(tmp_path / "src", docs, qs)
    stats = datasets.subset_by_pools(tmp_path / "src", tmp_path / "sub", 2, seed=3)
    sub_q = datasets.load_questions(tmp_path / "sub" / "questions.jsonl")
    sub_d = {d.doc_id for d in datasets.load_docs(tmp_path / "sub" / "docs.jsonl")}
    assert stats["questions"] == 2 and stats["referenced_docs_missing_from_corpus"] == 0
    assert sub_d == {d for q in sub_q for d in q.gold_doc_ids + q.negative_doc_ids}  # exactly the pool docs
    again = datasets.subset_by_pools(tmp_path / "src", tmp_path / "sub2", 2, seed=3)
    assert [q.qid for q in datasets.load_questions(tmp_path / "sub2" / "questions.jsonl")] == [q.qid for q in sub_q]


def test_ragbench_duplicate_ids_are_collapsed(tmp_path):
    row = {"id": "techqa_DEV_Q1", "question": "Q?", "response": "R", "documents": ["doc one", "doc two"],
           "all_relevant_sentence_keys": ["0a"], "documents_sentences": [[["0a", "doc one"]], [["1a", "doc two"]]]}
    other = dict(row, id="techqa_DEV_Q2", question="Q2?")
    f = tmp_path / "raw.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in (row, dict(row), other, dict(other))), encoding="utf-8")
    stats = datasets.convert_ragbench([f], tmp_path / "o", "techqa")
    qs = datasets.load_questions(tmp_path / "o" / "questions.jsonl")
    assert stats["questions"] == 2 and stats["duplicate_id_rows_skipped"] == 2
    assert len({q.qid for q in qs}) == 2
