import json
from types import SimpleNamespace

import pytest

from ragexp import datasets
from ragexp.calibrate import agreement
from ragexp.config import PricingCfg
from ragexp.judges import judge_completeness, judge_faithfulness, parse_json_object
from ragexp.report import contrasts, summarize
from ragexp.runner import grade_row
from ragexp.types import Chunk, Question


class Scripted:
    """Stand-in judge LLM: returns canned replies in order and records the prompts."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, **kw):
        self.prompts.append(messages)
        return SimpleNamespace(content=self.replies.pop(0), meta=None)


def test_parse_json_object_tolerates_chatter_and_nesting():
    reply = 'Sure!\n```json\n{"claims": [{"claim": "a", "supported": true}]}\n```'
    assert parse_json_object(reply) == {"claims": [{"claim": "a", "supported": True}]}
    assert parse_json_object("no json here") is None
    assert parse_json_object('{"bad": ') is None
    assert parse_json_object('{oops} then {"covered": [1]}') == {"covered": [1]}


def test_faithfulness_scoring():
    j = Scripted('{"claims": [{"claim": "x", "supported": true}, {"claim": "y", "supported": false}, '
                 '{"claim": "z", "supported": true}]}')
    r = judge_faithfulness(j, "evidence text", "an answer")
    assert r["faithfulness"] == pytest.approx(2 / 3) and r["fully_faithful"] is False and r["n_claims"] == 3
    assert "evidence text" in j.prompts[0][1]["content"] and "an answer" in j.prompts[0][1]["content"]
    # abstention (no factual claims) is excluded, not scored as "perfectly faithful"
    r = judge_faithfulness(Scripted('{"claims": []}'), "e", "The evidence is insufficient.")
    assert r["faithfulness"] is None and r["fully_faithful"] is None and r["n_claims"] == 0 and r["parse_ok"]
    # an unparsable reply is flagged, not silently scored
    r = judge_faithfulness(Scripted("I cannot do that"), "e", "a")
    assert r["faithfulness"] is None and not r["parse_ok"]
    ok = judge_faithfulness(Scripted('{"claims": [{"claim": "x", "supported": true}]}'), "e", "a")
    assert ok["fully_faithful"] is True


def test_completeness_scoring():
    j = Scripted('{"covered": [1, 3, 3, 9, "2", 2.0]}')  # duplicates / out-of-range ids must not inflate the score
    r = judge_completeness(j, "q", "ans", ["s1", "s2", "s3", "s4"])
    assert r["completeness"] == pytest.approx(3 / 4) and r["n_key"] == 4
    assert "1. s1" in j.prompts[0][1]["content"] and "4. s4" in j.prompts[0][1]["content"]
    assert judge_completeness(Scripted("{}"), "q", "a", ["s1"])["parse_ok"] is False
    assert judge_completeness(Scripted(), "q", "a", [])["completeness"] is None  # nothing annotated: no call


def test_relevant_sentences_and_converter_meta(tmp_path):
    row = {"id": "1", "question": "Q", "response": "R",
           "documents": ["Title: A\nPassage: alpha beta", "gamma"],
           "documents_sentences": [[["0a", "Title: A"], ["0b", "Passage: alpha."], ["0c", "beta."]], [["1a", "gamma."]]],
           "all_relevant_sentence_keys": ["0b", "1a", "9z"]}
    assert datasets.relevant_sentences(row) == ["alpha.", "gamma."]  # prefix stripped, unknown key skipped
    f = tmp_path / "raw.jsonl"
    f.write_text(json.dumps(row), encoding="utf-8")
    datasets.convert_ragbench([f], tmp_path / "o", "x")
    q = datasets.load_questions(tmp_path / "o" / "questions.jsonl")[0]
    assert q.meta["relevant_sentences"] == ["alpha.", "gamma."]


class Store:
    def get(self, cid):
        return Chunk(cid, "d", "T", f"text of {cid}", 3)


def test_grade_row_dispatches_by_dataset_and_is_idempotent():
    rb_q = Question("q1", "Q?", "llm ref", ["d"], meta={"relevant_sentences": ["s1", "s2"]})
    row = {"answer": "A", "evidence_chunk_ids": ["c1", "c2"]}
    j = Scripted('{"claims": [{"claim": "a", "supported": true}]}', '{"covered": [1]}')
    out = grade_row(j, rb_q, row, Store())
    assert out["faithfulness"] == 1.0 and out["fully_faithful"] is True and out["completeness"] == 0.5
    assert out.get("correct") is None  # never graded against RAGBench's LLM-generated reference
    assert "text of c1" in j.prompts[0][1]["content"] and "[c2] T" in j.prompts[0][1]["content"]
    n_calls = len(j.prompts)
    grade_row(j, rb_q, out, Store())  # re-grading must not call the judge again
    assert len(j.prompts) == n_calls
    ab = grade_row(Scripted('{"claims": []}', '{"covered": []}'), rb_q,
                   {"answer": "insufficient", "evidence_chunk_ids": []}, Store())
    assert ab["n_claims"] == 0 and ab["completeness"] == 0.0
    # gold-answer datasets still use correctness
    bc_q = Question("q2", "Q?", "Paris", ["d"])
    r2 = grade_row(Scripted('{"correct": true}'), bc_q, {"answer": "Paris."}, None)
    assert r2["correct"] is True and "faithfulness" not in r2


def _rows():
    def row(group, qid, ff, fa, co, stop="judge", covered=False, rnd=2):
        comp = {"input_tokens": 1000, "output_tokens": 0, "reasoning_tokens": 0,
                "latency_ms": 1.0, "retries": 0, "n_calls": 1}
        return {"group": group, "qid": qid, "replicate": 0, "stop_round": rnd, "stop_reason": stop, "answer": "x",
                "evidence_chunk_ids": [], "evidence_doc_ids": ["d1"], "covered_all": covered, "covered_any": True,
                "first_round_doc_ids": ["d1"], "first_round_ranking_doc_ids": ["d1"], "latency_ms": 5.0,
                "components": {"jev_judge": comp},
                "faithfulness": fa, "fully_faithful": ff, "n_claims": 0 if fa is None else 2, "completeness": co}

    rows = []
    for i in range(12):
        rows.append(row("G3", f"q{i}", True, 1.0, 0.8))
        rows.append(row("G4", f"q{i}", i % 2 == 0, 1.0 if i % 2 == 0 else 0.5, 0.6))
    rows.append(row("G1", "q0", None, None, 0.0, stop="answered", rnd=1))
    return rows


def test_report_summarize_and_contrasts_for_ragbench_metrics():
    qs = {f"q{i}": Question(f"q{i}", "Q", "A", ["d1"]) for i in range(12)}
    s = {r["group"]: r for r in summarize(_rows(), qs, PricingCfg(), 6)}
    assert s["G3"]["fully_faithful"] == 1.0 and s["G3"]["completeness"] == pytest.approx(0.8)
    assert s["G4"]["fully_faithful"] == 0.5 and s["G4"]["faithfulness"] == pytest.approx(0.75)
    assert s["G1"]["faithfulness"] is None and s["G1"]["abstain_rate"] == 1.0  # abstaining is visible
    assert s["G3"]["accuracy"] is None  # no gold answer on RAGBench
    assert s["G3"]["cost_usd_per_faithful"] == pytest.approx(1000 * 0.042 / 1e6)  # 12 rows x $4.2e-5 / 12 faithful
    assert s["G3"]["premature_stop"] == 1.0  # judge-stop with gold docs not all covered
    c = {(x["a"], x["b"]): x for x in contrasts(_rows(), PricingCfg())}[("G3", "G4")]
    assert set(c["metrics"]) == {"fully_faithful", "faithfulness", "completeness"}
    assert c["metrics"]["fully_faithful"]["mcnemar"]["a_only"] == 6
    assert c["metrics"]["completeness"]["diff"]["mean_diff"] == pytest.approx(0.2)
    assert "mcnemar" not in c["metrics"]["completeness"]


def test_calibration_agreement():
    res = []
    for i in range(20):
        ann = i % 2 == 0
        res.append({"id": i, "ann_adherence": ann, "ann_completeness": 0.1 * (i % 10), "has_keys": True,
                    "faithfulness": 1.0 if ann else 0.4, "fully_faithful": ann, "n_claims": 3,
                    "completeness": 0.1 * (i % 10), "parse_ok": True})
    a = agreement(res)
    assert a["adherence"]["accuracy"] == 1.0 and a["adherence"]["kappa"] == 1.0
    assert a["adherence"]["auc_fraction_supported"] == 1.0
    assert a["completeness"]["mae"] == pytest.approx(0.0) and a["completeness"]["pearson"] == pytest.approx(1.0)
    for r in res:  # a judge that always says "faithful" has chance-level agreement
        r["fully_faithful"], r["faithfulness"] = True, 1.0
    assert agreement(res)["adherence"]["kappa"] == pytest.approx(0.0)


def test_relevant_sentence_keys_with_stray_punctuation():
    row = {"documents_sentences": [[["0a", "x"], ["0b", "second"]]], "all_relevant_sentence_keys": ["0b."]}
    assert datasets.relevant_sentences(row) == ["second"]


def test_glm_cost_counts_reasoning_tokens_once():
    from ragexp.report import row_cost

    row = {"components": {"glm_answer": {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
                                         "reasoning_tokens": 900_000, "latency_ms": 0, "retries": 0, "n_calls": 1}}}
    p = PricingCfg(glm_input=0.15, glm_output=0.5)
    assert row_cost(row, p) == pytest.approx(0.15 + 0.5)  # reasoning is a subset of completion tokens
