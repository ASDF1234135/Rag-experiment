import json

import numpy as np
import pytest

from ragexp import metrics
from ragexp.cache import ApiCache
from ragexp.chunking import chunk_document, get_encoding
from ragexp.clients import ApiError, CallMeta, GLMClient, JevClient, call_with_retries
from ragexp.config import ChunkCfg, GLMCfg, JevCfg
from ragexp.evidence import Evidence, coverage
from ragexp.groups import run_g1, run_g2, run_g5, run_trajectory_groups
from ragexp.judges import parse_json_flag
from ragexp.pipeline import build_pool, split_dev_test
from ragexp.recorder import FIELDS, Recorder
from ragexp.simulate import group_calls, stop_round
from ragexp.trajectory import Trajectory, run_trajectory
from ragexp.types import Doc, Hit


def test_chunking_window_and_overlap():
    enc = get_encoding("cl100k_base")
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_document(Doc("d", "T", text), enc, ChunkCfg())
    ids = [enc.encode(c.text) for c in chunks]
    assert all(c.n_tokens <= 512 for c in chunks)
    assert len(chunks) > 3
    # consecutive windows share 64 tokens
    assert ids[0][-64:] == ids[1][:64]
    # last window reaches the end of the document
    assert enc.decode(enc.encode(text)[-10:]) in chunks[-1].text
    assert chunks[0].chunk_id == "d#0" and chunks[1].chunk_id == "d#1"


def test_chunking_short_doc_single_chunk():
    enc = get_encoding("cl100k_base")
    assert len(chunk_document(Doc("d", "T", "hello world"), enc, ChunkCfg())) == 1
    assert chunk_document(Doc("d", "T", ""), enc, ChunkCfg()) == []


def test_evidence_dedupe_best_score_and_cap(world):
    env, qs, docs, _ = world
    ids = [c.chunk_id for b in env.store.iter_chunks() for c in b][:6]
    ev = Evidence(env.store, env.enc, e_max=10_000)
    ev.add([Hit(ids[0], "x", 0.1), Hit(ids[1], "x", 0.9)])
    ev.add([Hit(ids[0], "x", 0.5)])  # same chunk, better score -> upgraded, not duplicated
    assert ev.chunk_ids == [ids[1], ids[0]]
    one = ev.n_tokens // 2
    small = Evidence(env.store, env.enc, e_max=one + 5)  # room for exactly one chunk-ish
    small.add([Hit(ids[0], "x", 0.1), Hit(ids[1], "x", 0.9)])
    assert small.chunk_ids == [ids[1]]
    assert small.n_tokens <= one + 5
    # a displaced chunk comes back if it is re-retrieved with a higher score
    small.add([Hit(ids[0], "x", 0.99)])
    assert small.chunk_ids == [ids[0]]
    assert set(small.seen_chunk_ids) == {ids[0], ids[1]}


def test_coverage():
    assert coverage(["a", "b", "c"], ["a", "b"]) == (True, True)
    assert coverage(["a"], ["a", "b"]) == (False, True)
    assert coverage(["z"], ["a"]) == (False, False)
    assert coverage(["a"], []) == (False, False)


def test_trajectory_two_hop_and_offline_stop(world):
    env, qs, *_ = world
    q = qs[0]
    traj = run_trajectory(q, "toy", 0, env.emb_retriever, env.store, env.enc, env.cfg.search, env.qgen,
                          env.llm_judge, env.jev_judge)
    assert len(traj.rounds) == env.cfg.search.r_max
    # round 1 finds doc_a (entity in question), a later round finds doc_b via the bridge query
    assert not traj.rounds[0].covered_all and traj.rounds[0].covered_any
    assert any(r.covered_all for r in traj.rounds)
    first_cov = next(r.r for r in traj.rounds if r.covered_all)
    assert stop_round(traj, "G7") == (first_cov, "judge")
    assert stop_round(traj, "G3")[1] == "judge" and stop_round(traj, "G4", tau=0.5)[1] == "judge"
    assert stop_round(traj, "G4", tau=0.99) == (4, "r_max")
    assert stop_round(traj, "G6", r_fixed=2) == (2, "fixed")
    # round trip through JSON (trajectories are stored on disk)
    again = Trajectory.from_dict(json.loads(json.dumps(traj.to_dict())))
    assert stop_round(again, "G4", tau=0.5) == stop_round(traj, "G4", tau=0.5)


def test_group_pays_only_until_its_own_stop(world):
    env, qs, *_ = world
    traj = run_trajectory(qs[0], "toy", 0, env.emb_retriever, env.store, env.enc, env.cfg.search, env.qgen,
                          env.llm_judge, env.jev_judge)
    stop_r, _ = stop_round(traj, "G4", tau=0.5)
    calls = group_calls(traj, "G4", stop_r)
    assert max(r for r, *_ in calls) == stop_r
    comps = [c for _, c, _ in calls]
    assert comps.count("jev_judge") == stop_r and "glm_judge" not in comps
    assert comps.count("glm_query_gen") == stop_r - 1  # no query is written after the stopping round
    g3 = group_calls(traj, "G3", stop_round(traj, "G3")[0])
    assert "jev_judge" not in [c for _, c, _ in g3]
    g6 = [c for _, c, _ in group_calls(traj, "G6", 2)]
    assert "jev_judge" not in g6 and "glm_judge" not in g6


def test_all_groups_end_to_end_and_records_complete(world):
    env, qs, *_ = world
    q = qs[1]
    rows = [run_g1(env, q, "toy", 0)]
    rows.append(run_g2(env, q, "toy", 0))
    traj = run_trajectory(q, "toy", 0, env.emb_retriever, env.store, env.enc, env.cfg.search, env.qgen,
                          env.llm_judge, env.jev_judge)
    rows += run_trajectory_groups(env, q, traj, tau=0.5, r_fixed=2)
    pool = build_pool(q, env.store, pool_size=40, seed=1)
    rows.append(run_g5(env, q, "toy", 0, tau=0.5, pool=pool))
    by = {r["group"]: r for r in rows}
    assert set(by) == {"G1", "G2", "G3", "G4", "G5", "G6", "G7"}
    # G1 only sees the first hop -> cannot answer a two-hop question; multi-round groups can
    assert "ANSWER" not in by["G1"]["answer"]
    for g in ("G2", "G3", "G4", "G5", "G7"):
        assert q.answer in by[g]["answer"], g
    assert by["G7"]["stop_round"] <= by["G4"]["stop_round"]
    assert by["G6"]["stop_round"] == 2 and by["G6"]["stop_reason"] == "fixed"
    assert by["G5"]["components"]["jev_rank"]["n_calls"] == by["G5"]["stop_round"] * len(pool)
    assert "glm_judge" in by["G3"]["components"] and "jev_judge" in by["G4"]["components"]
    # every record has the full schema, and components are only the spec's names
    recs = env.recorder.rows
    assert recs and all(set(r) == set(FIELDS) for r in recs)
    assert {r["group"] for r in recs} == set(by)
    answers = [r for r in recs if r["component"] == "glm_answer"]
    assert all(r["stop_reason"] in {"judge", "r_max", "answered", "fixed"} for r in answers)


def test_replicates_get_independent_cache_entries(tmp_path):
    cache = ApiCache(tmp_path / "c.sqlite")
    calls = []

    class Raw:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    from types import SimpleNamespace as N
                    msg = N(content=f"sample{len(calls)}", tool_calls=None)
                    usage = N(prompt_tokens=10, completion_tokens=5,
                              completion_tokens_details=N(reasoning_tokens=3))
                    return N(choices=[N(message=msg, finish_reason="stop")], usage=usage)

    cli = GLMClient(GLMCfg(provider="p", model="m"), cache=cache, client=Raw())
    m = [{"role": "user", "content": "hi"}]
    a = cli.chat(m, salt="rep0")
    b = cli.chat(m, salt="rep1")
    a2 = cli.chat(m, salt="rep0")
    assert a.content != b.content and a2.content == a.content and len(calls) == 2
    assert a.meta.reasoning_tokens == 3 and not a.meta.cached and a2.meta.cached
    assert a2.meta.input_tokens == 10  # cached call keeps the original token accounting


def test_jev_client_retries_counts_and_cache(tmp_path, monkeypatch):
    import httpx

    seen = {"n": 0}

    def handler(req: httpx.Request):
        seen["n"] += 1
        if seen["n"] <= 2:
            return httpx.Response(429, text="slow down")
        body = json.loads(req.content)
        assert body["model"] == "jev-1.13.0" and req.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={
            "model": body["model"],
            "answers": {"s": {"type": "noul", "noul": 0.7}},
            "usage": {"input_tokens": 100, "output_tokens": 4},
        })

    http = httpx.Client(base_url="https://api.typesafe.ai", transport=httpx.MockTransport(handler),
                        headers={"Authorization": "Bearer k"})
    monkeypatch.setattr("time.sleep", lambda s: None)
    jev = JevClient(JevCfg(), cache=ApiCache(tmp_path / "j.sqlite"), http=http)
    r = jev.ask({"question": "q", "evidence": []}, {"s": {"type": "noul", "instructions": "x"}})
    assert r.answers["s"]["noul"] == 0.7
    assert (r.meta.input_tokens, r.meta.output_tokens, r.meta.retries) == (100, 4, 2)
    r2 = jev.ask({"question": "q", "evidence": []}, {"s": {"type": "noul", "instructions": "x"}})
    assert r2.meta.cached and seen["n"] == 3


def test_call_with_retries_gives_up_on_non_retryable():
    def bad():
        raise ApiError(401, "no")

    with pytest.raises(ApiError):
        call_with_retries(bad, 3, sleep=lambda s: None)


def test_parse_json_flag():
    assert parse_json_flag('{"sufficient": true}', "sufficient") is True
    assert parse_json_flag('```json\n{"sufficient": false}\n```', "sufficient") is False
    assert parse_json_flag('I think "sufficient": true.', "sufficient") is True
    assert parse_json_flag("no idea", "sufficient") is None


def test_split_is_deterministic_and_order_independent():
    ids = [f"q{i}" for i in range(100)]
    a = split_dev_test(ids, 7, 0.2)
    b = split_dev_test(list(reversed(ids)), 7, 0.2)
    assert a == b and len(a["dev"]) == 20 and not set(a["dev"]) & set(a["test"])


def test_pool_contains_relevant_docs_and_is_stable(world):
    env, qs, *_ = world
    q = qs[0]
    p1 = build_pool(q, env.store, 30, seed=1)
    p2 = build_pool(q, env.store, 30, seed=1)
    assert p1 == p2 and len(p1) >= 30
    docs = {env.store.get(c).doc_id for c in p1}
    assert set(q.gold_doc_ids) <= docs


def test_metrics():
    assert metrics.recall_at_k(["a", "x", "b"], ["a", "b"], 2) == 0.5
    assert metrics.mrr(["x", "b"], ["b"]) == 0.5
    assert metrics.ndcg_at_k(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert metrics.auc([0.1, 0.9, 0.8, 0.2], [False, True, True, False]) == 1.0
    assert metrics.ece([1.0, 1.0, 0.0, 0.0], [True, True, False, False]) == 0.0
    tau, ba = metrics.best_threshold([0.1, 0.4, 0.6, 0.9], [False, False, True, True])
    assert 0.4 < tau <= 0.6 and ba == 1.0
    mc = metrics.mcnemar_exact([True] * 8 + [False] * 2, [False] * 8 + [True] * 2)
    assert mc["a_only"] == 8 and mc["b_only"] == 2 and 0 < mc["p_value"] < 0.2
    ci = metrics.paired_bootstrap_ci([1, 2, 3, 4], [0, 1, 2, 3])
    assert ci["mean_diff"] == 1.0 and ci["lo"] == ci["hi"] == 1.0


def test_flat_index_exact_and_pool(world):
    env, qs, *_ = world
    idx = env.emb_retriever.index
    vec = idx.vectors[5]
    rows, scores = idx.search(vec, 3)
    assert rows[0] == 5 and scores[0] == pytest.approx(1.0, abs=1e-5)
    rows2, _ = idx.search(vec, 3, rows=[1, 5, 9])
    assert rows2[0] == 5 and set(rows2) <= {1, 5, 9}


def test_out_of_credit_429_is_not_retried():
    from ragexp.clients import RETRY_STATUS, _status_of

    class E(Exception):
        status_code = 429

    quota = E("Error code: 429 - {'error': {'code': 'credit_balance_exhausted', 'type': 'insufficient_quota'}}")
    assert _status_of(quota) == 402 and 402 not in RETRY_STATUS
    assert _status_of(E("Rate limit reached for requests")) == 429 and 429 in RETRY_STATUS


def test_rate_limit_message_gives_retry_after():
    from ragexp.clients import ApiError

    e = ApiError(429, "Rate limit reached ... Please try again in 26.127s. Visit https://...")
    assert e.retry_after == pytest.approx(26.127)
    assert ApiError(429, "try again in 350ms").retry_after == pytest.approx(0.35)
    assert ApiError(500, "boom").retry_after is None


def test_cache_shared_between_processes(tmp_path):
    import subprocess
    import sys

    from ragexp.cache import ApiCache

    path = tmp_path / "shared.sqlite"
    a = ApiCache(path)
    a.put("k1", {"v": 1})  # visible to another connection immediately, no explicit flush
    code = ("from ragexp.cache import ApiCache; c = ApiCache(r'%s'); "
            "assert c.get('k1') == {'v': 1}; c.put('k2', {'v': 2})" % path)
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(__import__('pathlib').Path(__file__).resolve().parent.parent))
    assert a.get("k2") == {"v": 2}


def test_blocked_question_is_recorded_not_fatal(world, tmp_path, monkeypatch):
    import json as _json

    from ragexp import runner
    from ragexp.clients import ApiError

    env, qs, *_ = world
    f = runner.Failures(tmp_path / "failures.jsonl")
    assert f.guard("G1", qs[0], 0, lambda: 42) == 42
    def blocked():
        raise ApiError(400, "content filter 1301")
    assert f.guard("G1", qs[0], 0, blocked) is None
    rec = [_json.loads(l) for l in (tmp_path / "failures.jsonl").read_text().splitlines()]
    assert rec[0]["qid"] == qs[0].qid and rec[0]["group"] == "G1" and rec[0]["status"] == 400
    # auth / credit / rate errors are NOT swallowed
    import pytest as _pt
    for status in (401, 402, 429):
        def bad(status=status):
            raise ApiError(status, "x")
        with _pt.raises(ApiError):
            f.guard("G1", qs[0], 0, bad)


def test_content_filter_gets_two_retries_then_gives_up():
    from ragexp.clients import ApiError, call_with_retries

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ApiError(400, "{'code': '1301', 'message': 'System detected potentially unsafe or sensitive content'}")
        return "ok"

    val, retries, _ = call_with_retries(flaky, 6, sleep=lambda s: None)
    assert val == "ok" and retries == 2
    calls["n"] = -10  # always blocked
    with pytest.raises(ApiError):
        call_with_retries(flaky, 6, sleep=lambda s: None)
    assert calls["n"] == -7  # initial try + exactly 2 retries
    def other():
        raise ApiError(400, "invalid request")
    with pytest.raises(ApiError):
        call_with_retries(other, 6, sleep=lambda s: None)


def test_zai_insufficient_balance_is_permanent():
    from ragexp.clients import RETRY_STATUS, _status_of

    class E(Exception):
        status_code = 429

    e = E("Error code: 429 - {'error': {'code': '1113', 'message': 'Insufficient balance or no resource package. Please recharge.'}}")
    assert _status_of(e) == 402 and 402 not in RETRY_STATUS


def test_dedupe_records_keeps_last(tmp_path):
    from ragexp.recorder import FIELDS, dedupe_records, read_records

    def row(tokens):
        r = {f: None for f in FIELDS}
        r.update(run_id="r", group="G1", dataset="d", query_id="q", replicate=0, round=1,
                 component="glm_answer", input_tokens=tokens)
        return r

    other = row(5)
    other["query_id"] = "q2"
    p = tmp_path / "records.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (row(1), other, row(2))) + "\n", encoding="utf-8")
    assert dedupe_records(p) == 1
    kept = read_records(p)
    assert len(kept) == 2 and [r["input_tokens"] for r in kept] == [5, 2]
    assert dedupe_records(p) == 0


def test_test_subsample_is_deterministic_and_nested():
    from ragexp.pipeline import subsample

    ids = [f"q{i}" for i in range(400)]
    a = subsample(ids, 0.33, 7)
    assert len(a) == 132 and a == subsample(list(reversed(ids)), 0.33, 7) and a == sorted(a)
    b = subsample(ids, 0.5, 7)
    assert set(a) <= set(b) and len(b) == 200  # raising the fraction only adds questions
    assert subsample(ids, 1.0, 7) == sorted(ids) and len(subsample(ids[:5], 0.01, 7)) == 1
    assert subsample(ids, 0.33, 8) != a  # different seed, different sample


def test_per_dataset_test_fraction_and_replicate_grading(tmp_path):
    import json as _json

    from ragexp.config import Config as _Config
    from ragexp.runner import Workspace, grade

    cfg = _Config(work_dir=str(tmp_path / "work"), runs_dir=str(tmp_path / "runs"), test_fraction=0.5,
                  test_fraction_by_dataset={"keepall": 1.0})
    ids = [f"q{i}" for i in range(20)]
    for ds in ("sampled", "keepall"):
        ws = Workspace(cfg, ds)
        ws.split_path.write_text(_json.dumps({"dev": [], "test": ids}))
    assert len(Workspace(cfg, "sampled").eval_qids("test")) == 10
    assert len(Workspace(cfg, "keepall").eval_qids("test")) == 20

    # only repeats < N are judged
    ws = Workspace(cfg, "keepall")
    qs = [{"qid": q, "question": "Q", "answer": "A", "gold_doc_ids": [], "evidence_doc_ids": [], "negative_doc_ids": [], "meta": {}} for q in ids]
    ws.questions_path.write_text("\n".join(_json.dumps(q) for q in qs), encoding="utf-8")
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    rows = [{"group": "G1", "qid": q, "replicate": rep, "answer": "A"} for q in ids[:2] for rep in range(3)]
    (run / "results.jsonl").write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")

    class J:
        n = 0

        def chat(self, messages, **kw):
            J.n += 1
            return type("R", (), {"content": '{"correct": true}', "meta": None})()

    grade(cfg, ws, "r1", judge=J(), workers=1, log=lambda *_: None, replicates=1)
    out = [_json.loads(l) for l in (run / "results.jsonl").read_text().splitlines()]
    assert J.n == 2 and [r.get("correct") for r in out] == [True, None, None, True, None, None]


def test_glm_response_without_usage_is_handled():
    from types import SimpleNamespace as N

    from ragexp.clients import ApiError, GLMClient
    from ragexp.config import GLMCfg

    def make(choices, usage):
        class Raw:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        return N(choices=choices, usage=usage)
        return GLMClient(GLMCfg(provider="p", model="m", max_retries=0), client=Raw())

    msg = lambda c: [N(message=N(content=c, tool_calls=None), finish_reason="stop")]
    m = [{"role": "user", "content": "hi"}]
    r = make(msg("an answer"), None).chat(m)  # content present, usage missing -> kept, tokens estimated
    assert r.content == "an answer" and r.meta.input_tokens > 0 and r.meta.output_tokens > 0
    for bad in (make(msg(""), None), make([], None)):  # nothing usable -> retriable error
        with pytest.raises(ApiError) as e:
            bad.chat(m)
        assert e.value.status == 502


def test_glm_requests_per_min_limiter(monkeypatch):
    import time as _t
    from types import SimpleNamespace as N

    from ragexp.clients import GLMClient
    from ragexp.config import GLMCfg

    class Raw:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return N(choices=[N(message=N(content="x", tool_calls=None), finish_reason="stop")],
                             usage=N(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None))

    # 120 req/min -> capacity 6 (3 s of burst), then 2 requests per second
    cli = GLMClient(GLMCfg(provider="p", model="m", requests_per_min=120), client=Raw())
    t0 = _t.perf_counter()
    for i in range(10):
        cli.chat([{"role": "user", "content": str(i)}])
    assert _t.perf_counter() - t0 >= 1.5  # the last 4 calls had to wait ~0.5 s each
    free = GLMClient(GLMCfg(provider="p", model="m"), client=Raw())
    t0 = _t.perf_counter()
    for i in range(10):
        free.chat([{"role": "user", "content": str(i)}])
    assert _t.perf_counter() - t0 < 1.0


def test_g8_rerank_uses_embedding_shortlist_then_jev(world):
    from ragexp.groups import run_g8
    from ragexp.retrievers import RerankRetriever

    env, qs, *_ = world
    jev_calls = []
    orig = env.jev.ask

    def counting(state, questions):
        if "passage" in state:  # per-chunk scoring only; the stop judge has no passage
            jev_calls.append(state["passage"]["id"])
        return orig(state, questions)

    env.jev.ask = counting
    rr = RerankRetriever(env.store, env.emb_retriever.index, env.emb_retriever.embedder, env.jev, shortlist=7, workers=1)
    q = qs[0]
    res = rr.search(q.question, 2)
    emb_top7 = [h.chunk_id for h in env.emb_retriever.search(q.question, 7).hits]
    assert sorted(jev_calls) == sorted(emb_top7)  # Jev only ever sees the embedding shortlist
    assert [c for c, _ in res.calls] == ["embed_query", "retrieve", "jev_rank"]
    assert len(res.hits) == 2 and {h.chunk_id for h in res.hits} <= set(emb_top7)
    assert res.calls[2][1].n_calls == 7  # cost accounting: 7 Jev requests, not the whole corpus

    env.rerank_retriever = rr
    row = run_g8(env, q, "toy", 0, tau=0.5)
    assert row["group"] == "G8" and row["stop_reason"] in ("judge", "r_max")
    assert "jev_rank" in row["components"] and "jev_judge" in row["components"] and "glm_judge" not in row["components"]
    assert q.answer in row["answer"]
