import concurrent.futures as cf

import numpy as np

from ragexp.config import IndexCfg
from ragexp.index import FlatIndex, StreamingIndex, load_index, normalize


def _data():
    rng = np.random.default_rng(1)
    X = normalize(rng.standard_normal((5003, 64)).astype(np.float32)).astype(np.float16)
    Q = normalize(rng.standard_normal((24, 64)).astype(np.float32))
    return X, Q


def test_streaming_matches_exact():
    X, Q = _data()
    exact = FlatIndex(X, use_faiss=False)
    st = StreamingIndex(X, block_rows=700, threads=3, coalesce=False)
    for q in Q[:6]:
        a, sa = exact.search(q, 7)
        b, sb = st.search(q, 7)
        assert a == b and np.allclose(sa, sb, atol=1e-6)
    # k larger than a block still returns the right head of the ranking
    assert st.search(Q[0], 1000)[0][:5] == exact.search(Q[0], 5)[0]
    # candidate-pool path
    assert st.search(Q[0], 3, rows=[4, 9, 100])[0] == exact.search(Q[0], 3, rows=[4, 9, 100])[0]
    # tiny matrix: block smaller than k
    small = StreamingIndex(X[:5], block_rows=3, threads=2, coalesce=False)
    assert len(small.search(Q[0], 10)[0]) == 5


def test_streaming_coalesces_concurrent_queries():
    X, Q = _data()
    exact = FlatIndex(X, use_faiss=False)
    st = StreamingIndex(X, block_rows=700, threads=3, max_wait_ms=50)
    scans = {"n": 0}
    orig = st.search_batch

    def counting(Qb, k):
        scans["n"] += 1
        return orig(Qb, k)

    st._co._fn = counting
    with cf.ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda q: st.search(q, 5), Q))
    assert [r[0] for r in res] == [exact.search(q, 5)[0] for q in Q]
    assert scans["n"] < len(Q)  # fewer passes over the data than queries


def test_load_index_auto_picks_by_memory(tmp_path):
    np.save(tmp_path / "e.npy", np.ones((10, 8), np.float16))
    assert isinstance(load_index(tmp_path / "e.npy", IndexCfg(mode="auto", memory_fraction=1e-12)), StreamingIndex)
    assert isinstance(load_index(tmp_path / "e.npy", IndexCfg(mode="auto", memory_fraction=0.5)), FlatIndex)
    assert isinstance(load_index(tmp_path / "e.npy", IndexCfg(mode="stream")), StreamingIndex)


def test_streaming_file_read_path_matches(tmp_path):
    X, Q = _data()
    np.save(tmp_path / "e.npy", X)
    exact = FlatIndex(X, use_faiss=False)
    st = load_index(tmp_path / "e.npy", IndexCfg(mode="stream", block_rows=700, threads=2))
    assert isinstance(st, StreamingIndex) and st._path
    for q in Q[:5]:
        assert st.search(q, 6)[0] == exact.search(q, 6)[0]
