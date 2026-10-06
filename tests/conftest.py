import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragexp.cache import ApiCache  # noqa: E402
from ragexp.chunking import get_encoding  # noqa: E402
from ragexp.config import Config  # noqa: E402
from ragexp.fakes import FakeEmbedder, FakeJev, FakeLLM, make_two_hop_corpus  # noqa: E402
from ragexp.groups import Env  # noqa: E402
from ragexp.index import FlatIndex, normalize  # noqa: E402
from ragexp.pipeline import build_store  # noqa: E402
from ragexp.recorder import Recorder  # noqa: E402
from ragexp.retrievers import EmbeddingRetriever, JevRetriever  # noqa: E402
from ragexp.store import ChunkStore  # noqa: E402
from ragexp.chunking import embed_text  # noqa: E402


@pytest.fixture()
def world(tmp_path):
    cfg = Config()
    cfg.search.r_max = 4
    cfg.search.top_k = 2  # toy corpus: keep each hop in its own round
    enc = get_encoding(cfg.chunk.encoding)
    docs, questions = make_two_hop_corpus()
    store = ChunkStore(tmp_path / "chunks.sqlite")
    build_store(docs, store, enc, cfg.chunk)
    emb = FakeEmbedder()
    vecs = normalize(emb.embed([embed_text(c) for b in store.iter_chunks() for c in b])[0])
    index = FlatIndex(vecs, use_faiss=False)
    llm, jev = FakeLLM(), FakeJev()
    rec = Recorder(None)
    env = Env(cfg=cfg, store=store, enc=enc, llm=llm, emb_retriever=EmbeddingRetriever(store, index, emb),
              recorder=rec, run_id="t", jev=jev, jev_retriever=JevRetriever(store, jev, workers=4))
    return env, questions, docs, tmp_path
