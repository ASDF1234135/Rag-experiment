from __future__ import annotations

import tiktoken

from .config import ChunkCfg
from .types import Chunk, Doc


def get_encoding(name: str = "cl100k_base"):
    return tiktoken.get_encoding(name)


def chunk_document(doc: Doc, enc, cfg: ChunkCfg) -> list[Chunk]:
    """Sliding token window: `size` body tokens, `overlap` shared with the previous window.

    The title prefix is NOT counted in `size` (it is added on top when embedding).
    """
    if cfg.overlap >= cfg.size:
        raise ValueError("overlap must be smaller than size")
    ids = enc.encode(doc.text, disallowed_special=())
    if not ids:
        return []
    stride = cfg.size - cfg.overlap
    out: list[Chunk] = []
    start = 0
    while True:
        window = ids[start : start + cfg.size]
        out.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{len(out)}",
                doc_id=doc.doc_id,
                title=doc.title,
                text=enc.decode(window),
                n_tokens=len(window),
            )
        )
        if start + cfg.size >= len(ids):
            break
        start += stride
    return out


def embed_text(chunk: Chunk) -> str:
    """Text sent to the embedding model: title prefix + body."""
    return f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text
