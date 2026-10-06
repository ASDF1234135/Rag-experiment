from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class ChunkCfg(BaseModel):
    size: int = 512
    overlap: int = 64
    encoding: str = "cl100k_base"


class EmbedCfg(BaseModel):
    model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    batch_size: int = 128  # max texts per request
    max_batch_tokens: int = 16_000  # max (estimated) tokens per request; keeps requests under the TPM limit
    tokens_per_min: int | None = None  # client-side throttle; set to the account's embedding TPM limit
    workers: int = 4
    max_retries: int = 12


class IndexCfg(BaseModel):
    """How the exact search runs. `stream` keeps RAM flat (memory-mapped fp16, block scan,
    concurrent queries coalesced); `memory` loads faiss into RAM; `auto` picks by free RAM."""

    mode: str = "auto"  # auto | memory | stream
    memory_fraction: float = 0.5  # auto: RAM mode only if the index fits in this share of free RAM
    block_rows: int = 20_000  # stream: rows per scanned block (20k x 1536 x 4 B ~ 123 MB per thread)
    threads: int = 4
    max_batch: int = 64
    max_wait_ms: float = 10.0


class GLMCfg(BaseModel):
    """OpenAI-compatible chat endpoint. provider/base_url/model are TBD in the spec."""

    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key_env: str = "GLM_API_KEY"
    temperature: float | None = 0.0
    max_tokens: int | None = None
    extra_body: dict = Field(default_factory=dict)  # e.g. reasoning on/off, provider-specific
    max_retries: int = 6
    timeout_s: float = 300.0
    requests_per_min: int | None = None  # client-side cap per process (provider RPM limit / processes sharing it)

    @field_validator("base_url")
    @classmethod
    def _strip_endpoint(cls, v: str | None) -> str | None:
        # The OpenAI SDK appends /chat/completions itself; a full endpoint URL would be doubled.
        return v.rstrip("/").removesuffix("/chat/completions") if v else v


class JevCfg(BaseModel):
    model: str = "jev-1.13.0"  # pinned; never use the jev-latest alias
    base_url: str = "https://api.typesafe.ai"
    api_key_env: str = "TYPESAFE_API_KEY"
    max_retries: int = 8
    req_per_s: float = 40.0
    tokens_per_s: float = 100_000.0
    workers: int = 16
    timeout_s: float = 120.0


class SearchCfg(BaseModel):
    top_k: int = 5
    e_max: int = 8000  # accumulated-evidence cap, tokens of the serialized form
    r_max: int = 6
    rerank_shortlist: int = 200  # G8: embedding keeps this many chunks, Jev re-ranks only those


class Frozen(BaseModel):
    """Hyper-parameters frozen from dev (spec section 6). Filled by `ragexp tune`."""

    tau: dict[str, float] = Field(default_factory=dict)  # per dataset
    r_fixed: dict[str, int] = Field(default_factory=dict)  # per dataset


class PricingCfg(BaseModel):
    """USD per 1M tokens. Costs are computed offline from token counts only."""

    jev_input: float = 0.042
    jev_output: float = 0.0
    embed_input: float = 0.02  # text-embedding-3-small list price; verify before reporting
    glm_input: float | None = None  # TBD: depends on provider
    glm_output: float | None = None  # applies to completion tokens (which already include reasoning tokens)


class Config(BaseModel):
    chunk: ChunkCfg = Field(default_factory=ChunkCfg)
    embed: EmbedCfg = Field(default_factory=EmbedCfg)
    index: IndexCfg = Field(default_factory=IndexCfg)
    glm: GLMCfg = Field(default_factory=GLMCfg)
    judge: GLMCfg = Field(default_factory=GLMCfg)  # answer-correctness judge (TBD)
    jev: JevCfg = Field(default_factory=JevCfg)
    search: SearchCfg = Field(default_factory=SearchCfg)
    frozen: Frozen = Field(default_factory=Frozen)
    pricing: PricingCfg = Field(default_factory=PricingCfg)
    seed: int = 20261002
    dev_fraction: float = 0.2
    test_fraction: float = 1.0  # budget: evaluate only this share of the test split (seeded, nested subsample)
    test_fraction_by_dataset: dict[str, float] = Field(default_factory=dict)  # overrides test_fraction
    repeats: int = 3
    work_dir: str = "work"
    runs_dir: str = "runs"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})
