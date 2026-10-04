"""Configuration: environment settings (infrastructure, secrets) and pipeline configs (YAML).

``Settings`` decides *where* things run (store, providers, keys). ``PipelineConfig`` decides
*how* the RAG pipeline behaves; it is the unit the evaluation layer ablates, so every field
here is hashable into a reproducible ``config_id``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["fixed", "recursive", "semantic", "structure"] = "structure"
    size: int = Field(200, ge=16, le=4000, description="Max chunk size in regex tokens")
    overlap: int = Field(30, ge=0, description="Overlap in tokens (fixed/recursive splits)")
    contextual_headers: bool = Field(True, description="Prepend 'Doc › Section' to embed text")
    semantic_percentile: float = Field(85.0, gt=0, lt=100)
    min_size: int = Field(25, ge=0)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["dense", "lexical", "hybrid"] = "hybrid"
    k: int = Field(5, ge=1, le=50)
    fetch_k: int = Field(30, ge=1, le=500)
    rrf_k: int = Field(60, ge=1)
    dense_weight: float = Field(1.0, ge=0)
    lexical_weight: float = Field(1.0, ge=0)
    rerank: Literal["none", "mmr", "maxsim", "llm", "cross_encoder"] = "none"
    rerank_candidates: int = Field(20, ge=1)
    mmr_lambda: float = Field(0.7, ge=0, le=1)
    query_expansion: Literal["none", "multi_query", "hyde"] = "none"
    exclude_superseded: bool = Field(True, description="Drop chunks of documents marked status: superseded")
    num_query_variants: int = Field(3, ge=1, le=8)
    cross_encoder_model: str = "BAAI/bge-reranker-base"


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field("strict", description="Template name in prompts/templates.yaml")
    max_context_tokens: int = Field(1800, ge=100)
    context_order: Literal["relevance", "sandwich"] = "relevance"
    max_sentences: int = Field(2, ge=1, le=10, description="Extractive generator only")
    abstain_threshold: float = Field(0.46, description="Extractive generator only")
    temperature: float = Field(0.0, ge=0, description="Extractive sampling temperature")


class GuardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    injection: Literal["off", "flag", "drop"] = "drop"
    grounding: Literal["off", "flag", "abstain"] = "flag"
    grounding_threshold: float = Field(0.5, ge=0, le=1)

    @field_validator("injection", "grounding", mode="before")
    @classmethod
    def _yaml_off(cls, value: Any) -> Any:
        return "off" if value is False else value  # YAML 1.1 parses a bare `off` as False


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data.get("pipeline", data))

    def with_overrides(self, overrides: dict[str, Any]) -> PipelineConfig:
        """Return a copy with dotted-path overrides, e.g. ``{"retrieval.k": 3}``."""
        data = copy.deepcopy(self.model_dump())
        for dotted, value in overrides.items():
            node = data
            *parents, leaf = dotted.split(".")
            for key in parents:
                if key not in node or not isinstance(node[key], dict):
                    raise KeyError(f"Unknown config section in override: {dotted}")
                node = node[key]
            if leaf not in node:
                raise KeyError(f"Unknown config field in override: {dotted}")
            node[leaf] = value
        return PipelineConfig.model_validate(data)

    def fingerprint(self, *sections: str) -> str:
        """Short stable hash of the whole config, or only of the named sections."""
        data = self.model_dump()
        if sections:
            data = {s: data[s] for s in sections}
        blob = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:10]

    @property
    def config_id(self) -> str:
        return self.fingerprint()


class Settings(BaseSettings):
    """Runtime settings from environment variables (prefix ``RAGQA_``) or a ``.env`` file."""

    model_config = SettingsConfigDict(env_prefix="RAGQA_", env_file=".env", extra="ignore", populate_by_name=True)

    store: Literal["memory", "pgvector"] = "memory"
    database_url: str = "postgresql://ragqa:ragqa@localhost:5432/ragqa"
    data_dir: Path = Path(".ragqa")
    collection: str = "default"
    pipeline_config: Path | None = None

    embedder: Literal["auto", "openai", "wordllama", "hashing"] = "auto"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None
    embedding_cache: bool = True

    generator: Literal["auto", "llm", "extractive"] = "auto"
    llm_model: str = "gpt-6-luna"
    llm_base_url: str | None = None
    llm_temperature: float | None = None
    llm_seed: int | None = None
    llm_reasoning_effort: str | None = "none"
    llm_max_tokens: int = 500
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 4

    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("RAGQA_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    api_key: SecretStr | None = Field(default=None, description="If set, required as X-API-Key")
    cors_origins: list[str] = Field(default_factory=list)
    max_upload_mb: int = 20

    log_level: str = "INFO"
    log_json: bool = False

    @property
    def has_openai(self) -> bool:
        return self.openai_api_key is not None and bool(self.openai_api_key.get_secret_value())

    @property
    def resolved_embedder(self) -> str:
        if self.embedder != "auto":
            return self.embedder
        return "openai" if self.has_openai else "wordllama"

    @property
    def resolved_generator(self) -> str:
        if self.generator != "auto":
            return self.generator
        return "llm" if (self.has_openai or self.llm_base_url) else "extractive"

    def load_pipeline_config(self) -> PipelineConfig:
        if self.pipeline_config and Path(self.pipeline_config).exists():
            return PipelineConfig.from_yaml(self.pipeline_config)
        return PipelineConfig()
