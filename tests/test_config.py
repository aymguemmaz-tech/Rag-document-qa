from pathlib import Path

import pytest

from ragqa.config import PipelineConfig, Settings


def test_overrides_and_fingerprints() -> None:
    base = PipelineConfig()
    changed = base.with_overrides({"retrieval.k": 8, "chunking.strategy": "fixed"})
    assert changed.retrieval.k == 8 and base.retrieval.k == 5
    assert changed.config_id != base.config_id
    assert base.fingerprint("chunking") == base.with_overrides({"retrieval.k": 8}).fingerprint("chunking")
    with pytest.raises(KeyError):
        base.with_overrides({"retrieval.top_k": 3})


def test_yaml_off_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("pipeline:\n  guard:\n    injection: off\n  retrieval:\n    mode: dense\n")
    config = PipelineConfig.from_yaml(path)
    assert config.guard.injection == "off" and config.retrieval.mode == "dense"


def test_settings_resolve_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    offline = Settings(_env_file=None)  # type: ignore[call-arg]
    assert offline.resolved_embedder == "wordllama" and offline.resolved_generator == "extractive"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    online = Settings(_env_file=None)  # type: ignore[call-arg]
    assert online.resolved_embedder == "openai" and online.resolved_generator == "llm"
    local = Settings(llm_base_url="http://localhost:11434/v1", embedder="wordllama", _env_file=None)  # type: ignore[call-arg]
    assert local.resolved_generator == "llm"
