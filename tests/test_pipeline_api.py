import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragqa.api.app import create_app
from ragqa.config import PipelineConfig, Settings
from ragqa.ingest.loaders import parse_markdown
from ragqa.pipeline import RAGPipeline
from ragqa.store import CollectionMismatchError
from tests.conftest import SAMPLE_MD


def test_ingest_is_incremental(pipeline: RAGPipeline) -> None:
    doc = parse_markdown(SAMPLE_MD, "battery-sop", "battery-sop.md")
    assert pipeline.ingest([doc]).skipped == ["battery-sop"]
    changed = parse_markdown(SAMPLE_MD + "\nNew paragraph.\n", "battery-sop", "battery-sop.md")
    report = pipeline.ingest([changed])
    assert report.updated == ["battery-sop"] and report.chunks > 0


def test_ask_returns_cited_grounded_answer(pipeline: RAGPipeline) -> None:
    answer = pipeline.ask("How many packs per van?")
    assert "40" in answer.text and answer.citations
    assert answer.citations[0].label == "S1" or answer.citations[0].chunk_id
    assert answer.grounded_score == 1.0
    assert {"retrieve", "generate", "total"} <= set(answer.timings_ms)
    data = answer.to_dict()
    assert data["sources"][0]["label"] == "S1" and data["config_id"] == pipeline.config.config_id


def test_stream_emits_sources_tokens_done(pipeline: RAGPipeline) -> None:
    events = list(pipeline.stream("Which extinguisher should be used?"))
    assert events[0]["event"] == "sources"
    assert events[-1]["event"] == "done" and "AVD" in events[-1]["data"]["answer"]
    assert any(e["event"] == "token" for e in events)


def test_from_settings_lite_mode_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(data_dir=tmp_path, embedder="hashing", generator="extractive", _env_file=None)  # type: ignore[call-arg]
    (tmp_path / "doc.md").write_text(SAMPLE_MD, encoding="utf-8")
    first = RAGPipeline.from_settings(settings, PipelineConfig())
    assert first.ingest_path(tmp_path / "doc.md").added == ["doc"]
    first.close()
    second = RAGPipeline.from_settings(settings, PipelineConfig())  # new process, same data dir
    assert [d["doc_id"] for d in second.list_documents()] == ["doc"]
    assert second.delete_document("doc") is True
    second.close()


def test_changing_embedder_on_existing_collection_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "doc.md").write_text(SAMPLE_MD, encoding="utf-8")
    first = RAGPipeline.from_settings(Settings(data_dir=tmp_path, embedder="hashing", _env_file=None))  # type: ignore[call-arg]
    first.ingest_path(tmp_path)
    first.close()
    other = RAGPipeline.from_settings(Settings(data_dir=tmp_path, embedder="wordllama", _env_file=None))  # type: ignore[call-arg]
    with pytest.raises(CollectionMismatchError):
        other.ingest_path(tmp_path, force=True)
    other.close()


@pytest.fixture
def client(pipeline: RAGPipeline) -> TestClient:
    return TestClient(create_app(pipeline=pipeline, settings=Settings(_env_file=None)))  # type: ignore[call-arg]


def test_api_health_query_and_stats(client: TestClient) -> None:
    with client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["documents"] == 1
        response = client.post("/v1/query", json={"question": "How many packs per van?"})
        assert response.status_code == 200 and response.headers["x-request-id"]
        body = response.json()
        assert "40" in body["answer"] and body["citations"]
        assert client.post("/v1/query", json={"question": ""}).status_code == 422
        assert client.get("/v1/stats").json()["requests"] == 1
        assert "<title>Document Q&amp;A</title>" in client.get("/").text


def test_api_streaming_sse(client: TestClient) -> None:
    with client, client.stream("POST", "/v1/query/stream", json={"question": "Which extinguisher?"}) as response:
        raw = "".join(response.iter_text())
    events = [block.split("\n") for block in raw.strip().split("\n\n")]
    names = [lines[0].removeprefix("event: ") for lines in events]
    assert names[0] == "sources" and names[-1] == "done"
    done = json.loads(events[-1][1].removeprefix("data: "))
    assert "AVD" in done["answer"]


def test_api_document_lifecycle(client: TestClient) -> None:
    with client:
        files = {"files": ("faq.md", b"# FAQ\n\nThe office opens at 08:00 every weekday.\n", "text/markdown")}
        report = client.post("/v1/documents", files=files).json()
        assert report["added"] == ["faq"]
        assert {d["doc_id"] for d in client.get("/v1/documents").json()} == {"battery-sop", "faq"}
        bad = client.post("/v1/documents", files={"files": ("x.exe", b"MZ", "application/octet-stream")})
        assert bad.status_code == 415
        assert client.delete("/v1/documents/faq").status_code == 204
        assert client.delete("/v1/documents/faq").status_code == 404


def test_api_key_is_enforced(pipeline: RAGPipeline) -> None:
    settings = Settings(api_key="s3cret", _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(pipeline=pipeline, settings=settings)) as client:
        assert client.post("/v1/query", json={"question": "q"}).status_code == 401
        ok = client.post("/v1/query", json={"question": "packs per van"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
        assert client.get("/health").status_code == 200  # health stays public


def test_api_empty_collection_returns_409(pipeline: RAGPipeline, embedder: object) -> None:
    from ragqa.store import MemoryStore

    empty = RAGPipeline(
        pipeline.config,
        store=MemoryStore(),
        embedder=pipeline.embedder,
        generator=pipeline.generator,
        collection="empty",
    )
    with TestClient(create_app(pipeline=empty, settings=Settings(_env_file=None))) as client:  # type: ignore[call-arg]
        assert client.post("/v1/query", json={"question": "anything"}).status_code == 409


def test_empty_api_key_disables_auth(pipeline: RAGPipeline) -> None:
    settings = Settings(api_key="", _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(pipeline=pipeline, settings=settings)) as client:
        assert client.post("/v1/query", json={"question": "packs per van"}).status_code == 200
