"""FastAPI service: JSON and streaming (SSE) Q&A, document management, stats and a web UI."""

import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from ragqa import __version__
from ragqa.api.schemas import AnswerOut, DocumentOut, HealthOut, IngestOut, QueryRequest
from ragqa.config import Settings
from ragqa.ingest.loaders import SUPPORTED_EXTENSIONS, load_bytes
from ragqa.observability import Stats, request_id_var
from ragqa.pipeline import RAGPipeline
from ragqa.store.base import CollectionMismatchError

log = logging.getLogger(__name__)


def create_app(pipeline: RAGPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state: dict[str, Any] = {"pipeline": pipeline, "owned": pipeline is None}
    stats = Stats()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["pipeline"] is None:
            state["pipeline"] = RAGPipeline.from_settings(settings)
        yield
        if state["owned"] and state["pipeline"] is not None:
            state["pipeline"].close()

    app = FastAPI(
        title="RAG Document Q&A",
        version=__version__,
        description="Citation-grounded question answering over your documents.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        response.headers["server-timing"] = f"app;dur={(time.perf_counter() - start) * 1e3:.1f}"
        return response

    def get_pipeline() -> RAGPipeline:
        if state["pipeline"] is None:  # pragma: no cover - lifespan always sets it
            raise HTTPException(503, "Pipeline not ready")
        return state["pipeline"]

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        expected = settings.api_key.get_secret_value() if settings.api_key else ""
        if not expected:  # unset or empty (e.g. RAGQA_API_KEY= in docker compose) disables auth
            return
        if not x_api_key or not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(401, "Invalid or missing X-API-Key")

    Pipe = Annotated[RAGPipeline, Depends(get_pipeline)]
    secured = [Depends(require_key)]

    def _run_query(pipe: RAGPipeline, body: QueryRequest) -> dict[str, Any]:
        try:
            answer = pipe.ask(body.question, body.k)
        except KeyError as exc:
            raise HTTPException(409, "No documents have been ingested into this collection yet") from exc
        except CollectionMismatchError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception:
            stats.record_error()
            log.exception("query failed")
            raise
        stats.record(
            answer.timings_ms.get("total", 0.0),
            answer.abstained,
            answer.usage.prompt_tokens,
            answer.usage.completion_tokens,
            answer.usage.cost_usd,
        )
        return answer.to_dict()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return resources.files("ragqa.api").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/health", response_model=HealthOut)
    def health(pipe: Pipe) -> dict[str, Any]:
        info = pipe.store.get_collection(pipe.collection)
        return {
            "status": "ok",
            "version": __version__,
            "collection": pipe.collection,
            "documents": info.num_documents if info else 0,
            "chunks": info.num_chunks if info else 0,
            "embedder": pipe.embedder.name,
            "generator": pipe.generator.name,
            "config_id": pipe.config.config_id,
        }

    @app.get("/v1/config", dependencies=secured)
    def config(pipe: Pipe) -> dict[str, Any]:
        return {
            "config_id": pipe.config.config_id,
            "pipeline": pipe.config.model_dump(),
            "embedder": pipe.embedder.name,
            "generator": pipe.generator.name,
        }

    @app.post("/v1/query", response_model=AnswerOut, dependencies=secured)
    def query(body: QueryRequest, pipe: Pipe) -> dict[str, Any]:
        return _run_query(pipe, body)

    @app.post("/v1/query/stream", dependencies=secured)
    def query_stream(body: QueryRequest, pipe: Pipe) -> StreamingResponse:
        rid = request_id_var.get()

        def events() -> Iterator[str]:
            # Starlette iterates sync generators in a threadpool, each step in a copied context,
            # so the request id is set (never reset) inside the generator for log correlation.
            request_id_var.set(rid)
            try:
                for event in pipe.stream(body.question, body.k):
                    if event["event"] == "done":
                        data = event["data"]
                        stats.record(
                            data["timings_ms"].get("total", 0.0),
                            data["abstained"],
                            data["usage"]["prompt_tokens"],
                            data["usage"]["completion_tokens"],
                            data["usage"]["cost_usd"],
                        )
                    yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
            except KeyError:
                yield 'event: error\ndata: "No documents have been ingested into this collection yet"\n\n'
            except Exception as exc:
                stats.record_error()
                log.exception("streaming query failed")
                yield f"event: error\ndata: {json.dumps(type(exc).__name__)}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"cache-control": "no-cache", "x-accel-buffering": "no"}
        )

    @app.post("/v1/documents", response_model=IngestOut, dependencies=secured)
    async def upload(pipe: Pipe, files: Annotated[list[UploadFile], File(...)]) -> dict[str, Any]:
        documents = []
        limit = settings.max_upload_mb * 1024 * 1024
        for upload_file in files:
            name = upload_file.filename or "upload.txt"
            if not any(name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                raise HTTPException(415, f"Unsupported file type: {name}")
            data = await upload_file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(413, f"{name} exceeds {settings.max_upload_mb} MB")
            try:
                documents.append(load_bytes(name, data))
            except Exception as exc:
                raise HTTPException(422, f"Could not parse {name}: {exc}") from exc
        try:
            report = await run_in_threadpool(pipe.ingest, documents)  # embedding is CPU/IO bound
            return report.to_dict()
        except CollectionMismatchError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/v1/documents", response_model=list[DocumentOut], dependencies=secured)
    def list_documents(pipe: Pipe) -> list[dict[str, Any]]:
        return pipe.list_documents()

    @app.delete("/v1/documents/{doc_id}", status_code=204, dependencies=secured)
    def delete_document(doc_id: str, pipe: Pipe) -> Response:
        try:
            removed = pipe.delete_document(doc_id)
        except KeyError:
            removed = False
        if not removed:
            raise HTTPException(404, f"Document '{doc_id}' not found")
        return Response(status_code=204)

    @app.get("/v1/stats", dependencies=secured)
    def service_stats() -> dict[str, Any]:
        return stats.snapshot()

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app
