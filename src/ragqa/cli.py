"""Command-line interface: ``ragqa ingest | ask | serve | documents | delete | eval ...``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import yaml

from ragqa.config import Settings
from ragqa.observability import setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Citation-grounded document Q&A (RAG) with a measurable evaluation layer.",
)
eval_app = typer.Typer(no_args_is_help=True, help="Validate the gold set, run ablations, gate CI.")
app.add_typer(eval_app, name="eval")


def _settings(config: Path | None, collection: str | None) -> Settings:
    settings = Settings()
    if config is not None:
        settings.pipeline_config = config
    if collection is not None:
        settings.collection = collection
    return settings


ConfigOpt = typer.Option(None, "--config", "-c", help="Pipeline YAML (default: built-in defaults)")
CollectionOpt = typer.Option(None, "--collection", help="Collection name (default: RAGQA_COLLECTION)")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    settings = Settings()
    setup_logging("DEBUG" if verbose else settings.log_level, settings.log_json)


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="File or folder (.md .txt .pdf .html)"),
    force: bool = typer.Option(False, help="Re-index unchanged documents"),
    config: Path | None = ConfigOpt,
    collection: str | None = CollectionOpt,
) -> None:
    """Index documents (incremental: unchanged files are skipped by content hash)."""
    from ragqa.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(_settings(config, collection))
    try:
        typer.echo(json.dumps(pipeline.ingest_path(path, force=force).to_dict(), indent=2))
    finally:
        pipeline.close()


@app.command()
def ask(
    question: str = typer.Argument(...),
    k: int | None = typer.Option(None, help="Number of chunks to retrieve"),
    as_json: bool = typer.Option(False, "--json", help="Print the full answer object"),
    config: Path | None = ConfigOpt,
    collection: str | None = CollectionOpt,
) -> None:
    """Answer a question with citations."""
    from ragqa.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(_settings(config, collection))
    try:
        answer = pipeline.ask(question, k)
    finally:
        pipeline.close()
    if as_json:
        typer.echo(json.dumps(answer.to_dict(include_text=False), indent=2))
        return
    typer.secho(answer.text, bold=True)
    typer.echo("")
    for i, source in enumerate(answer.sources, start=1):
        cited = any(c.label == f"S{i}" for c in answer.citations)
        flag = f"  {source.flags}" if source.flags else ""
        typer.echo(f"  {'*' if cited else ' '} [S{i}] {source.chunk.location}  (score {source.score:.3f}){flag}")
    grounded = "n/a" if answer.grounded_score is None else f"{answer.grounded_score:.2f}"
    typer.echo(
        f"\ngrounded={grounded}  flags={answer.flags or '-'}  generator={answer.generator}  "
        f"total={answer.timings_ms.get('total', 0):.0f} ms  cost=${answer.usage.cost_usd:.5f}"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev)"),
) -> None:
    """Run the HTTP API and web UI."""
    import uvicorn

    uvicorn.run("ragqa.api.app:create_app", factory=True, host=host, port=port, reload=reload)


@app.command()
def documents(collection: str | None = CollectionOpt) -> None:
    """List indexed documents."""
    from ragqa.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(_settings(None, collection))
    try:
        for doc in pipeline.list_documents():
            typer.echo(f"{doc['doc_id']:<40} {doc['num_chunks']:>4} chunks  {doc['title']}")
    finally:
        pipeline.close()


@app.command()
def delete(doc_id: str, collection: str | None = CollectionOpt) -> None:
    """Delete a document and its chunks."""
    from ragqa.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(_settings(None, collection))
    try:
        removed = pipeline.delete_document(doc_id)
    finally:
        pipeline.close()
    typer.echo("deleted" if removed else "not found")
    raise typer.Exit(0 if removed else 1)


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------


@eval_app.command("validate")
def eval_validate(
    dataset: Path = typer.Option(Path("data/eval/qa.jsonl"), exists=True),
    corpus: Path = typer.Option(Path("data/corpus"), exists=True),
) -> None:
    """Check that every evidence span exists verbatim in the corpus."""
    from collections import Counter

    from ragqa.eval.dataset import load_dataset, locate_evidence
    from ragqa.ingest.loaders import load_path

    items = load_dataset(dataset)
    docs = {d.doc_id: d for d in load_path(corpus)}
    report = locate_evidence(items, docs)
    typer.echo(f"{len(items)} questions over {len(docs)} documents")
    typer.echo("by category: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(i.category for i in items).items())))
    typer.echo("by split:    " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(i.split for i in items).items())))
    for warning in report.warnings:
        typer.secho(f"warning: {warning}", fg="yellow")
    for error in report.errors:
        typer.secho(f"error: {error}", fg="red")
    raise typer.Exit(0 if report.ok else 1)


@eval_app.command("run")
def eval_run(
    spec: Path = typer.Option(Path("configs/ablation.yaml"), exists=True, help="Ablation spec YAML"),
    out: Path | None = typer.Option(None, help="Output folder (default results/<name>)"),
    limit: int | None = typer.Option(None, help="Only the first N questions (smoke test)"),
) -> None:
    """Run an ablation grid and write records, summary, report and charts."""
    from ragqa.eval.ablation import AblationRunner, AblationSpec, save_result
    from ragqa.eval.report import write_report

    ablation = AblationSpec.from_yaml(spec)
    runner = AblationRunner(ablation, log=typer.echo)
    if limit:
        runner.items = runner.items[:limit]
    result = runner.run()
    folder = save_result(result, out or Path("results") / ablation.name)
    report = write_report(result, folder)
    typer.echo(f"\nReport: {report}")


@eval_app.command("gate")
def eval_gate(config: Path = typer.Option(Path("configs/quality_gate.yaml"), exists=True)) -> None:
    """Run the baseline only and fail (exit 1) if any metric crosses its threshold."""
    from ragqa.eval.ablation import AblationRunner, AblationSpec

    gate = yaml.safe_load(config.read_text(encoding="utf-8"))
    ablation = AblationSpec.from_yaml(gate["spec"]).model_copy(
        update={"factors": {}, "interactions": [], "variability": None, "n_boot": 500, "n_perm": 1000}
    )
    result = AblationRunner(ablation, log=typer.echo).run()
    summary = result.summaries["baseline"]
    failed = False
    for name, threshold in gate["thresholds"].items():
        metric, upper = (name[:-4], True) if name.endswith("_max") else (name, False)
        value = summary.get(metric, {}).get("mean", float("nan"))
        ok = value <= threshold if upper else value >= threshold
        failed |= not ok
        sign = "<=" if upper else ">="
        typer.secho(
            f"{'PASS' if ok else 'FAIL'}  {metric:<20} {value:.3f} {sign} {threshold}", fg="green" if ok else "red"
        )
    raise typer.Exit(1 if failed else 0)


@eval_app.command("calibrate")
def eval_calibrate(
    spec: Path = typer.Option(Path("configs/ablation.yaml"), exists=True),
    split: str = typer.Option("dev", help="Calibrate on this split - never on the test split"),
    start: float = 0.30,
    stop: float = 0.60,
    step: float = 0.02,
) -> None:
    """Sweep the extractive abstention threshold on the dev split and pick the best."""
    import numpy as np

    from ragqa.config import PipelineConfig
    from ragqa.eval.ablation import AblationRunner, AblationSpec
    from ragqa.eval.runner import run_items

    ablation = AblationSpec.from_yaml(spec).model_copy(update={"split": split})
    runner = AblationRunner(ablation, log=typer.echo)
    base = PipelineConfig.model_validate(ablation.baseline)
    best: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    typer.echo(f"{'threshold':>9}  overall  accuracy  abstention  balanced")
    for threshold in np.arange(start, stop + 1e-9, step):
        cfg = base.with_overrides({"generation.abstain_threshold": round(float(threshold), 3)})
        records = run_items(runner.pipeline_for(cfg), runner.items, "calibrate")

        def mean(metric: str, recs: list = records) -> float:
            vals = [r.metrics[metric] for r in recs if metric in r.metrics]
            return sum(vals) / len(vals) if vals else float("nan")

        overall, accuracy, abstention = mean("overall_correct"), mean("accuracy"), mean("abstention_correct")
        balanced = (accuracy + abstention) / 2  # tie-break: do not trade away abstention for free
        typer.echo(f"{threshold:9.2f}  {overall:7.3f}  {accuracy:8.3f}  {abstention:10.3f}  {balanced:8.3f}")
        if (overall, balanced) > best[:2]:
            best = (overall, balanced, float(threshold))
    typer.secho(
        f"best abstain_threshold on '{split}': {best[2]:.2f} (overall_correct {best[0]:.3f}, balanced {best[1]:.3f})",
        bold=True,
    )


def run() -> None:  # pragma: no cover
    app(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    app()
