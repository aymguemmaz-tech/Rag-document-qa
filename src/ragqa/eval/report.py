"""Markdown report and charts for an ablation run."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from ragqa.eval.ablation import AblationResult, _label
from ragqa.eval.runner import by_category

# Reference data-viz palette (validated categorical order, text and chrome tokens).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

METRIC_NAMES = {
    "recall@5": "Recall@5",
    "mrr": "MRR",
    "ndcg@5": "nDCG@5",
    "accuracy": "Answer accuracy",
    "f1": "Token F1",
    "abstention_correct": "Correct abstention",
    "overall_correct": "Overall correct",
    "citation_precision": "Citation precision",
    "grounded": "Groundedness",
    "injection_followed": "Injection followed",
}


def _pct(summary: dict[str, float] | None, ci: bool = True) -> str:
    if not summary:
        return "–"
    text = f"{100 * summary['mean']:.1f}"
    if ci and "lo" in summary:
        text += f" [{100 * summary['lo']:.1f}, {100 * summary['hi']:.1f}]"
    return text


def _delta(cmp: dict[str, float] | None) -> str:
    if not cmp:
        return "–"
    stars = " **✓**" if cmp.get("p_holm", 1.0) < 0.05 else ""
    return (
        f"{100 * cmp['diff']:+.1f} [{100 * cmp['lo']:+.1f}, {100 * cmp['hi']:+.1f}] "
        f"(p={cmp.get('p_holm', cmp['p']):.3f}){stars}"
    )


def write_markdown(result: AblationResult, out_dir: Path, charts: list[str]) -> Path:
    spec, m = result.spec, result.manifest
    base = result.summaries["baseline"]
    lines = [
        f"# Ablation report: {spec.name}",
        "",
        f"*{m['created_utc']} · ragqa {m['ragqa_version']} · commit {m['git_commit'] or 'n/a'} · "
        f"{m['n_questions']} questions ({m['split']} split) · {m['n_documents']} documents · "
        f"embedder `{m['embedder']}` · generator `{m['generator']}` · baseline `{m['baseline_config_id']}`*",
        "",
        "All rates are percentages with 95% bootstrap confidence intervals. Differences are paired "
        "against the baseline (same questions); p-values come from a sign-flip permutation test with "
        "Holm-Bonferroni correction across all experiments. **✓** marks p < 0.05 after correction.",
        "",
        "## Baseline",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for metric in spec.report_metrics:
        if metric in base:
            lines.append(f"| {METRIC_NAMES.get(metric, metric)} | {_pct(base[metric])} |")
    lat = base["latency_ms"]
    lines.append(f"| Latency p50 / p95 | {lat['p50']:.1f} ms / {lat['p95']:.1f} ms |")
    lines.append(f"| Cost per question | ${base['cost_usd']['per_question']:.5f} |")
    lines.append("")

    primary = spec.primary_metrics
    lines += ["## One-factor-at-a-time effects", ""]
    for factor, levels in result.factor_levels.items():
        lines += [f"### `{factor}`", ""]
        header = "| Level | " + " | ".join(f"{METRIC_NAMES.get(p, p)} | Δ vs baseline" for p in primary) + " |"
        lines += [header, "| --- |" + " --- | --- |" * len(primary)]
        for level, name in levels:
            summary = result.summaries[name]
            label = f"**{_label(level)}** (baseline)" if name == "baseline" else _label(level)
            cells = []
            for p in primary:
                cells += [
                    _pct(summary.get(p)),
                    "–" if name == "baseline" else _delta(result.comparisons.get(name, {}).get(p)),
                ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    for table in result.interaction_tables:
        f1, f2 = table["factors"]
        metric = primary[0]
        lines += [f"## Interaction: `{f1}` × `{f2}` ({METRIC_NAMES.get(metric, metric)})", ""]
        lines += [
            "| " + f1 + " \\ " + f2 + " | " + " | ".join(table["levels"][1]) + " |",
            "| --- |" + " --- |" * len(table["levels"][1]),
        ]
        for a in table["levels"][0]:
            row = [_pct(result.summaries[table["cells"][f"{a}|{b}"]].get(metric), ci=False) for b in table["levels"][1]]
            lines.append(f"| {a} | " + " | ".join(row) + " |")
        lines.append("")

    lines += [
        "## Per-category results (baseline)",
        "",
        "| Category | Questions | "
        + " | ".join(METRIC_NAMES.get(x, x) for x in ("recall@5", "overall_correct", "citation_precision"))
        + " |",
        "| --- | --- | --- | --- | --- |",
    ]
    counts: dict[str, int] = defaultdict(int)
    for rec in result.records["baseline"]:
        counts[rec.category] += 1
    per = {x: by_category(result.records["baseline"], x) for x in ("recall@5", "overall_correct", "citation_precision")}
    for cat in sorted(counts):
        vals = [f"{100 * per[x][cat]:.1f}" if cat in per[x] else "–" for x in per]
        lines.append(f"| {cat} | {counts[cat]} | " + " | ".join(vals) + " |")
    lines.append("")

    if result.variability:
        v = result.variability
        lines += [
            "## Output variability",
            "",
            f"{v['runs']} runs per temperature over {v['n_questions']} questions (generator: {v['generator']}).",
            "",
            "| Temperature | Accuracy (mean ± sd) | Flip rate | Answer agreement | Pairwise F1 | Citation Jaccard |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in v["rows"]:
            lines.append(
                f"| {row['temperature']} | {100 * row['accuracy_mean']:.1f} ± {100 * row['accuracy_sd']:.1f} | "
                f"{100 * row['flip_rate']:.1f} | {100 * row['answer_agreement']:.1f} | "
                f"{100 * row['pairwise_f1']:.1f} | {100 * row['citation_jaccard']:.1f} |"
            )
        lines.append("")

    if charts:
        lines += ["## Charts", ""] + [f"![{Path(c).stem}](charts/{Path(c).name})" for c in charts] + [""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
# Charts (matplotlib, static PNG for README/PDF)
# ----------------------------------------------------------------------------------------------


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def forest_chart(result: AblationResult, metric: str, path: Path) -> Path:
    """Effect of every one-factor change vs the baseline: paired difference with 95% CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows: list[tuple[str, float, float, float, bool, str]] = []
    for factor, levels in result.factor_levels.items():
        for level, name in levels:
            cmp = result.comparisons.get(name, {}).get(metric)
            if name == "baseline" or not cmp:
                continue
            if cmp["diff"] == cmp["lo"] == cmp["hi"] == 0:
                continue  # identical on every question (e.g. generation settings for a retrieval metric)
            rows.append(
                (
                    f"{factor} = {_label(level)}",
                    100 * cmp["diff"],
                    100 * cmp["lo"],
                    100 * cmp["hi"],
                    cmp.get("p_holm", 1.0) < 0.05,
                    factor,
                )
            )
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(rows) + 1.3), facecolor=SURFACE)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)
    previous = None
    for i, (_, diff, lo, hi, significant, factor) in enumerate(rows):
        y = len(rows) - i
        if previous is not None and factor != previous:
            ax.axhline(y + 0.5, color=GRID, linewidth=0.8)
        previous = factor
        color = SERIES[0] if significant else "#9a9993"
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([diff], [y], s=42, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(0, color=TEXT_2, linewidth=1, zorder=1)
    ax.set_yticks(range(len(rows), 0, -1), [r[0] for r in rows], fontsize=8, color=TEXT)
    ax.set_ylim(0.4, len(rows) + 0.6)
    span = max([abs(v) for r in rows for v in (r[2], r[3])] + [1.0])
    ax.set_xlim(-span * 1.1, span * 1.1)
    ax.set_xlabel(
        "difference vs baseline (percentage points, paired 95% CI)\n"
        "changes that left every question unchanged are omitted",
        fontsize=8,
        color=TEXT_2,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=SERIES[0],
            markeredgecolor=SURFACE,
            linewidth=2,
            label="significant (Holm p < 0.05)",
        ),
        Line2D([0], [0], marker="o", color="#9a9993", markeredgecolor=SURFACE, linewidth=2, label="not significant"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT)
    ax.set_title(
        f"{METRIC_NAMES.get(metric, metric)}: effect of each change vs the baseline",
        fontsize=10,
        color=TEXT,
        loc="left",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def recall_curve_chart(result: AblationResult, path: Path) -> Path | None:
    levels = result.factor_levels.get("retrieval.mode")
    if not levels:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = result.spec.ks
    lows: list[float] = []
    fig, ax = plt.subplots(figsize=(5.2, 3.2), facecolor=SURFACE)
    _style(ax)
    for color, (level, name) in zip(SERIES, levels):
        ys = [100 * result.summaries[name].get(f"recall@{k}", {"mean": 0})["mean"] for k in ks]
        ax.plot(
            ks,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=str(level),
            zorder=3,
        )
        lows.append(min(ys))
    ax.set_xticks(ks)
    ax.set_xlim(min(ks) - 0.5, max(ks) + 0.5)
    ax.set_ylim(max(0, (min(lows) // 10) * 10 - 10), 100.5)
    ax.set_xlabel("k (chunks retrieved)", fontsize=8, color=TEXT_2)
    ax.set_ylabel("Recall@k (%)", fontsize=8, color=TEXT_2)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT)
    ax.set_title("Evidence recall vs retrieval depth", fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def category_heatmap(result: AblationResult, factor: str, metric: str, path: Path) -> Path | None:
    levels = result.factor_levels.get(factor)
    if not levels:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    per = {_label(lv): by_category(result.records[name], metric) for lv, name in levels}
    cats = sorted({c for d in per.values() for c in d})
    cols = list(per)
    data = [[100 * per[c].get(cat, float("nan")) for c in cols] for cat in cats]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2.2, 0.42 * len(cats) + 1.2), facecolor=SURFACE)
    image = ax.imshow(data, cmap=cmap, vmin=50, vmax=100, aspect="auto")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cast(Any, bar.outline).set_visible(False)  # matplotlib 3.10 stubs mistype the colorbar outline
    bar.ax.tick_params(labelsize=7, colors=TEXT_2, length=0)
    ax.set_xticks(range(len(cols)), cols, fontsize=8, color=TEXT)
    ax.set_yticks(range(len(cats)), cats, fontsize=8, color=TEXT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if value == value:  # not NaN
                ax.text(
                    j, i, f"{value:.0f}", ha="center", va="center", fontsize=8, color="#ffffff" if value >= 80 else TEXT
                )
    ax.set_title(
        f"{METRIC_NAMES.get(metric, metric)} (%) by category × {factor} (scale 50-100)",
        fontsize=9,
        color=TEXT,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def variability_chart(result: AblationResult, path: Path) -> Path | None:
    if not result.variability:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result.variability["rows"]
    temps = [r["temperature"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.8), facecolor=SURFACE)
    for ax in (ax1, ax2):
        _style(ax)
        ax.set_xticks(temps)
        ax.set_xlabel("sampling temperature", fontsize=8, color=TEXT_2)
    ax1.plot(
        temps,
        [100 * r["flip_rate"] for r in rows],
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    ax1.set_ylim(0, max(10, max(100 * r["flip_rate"] for r in rows) * 1.3))
    ax1.set_title("Questions that flip right↔wrong (%)", fontsize=9, color=TEXT, loc="left")
    means = [100 * r["accuracy_mean"] for r in rows]
    sds = [100 * r["accuracy_sd"] for r in rows]
    ax2.errorbar(
        temps,
        means,
        yerr=sds,
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        ecolor=TEXT_2,
        elinewidth=1,
        capsize=3,
    )
    ax2.set_ylim(0, 100)
    ax2.set_title("Overall correct (%, mean ± sd over runs)", fontsize=9, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_report(result: AblationResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    charts_dir = out / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts: list[str] = []
    for metric in result.spec.primary_metrics:
        charts.append(str(forest_chart(result, metric, charts_dir / f"effects_{metric.replace('@', '_at_')}.png")))
    for maybe in (
        recall_curve_chart(result, charts_dir / "recall_at_k.png"),
        category_heatmap(result, "chunking.strategy", "recall@5", charts_dir / "category_by_chunking.png"),
        category_heatmap(result, "retrieval.mode", "recall@5", charts_dir / "category_by_mode.png"),
        variability_chart(result, charts_dir / "variability.png"),
    ):
        if maybe is not None:
            charts.append(str(maybe))
    return write_markdown(result, out, charts)
