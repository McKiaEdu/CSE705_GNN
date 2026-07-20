"""Figure functions -- one per report figure. matplotlib, Agg backend, sized
for IEEE single-column width (D-033). Each function writes one PDF file; the
aggregation layer (aggregation.py) is the only thing that computes the numbers.
"""

from __future__ import annotations

import math
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE

from .aggregation import Aggregate, EnergyCurve

FIGURE_WIDTH_INCHES = 3.5
FIGURE_HEIGHT_INCHES = 2.6
MIN_FONT_SIZE = 8

# categorical palette, validated colorblind-safe order (dataviz skill reference)
ARCHITECTURE_ORDER = ["gcn", "sage", "gat"]
ARCHITECTURE_COLORS = {"gcn": "#2a78d6", "sage": "#008300", "gat": "#e87ba4"}
ARCHITECTURE_MARKERS = {"gcn": "o", "sage": "s", "gat": "^"}
ARCHITECTURE_LINESTYLES = {"gcn": "-", "sage": "--", "gat": ":"}

MITIGATION_SERIES_ORDER = ["residual", "pairnorm", "jk", "pairnorm+residual", "gcnii"]
MITIGATION_COLORS = dict(zip(MITIGATION_SERIES_ORDER, ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]))
MITIGATION_MARKERS = dict(zip(MITIGATION_SERIES_ORDER, ["o", "s", "^", "D", "v"]))
MITIGATION_LINESTYLES = dict(zip(MITIGATION_SERIES_ORDER, ["-", "--", ":", "-.", "-"]))


def _NewFigure(nPanels: int = 1) -> tuple[plt.Figure, Any]:
    fig, axes = plt.subplots(1, nPanels, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    for ax in axes if nPanels > 1 else [axes]:
        ax.tick_params(labelsize=MIN_FONT_SIZE)
        ax.xaxis.label.set_size(MIN_FONT_SIZE + 1)
        ax.yaxis.label.set_size(MIN_FONT_SIZE + 1)
    return fig, axes


def _SaveFigure(fig: plt.Figure, outputPath: str) -> None:
    os.makedirs(os.path.dirname(outputPath) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(outputPath, format="pdf")
    plt.close(fig)


def PlotAccuracyVsDepth(table: pd.DataFrame, outputPath: str) -> None:
    """Arm A's headline: test accuracy vs depth, one series per architecture."""
    fig, ax = _NewFigure()
    for convType in ARCHITECTURE_ORDER:
        subset = table[(table["convType"] == convType) & (table["mitigations"].apply(len) == 0)]
        agg = Aggregate(subset, ["numLayers"]).sort_values("numLayers")
        ax.errorbar(
            agg["numLayers"],
            agg["testAccuracy_mean"],
            yerr=agg["testAccuracy_std"],
            label=convType,
            color=ARCHITECTURE_COLORS[convType],
            marker=ARCHITECTURE_MARKERS[convType],
            linestyle=ARCHITECTURE_LINESTYLES[convType],
            markersize=4,
            linewidth=1.2,
            capsize=2,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=MIN_FONT_SIZE)
    _SaveFigure(fig, outputPath)


def PlotEnergyVsLayer(
    records: list[dict], outputPath: str, capture: str = "checkpointMetrics", labels: list[str] | None = None
) -> None:
    """Normalized per-dimension energy vs layer index, log y-axis, one series
    per record (e.g. one per architecture at a fixed depth). Reads `capture`
    (checkpointMetrics by default: the state the reported accuracy comes from).
    """
    fig, ax = _NewFigure()
    labels = labels or [r["config"]["convType"] for r in records]
    for record, label in zip(records, labels):
        bandIndices, normalizedEnergy = EnergyCurve(record, capture)
        color = ARCHITECTURE_COLORS.get(label)
        ax.plot(bandIndices, normalizedEnergy, label=label, color=color, marker="o", markersize=3, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("layer index")
    ax.set_ylabel("normalized energy")
    ax.legend(fontsize=MIN_FONT_SIZE)
    _SaveFigure(fig, outputPath)


def PlotMadVsDepth(table: pd.DataFrame, outputPath: str, capture: str = "checkpoint") -> None:
    """MAD at the last band index vs depth, one series per architecture."""
    fig, ax = _NewFigure()
    column = f"{capture}MadAtLastBand"
    for convType in ARCHITECTURE_ORDER:
        subset = table[(table["convType"] == convType) & (table["mitigations"].apply(len) == 0)]
        agg = Aggregate(subset, ["numLayers"]).sort_values("numLayers")
        ax.errorbar(
            agg["numLayers"],
            agg[f"{column}_mean"],
            yerr=agg[f"{column}_std"],
            label=convType,
            color=ARCHITECTURE_COLORS[convType],
            marker=ARCHITECTURE_MARKERS[convType],
            linestyle=ARCHITECTURE_LINESTYLES[convType],
            markersize=4,
            linewidth=1.2,
            capsize=2,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("MAD (last band index)")
    ax.legend(fontsize=MIN_FONT_SIZE)
    _SaveFigure(fig, outputPath)


def PlotMitigationAblation(table: pd.DataFrame, outputPath: str) -> None:
    """Arm B (4 mitigation combos) plus arm C (GCNII): test accuracy vs depth,
    one series per mitigation -- 5 total (experiments_spec.md's open question
    on panel count/legibility is not resolved here; this renders all 5 on one
    axes)."""
    fig, ax = _NewFigure()
    for seriesName in MITIGATION_SERIES_ORDER:
        if seriesName == "gcnii":
            subset = table[table["convType"] == "gcnii"]
        else:
            mitigationTuple = tuple(sorted(seriesName.split("+")))
            subset = table[(table["convType"] == "gcn") & (table["mitigations"] == mitigationTuple)]
        agg = Aggregate(subset, ["numLayers"]).sort_values("numLayers")
        ax.errorbar(
            agg["numLayers"],
            agg["testAccuracy_mean"],
            yerr=agg["testAccuracy_std"],
            label=seriesName,
            color=MITIGATION_COLORS[seriesName],
            marker=MITIGATION_MARKERS[seriesName],
            linestyle=MITIGATION_LINESTYLES[seriesName],
            markersize=4,
            linewidth=1.2,
            capsize=2,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=MIN_FONT_SIZE, ncol=1)
    _SaveFigure(fig, outputPath)


def PlotLossCurves(records: list[dict], outputPath: str, labels: list[str] | None = None) -> None:
    """trainingCurve at depth 32, baseline vs mitigated -- the D-005 diagnostic
    that separates "never trained" from "trained then oversmoothed"."""
    fig, ax = _NewFigure()
    labels = labels or [r["runId"] for r in records]
    for record, label in zip(records, labels):
        epochs = [e["epoch"] for e in record["trainingCurve"]]
        trainLosses = [e["trainLoss"] for e in record["trainingCurve"]]
        ax.plot(epochs, trainLosses, label=label, linewidth=1.2)
    ax.axhline(y=math.log(7), color="gray", linestyle=":", linewidth=0.8, label="ln 7 (untrained)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.legend(fontsize=MIN_FONT_SIZE)
    _SaveFigure(fig, outputPath)


def PlotEnergyShift(table: pd.DataFrame, outputPath: str) -> None:
    """epoch0Metrics vs checkpointMetrics vs finalMetrics: three points per
    depth, testing the C5 recorded prediction (does energy fall or rise over
    training at a working depth)."""
    fig, ax = _NewFigure()
    captureLabels = ["epoch0", "checkpoint", "final"]
    for depth in sorted(table["numLayers"].unique()):
        subset = table[table["numLayers"] == depth]
        means = [subset[f"{c}EnergyAtLastBand"].mean() for c in captureLabels]
        ax.plot(captureLabels, means, marker="o", markersize=4, linewidth=1.2, label=f"depth={depth}")
    ax.set_yscale("log")
    ax.set_ylabel("energy (last band index)")
    ax.legend(fontsize=MIN_FONT_SIZE)
    _SaveFigure(fig, outputPath)


def PlotEmbeddingProjection(embeddingPaths: dict[str, str], labels, outputPath: str) -> None:
    """t-SNE of saved embeddings (D-031), coloured by data.y, one panel per
    entry in embeddingPaths (e.g. shallow vs deep). Fitted separately per panel
    (D-032): a shared fit would impose one neighbourhood structure on both and
    manufacture the difference the figure exists to show. random_state=0,
    perplexity=30 -- both belong in the figure caption, per D-032."""
    fig, axes = _NewFigure(nPanels=len(embeddingPaths))
    axes = axes if len(embeddingPaths) > 1 else [axes]
    for ax, (panelLabel, path) in zip(axes, embeddingPaths.items()):
        embedding = torch.load(path).numpy()
        projection = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(embedding)
        ax.scatter(projection[:, 0], projection[:, 1], c=labels, s=2, cmap="tab10")
        ax.set_title(panelLabel, fontsize=MIN_FONT_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])
    _SaveFigure(fig, outputPath)
