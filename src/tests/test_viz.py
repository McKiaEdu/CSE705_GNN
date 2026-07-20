"""Test plan for viz_spec.md."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch

from viz import (
    Aggregate,
    BuildTable,
    CheckCoverage,
    EnergyCurve,
    ExportTable,
    LoadRecords,
    PlotAccuracyVsDepth,
    PlotEmbeddingProjection,
    PlotEnergyShift,
    PlotEnergyVsLayer,
    PlotLossCurves,
    PlotMadVsDepth,
    PlotMitigationAblation,
)
from viz.figures import FIGURE_WIDTH_INCHES, MIN_FONT_SIZE, _NewFigure


def _MakeRecord(
    convType: str = "gcn",
    numLayers: int = 4,
    mitigations: list[str] | None = None,
    readout: str = "lastLayer",
    seed: int = 0,
    testAccuracy: float = 0.5,
    bandIndices: list[int] | None = None,
    dirichletEnergy: list[float] | None = None,
    mad: list[float] | None = None,
    contractionSlope: float = -0.5,
    trainingCurve: list[dict] | None = None,
) -> dict:
    mitigations = mitigations if mitigations is not None else []
    bandIndices = bandIndices if bandIndices is not None else list(range(1, numLayers))
    dirichletEnergy = dirichletEnergy if dirichletEnergy is not None else [1.0] * (numLayers + 1)
    mad = mad if mad is not None else [0.5] * (numLayers + 1)
    captureBlock = {
        "dirichletEnergy": dirichletEnergy,
        "mad": mad,
        "frobeniusSquared": [1.0] * (numLayers + 1),
        "contractionSlope": contractionSlope,
    }
    return {
        "runId": f"{convType}_{'+'.join(sorted(mitigations)) or 'none'}_d{numLayers}_s{seed}",
        "timestamp": "2026-07-20T00:00:00+00:00",
        "config": {
            "convType": convType,
            "numLayers": numLayers,
            "mitigations": mitigations,
            "readout": readout,
            "hiddenDim": 64,
            "dropout": 0.5,
            "learningRate": 0.01,
            "weightDecay": 5e-4,
            "maxEpochs": 5,
            "patience": 5,
            "seed": seed,
        },
        "bandIndices": bandIndices,
        "results": {
            "testAccuracy": testAccuracy,
            "testMacroF1": testAccuracy,
            "valAccuracy": testAccuracy,
            "valLoss": 1.0,
            "bestEpoch": 3,
            "epochsRun": 5,
        },
        "trainingCurve": trainingCurve
        or [{"epoch": e, "trainLoss": 1.9 - 0.1 * e, "valLoss": 1.9 - 0.1 * e, "valAccuracy": 0.1 * e} for e in range(1, 6)],
        "epoch0Metrics": captureBlock,
        "checkpointMetrics": captureBlock,
        "finalMetrics": captureBlock,
        "trajectory": [],
        "environment": {"python": "3.14.4", "torch": "2.13.0", "torchGeometric": "2.8.0", "device": "cpu"},
    }


def test_aggregation_correctness() -> None:
    records = [
        _MakeRecord(convType="gcn", numLayers=2, seed=0, testAccuracy=0.80),
        _MakeRecord(convType="gcn", numLayers=2, seed=1, testAccuracy=0.82),
        _MakeRecord(convType="gcn", numLayers=2, seed=2, testAccuracy=0.78),
    ]
    table = BuildTable(records)
    agg = Aggregate(table, ["convType", "numLayers"])
    row = agg.iloc[0]
    assert row["count"] == 3
    assert row["testAccuracy_mean"] == pytest.approx(0.80, abs=1e-9)
    assert row["testAccuracy_std"] == pytest.approx(np.std([0.80, 0.82, 0.78], ddof=1), abs=1e-9)


def test_coverage_detection(tmp_path) -> None:
    expected = [
        {"convType": "gcn", "mitigations": [], "numLayers": 2, "seed": s} for s in range(3)
    ]
    for item in expected[:2]:  # write only 2 of the 3 expected configs
        record = _MakeRecord(convType=item["convType"], numLayers=item["numLayers"], seed=item["seed"])
        path = os.path.join(str(tmp_path), f"{record['runId']}.json")
        with open(path, "w") as f:
            json.dump(record, f)

    table = BuildTable(LoadRecords(str(tmp_path)))
    missing = CheckCoverage(table, expected)
    assert missing == ["gcn_none_d2_s2"]


def test_load_records_skips_failure_markers(tmp_path) -> None:
    record = _MakeRecord()
    with open(os.path.join(str(tmp_path), f"{record['runId']}.json"), "w") as f:
        json.dump(record, f)
    with open(os.path.join(str(tmp_path), "gcn_none_d32_s5.failed.json"), "w") as f:
        json.dump({"runId": "gcn_none_d32_s5", "error": "nan loss"}, f)

    records = LoadRecords(str(tmp_path))
    assert len(records) == 1


def test_load_records_raises_on_malformed_file(tmp_path) -> None:
    with open(os.path.join(str(tmp_path), "broken.json"), "w") as f:
        json.dump({"runId": "broken"}, f)  # missing every other required key
    with pytest.raises(ValueError):
        LoadRecords(str(tmp_path))


def test_band_is_read_not_recomputed() -> None:
    numLayers = 4
    record = _MakeRecord(
        readout="jk",
        numLayers=numLayers,
        bandIndices=[1, 2, 3, 4],  # 1..L under JK, per D-001 C1
        dirichletEnergy=[1.0, 2.0, 1.5, 1.0, 0.5],
    )
    bandIndices, normalizedEnergy = EnergyCurve(record, "checkpointMetrics")
    assert bandIndices == [1, 2, 3, 4]
    assert len(normalizedEnergy) == 4  # L points, not L-1 -- the JK truncation regression test


def test_normalization_reference() -> None:
    record = _MakeRecord(numLayers=4, bandIndices=[1, 2, 3], dirichletEnergy=[9.0, 4.0, 2.0, 1.0, 0.5])
    _, normalizedEnergy = EnergyCurve(record, "checkpointMetrics")
    assert normalizedEnergy[0] == pytest.approx(1.0, abs=1e-12)


def test_nan_propagation_not_fabricated_to_zero() -> None:
    # the L=2 LastLayerReadout case (metrics_spec.md): contractionSlope is
    # legitimately nan. BuildTable/Aggregate must preserve it, not coerce to 0
    # (there is no dedicated slope plot yet -- viz_spec.md's own open question
    # leaves that undecided -- so this is tested at the aggregation layer,
    # which any future slope plot would read from)
    records = [_MakeRecord(numLayers=2, seed=s, contractionSlope=float("nan")) for s in range(3)]
    table = BuildTable(records)
    assert table["checkpointContractionSlope"].isna().all()

    agg = Aggregate(table, ["numLayers"])
    assert math.isnan(agg["checkpointContractionSlope_mean"].iloc[0])


def test_tsne_determinism() -> None:
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(80, 16)).astype(np.float32)
    first = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(embedding)
    second = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(embedding)
    assert np.array_equal(first, second)


def test_figure_sizing_and_font_floor() -> None:
    fig, ax = _NewFigure()
    assert fig.get_size_inches()[0] == pytest.approx(FIGURE_WIDTH_INCHES)
    assert ax.xaxis.label.get_size() >= MIN_FONT_SIZE
    assert ax.yaxis.label.get_size() >= MIN_FONT_SIZE
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        assert tick.get_size() >= MIN_FONT_SIZE


def test_all_figures_render_headless_and_nonempty(tmp_path) -> None:
    records = [
        _MakeRecord(convType=c, numLayers=n, seed=s, testAccuracy=0.5 + 0.01 * n)
        for c in ("gcn", "sage", "gat")
        for n in (2, 4)
        for s in range(3)
    ]
    records += [
        _MakeRecord(convType="gcn", numLayers=n, mitigations=m, seed=s, testAccuracy=0.4)
        for m in (["residual"], ["pairnorm"], ["jk"], ["pairnorm", "residual"])
        for n in (2, 4)
        for s in range(3)
    ]
    records += [_MakeRecord(convType="gcnii", numLayers=n, seed=s, testAccuracy=0.4) for n in (2, 4) for s in range(3)]
    table = BuildTable(records)

    def _AssertWritten(path: str) -> None:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    accPath = str(tmp_path / "accuracy_vs_depth.pdf")
    PlotAccuracyVsDepth(table, accPath)
    _AssertWritten(accPath)

    energyPath = str(tmp_path / "energy_vs_layer.pdf")
    PlotEnergyVsLayer([r for r in records if r["config"]["numLayers"] == 4][:3], energyPath)
    _AssertWritten(energyPath)

    madPath = str(tmp_path / "mad_vs_depth.pdf")
    PlotMadVsDepth(table, madPath)
    _AssertWritten(madPath)

    ablationPath = str(tmp_path / "mitigation_ablation.pdf")
    PlotMitigationAblation(table, ablationPath)
    _AssertWritten(ablationPath)

    lossPath = str(tmp_path / "loss_curves.pdf")
    PlotLossCurves(records[:2], lossPath)
    _AssertWritten(lossPath)

    shiftPath = str(tmp_path / "energy_shift.pdf")
    PlotEnergyShift(table, shiftPath)
    _AssertWritten(shiftPath)

    rng = np.random.default_rng(0)
    embeddingA = torch.tensor(rng.normal(size=(80, 16)).astype("float32"))
    embeddingB = torch.tensor(rng.normal(size=(80, 16)).astype("float32"))
    ptPathA = str(tmp_path / "embA.pt")
    ptPathB = str(tmp_path / "embB.pt")
    torch.save(embeddingA, ptPathA)
    torch.save(embeddingB, ptPathB)
    projectionPath = str(tmp_path / "embedding_projection.pdf")
    labels = rng.integers(0, 7, size=80)
    PlotEmbeddingProjection({"shallow": ptPathA, "deep": ptPathB}, labels, projectionPath)
    _AssertWritten(projectionPath)


def test_export_table_md_and_tex(tmp_path) -> None:
    records = [_MakeRecord(numLayers=2, seed=s, testAccuracy=0.8) for s in range(3)]
    table = BuildTable(records)
    agg = Aggregate(table, ["numLayers"])[["numLayers", "testAccuracy_mean", "testAccuracy_std", "count"]]

    mdPath = str(tmp_path / "table.md")
    ExportTable(agg, mdPath, "md")
    mdContent = open(mdPath).read()
    assert "testAccuracy_mean" in mdContent
    assert mdContent.startswith("|")

    texPath = str(tmp_path / "table.tex")
    ExportTable(agg, texPath, "tex")
    texContent = open(texPath).read()
    assert "tabular" in texContent


# --- integration smoke tests against the real 534-run sweep ---


@pytest.fixture(scope="module")
def realTable():
    if not os.path.isdir("results"):
        pytest.skip("results/ from the real sweep not present")
    return BuildTable(LoadRecords("results"))


def test_real_sweep_loads_and_aggregates(realTable) -> None:
    assert len(realTable) == 500  # arms A + B + C + D
    agg = Aggregate(realTable[(realTable["convType"] == "gcn") & (realTable["mitigations"].apply(len) == 0)], ["numLayers"])
    assert set(agg["numLayers"]) == {2, 4, 8, 16, 32}
    assert (agg["count"] == 10).all()


def test_real_sweep_accuracy_figure_renders(realTable, tmp_path) -> None:
    outputPath = str(tmp_path / "real_accuracy_vs_depth.pdf")
    PlotAccuracyVsDepth(realTable, outputPath)
    assert os.path.getsize(outputPath) > 0
