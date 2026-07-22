"""Test plan for train: the training loop and results record."""

from __future__ import annotations

import json
import math

import pytest
import torch
from sklearn.metrics import f1_score

from data import LoadCora
from metrics import OversmoothingMetrics
from models import GcnModel, LastLayerReadout
from train import CaptureMetrics, MacroF1, TrainConfig, TrainRun

HIDDEN_DIM = 64
OUT_DIM = 7
IN_DIM = 1433


@pytest.fixture(scope="module")
def cora():
    return LoadCora()


@pytest.fixture(scope="module")
def coraMetrics(cora):
    return OversmoothingMetrics(cora.edge_index, cora.num_nodes)


def _BuildGcn(numLayers: int, dropout: float = 0.5) -> GcnModel:
    return GcnModel(
        numLayers=numLayers,
        inDim=IN_DIM,
        hiddenDim=HIDDEN_DIM,
        outDim=OUT_DIM,
        dropout=dropout,
        layerHooks=[],
        readout=LastLayerReadout(),
    )


def test_macro_f1_matches_sklearn_with_unrepresented_class() -> None:
    numClasses = 4
    # class 3 never appears in predictions OR targets: the case that separates a
    # correct implementation (0-contribution to the mean) from a plausible one
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    predictions = torch.tensor([0, 1, 1, 1, 2, 0, 0, 2])

    ours = MacroF1(predictions, targets, numClasses)
    reference = f1_score(
        targets.numpy(), predictions.numpy(), labels=list(range(numClasses)), average="macro", zero_division=0
    )
    assert ours == pytest.approx(float(reference), abs=1e-6)


def test_seed_determinism(cora, coraMetrics) -> None:
    config = TrainConfig(seed=0, maxEpochs=5, patience=5)

    torch.manual_seed(0)
    modelA = _BuildGcn(numLayers=2)
    recordA = TrainRun(modelA, cora, config, coraMetrics)

    torch.manual_seed(0)
    modelB = _BuildGcn(numLayers=2)
    recordB = TrainRun(modelB, cora, config, coraMetrics)

    recordA = {k: v for k, v in recordA.items() if k != "timestamp"}
    recordB = {k: v for k, v in recordB.items() if k != "timestamp"}
    # dict == dict fails purely on nan != nan (contractionSlope is legitimately
    # nan at depth 2, a single-point band); JSON round-trips nan identically in
    # both, so string comparison sidesteps IEEE 754's nan self-inequality
    assert json.dumps(recordA, sort_keys=True) == json.dumps(recordB, sort_keys=True)


def test_early_stopping_fires(cora, coraMetrics) -> None:
    config = TrainConfig(seed=0, maxEpochs=1000, patience=1)
    model = _BuildGcn(numLayers=2)
    record = TrainRun(model, cora, config, coraMetrics)
    assert record["results"]["epochsRun"] < config.maxEpochs
    assert record["results"]["bestEpoch"] <= record["results"]["epochsRun"]


def test_checkpoint_restore_is_real(cora, coraMetrics) -> None:
    config = TrainConfig(seed=0, maxEpochs=1000, patience=1)
    model = _BuildGcn(numLayers=2)
    record = TrainRun(model, cora, config, coraMetrics)
    assert record["results"]["bestEpoch"] != record["results"]["epochsRun"]
    # regression test for the final-capture-then-restore ordering: reversing
    # it would make finalMetrics a duplicate of checkpointMetrics on every run
    assert record["finalMetrics"]["dirichletEnergy"] != record["checkpointMetrics"]["dirichletEnergy"]


def test_epoch0_capture_precedes_training(cora, coraMetrics) -> None:
    torch.manual_seed(0)
    model = _BuildGcn(numLayers=2)
    expectedEpoch0 = CaptureMetrics(model, cora, coraMetrics)
    expectedEpoch0.pop("bandIndices")

    config = TrainConfig(seed=0, maxEpochs=5, patience=5)
    record = TrainRun(model, cora, config, coraMetrics)

    assert record["epoch0Metrics"]["dirichletEnergy"] == expectedEpoch0["dirichletEnergy"]
    assert record["epoch0Metrics"]["mad"] == expectedEpoch0["mad"]


def test_capture_count_is_exactly_three(cora, coraMetrics, monkeypatch) -> None:
    callCount = 0
    originalComputeAll = OversmoothingMetrics.ComputeAll

    def _CountingComputeAll(self, layerEmbeddings):
        nonlocal callCount
        callCount += 1
        return originalComputeAll(self, layerEmbeddings)

    monkeypatch.setattr(OversmoothingMetrics, "ComputeAll", _CountingComputeAll)

    config = TrainConfig(seed=0, maxEpochs=5, patience=5)
    model = _BuildGcn(numLayers=2)
    TrainRun(model, cora, config, coraMetrics)

    assert callCount == 3


def test_curve_length_and_keys(cora, coraMetrics) -> None:
    config = TrainConfig(seed=0, maxEpochs=5, patience=5)
    model = _BuildGcn(numLayers=2)
    record = TrainRun(model, cora, config, coraMetrics)

    assert len(record["trainingCurve"]) == record["results"]["epochsRun"]
    for entry in record["trainingCurve"]:
        assert set(entry.keys()) == {"epoch", "trainLoss", "valLoss", "valAccuracy"}


def test_record_conformance(cora, coraMetrics) -> None:
    config = TrainConfig(seed=0, maxEpochs=5, patience=5)
    model = _BuildGcn(numLayers=2)
    record = TrainRun(model, cora, config, coraMetrics)

    topLevelKeys = {
        "runId",
        "timestamp",
        "config",
        "bandIndices",
        "results",
        "trainingCurve",
        "epoch0Metrics",
        "checkpointMetrics",
        "finalMetrics",
        "trajectory",
        "environment",
    }
    assert topLevelKeys <= set(record.keys())

    configKeys = {
        "convType",
        "numLayers",
        "mitigations",
        "readout",
        "hiddenDim",
        "dropout",
        "learningRate",
        "weightDecay",
        "maxEpochs",
        "patience",
        "seed",
    }
    assert configKeys <= set(record["config"].keys())

    resultsKeys = {"testAccuracy", "testMacroF1", "valAccuracy", "valLoss", "bestEpoch", "epochsRun"}
    assert resultsKeys == set(record["results"].keys())

    numLayers = record["config"]["numLayers"]
    captureKeys = {"dirichletEnergy", "mad", "frobeniusSquared", "contractionSlope"}
    for block in ("epoch0Metrics", "checkpointMetrics", "finalMetrics"):
        assert set(record[block].keys()) == captureKeys
        assert len(record[block]["dirichletEnergy"]) == numLayers + 1
        assert len(record[block]["mad"]) == numLayers + 1
        assert len(record[block]["frobeniusSquared"]) == numLayers + 1

    assert len(record["bandIndices"]) > 0  # depth 2, LastLayerReadout: band is [1]
    assert record["trajectory"] == []


@pytest.fixture(scope="module")
def smokeRunRecord(cora, coraMetrics) -> dict:
    torch.manual_seed(0)
    model = GcnModel(
        numLayers=2,
        inDim=IN_DIM,
        hiddenDim=16,  # published-fidelity width
        outDim=OUT_DIM,
        dropout=0.5,
        layerHooks=[],
        readout=LastLayerReadout(),
    )
    config = TrainConfig(seed=0)  # defaults: patience=100, maxEpochs=1000
    return TrainRun(model, cora, config, coraMetrics)


def test_smoke_two_layer_gcn_reaches_published_accuracy(smokeRunRecord) -> None:
    testAccuracy = smokeRunRecord["results"]["testAccuracy"]
    print(f"\nsmoke test accuracy: {testAccuracy:.4f}, epochsRun={smokeRunRecord['results']['epochsRun']}")
    assert testAccuracy > 0.75


def test_loss_descends_on_trainable_config(smokeRunRecord) -> None:
    finalTrainLoss = smokeRunRecord["trainingCurve"][-1]["trainLoss"]
    print(f"\nfinal train loss: {finalTrainLoss:.4f}")
    assert finalTrainLoss < 0.5 * math.log(7)
