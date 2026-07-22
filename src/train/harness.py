"""TrainRun: the training loop, early stopping, checkpoint selection, metric
captures, and results-record assembly. The only component that owns an
optimizer or decides which weights the reported numbers come from.
"""

from __future__ import annotations

import copy
import platform
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import torch_geometric
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data

from metrics import OversmoothingMetrics
from models import GnnModel


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    learningRate: float = 0.01
    weightDecay: float = 5e-4
    maxEpochs: int = 1000
    patience: int = 100
    seed: int = 0


def SetSeed(seed: int) -> None:
    """Seeds torch, random, and numpy. Deterministic-algorithm flags are not set."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def MacroF1(predictions: Tensor, targets: Tensor, numClasses: int) -> float:
    """Vectorized macro-F1 via a bincount confusion matrix.

    Matches sklearn's f1_score(average='macro', zero_division=0): a class with
    no predicted or no actual instances contributes 0 to the mean rather than
    raising or being excluded from it, over the full numClasses label set.
    """
    combinedIndex = targets * numClasses + predictions
    confusion = torch.bincount(combinedIndex, minlength=numClasses * numClasses)
    confusion = confusion.reshape(numClasses, numClasses).float()

    truePositive = confusion.diag()
    predictedPositive = confusion.sum(dim=0)  # column sums: predicted as class c
    actualPositive = confusion.sum(dim=1)  # row sums: actually class c

    precision = torch.where(predictedPositive > 0, truePositive / predictedPositive, torch.zeros_like(truePositive))
    recall = torch.where(actualPositive > 0, truePositive / actualPositive, torch.zeros_like(truePositive))
    denominator = precision + recall
    perClassF1 = torch.where(denominator > 0, 2 * precision * recall / denominator, torch.zeros_like(denominator))
    return float(perClassF1.mean())


def EvaluateSplit(model: GnnModel, data: Data, mask: Tensor) -> tuple[float, float, float]:
    """(loss, accuracy, macroF1) in eval mode under no_grad, on masked rows only."""
    model.eval()
    with torch.no_grad():
        logits, _ = model.Forward(data.x, data.edge_index)
        maskedLogits = logits[mask]
        maskedTargets = data.y[mask]
        loss = float(F.cross_entropy(maskedLogits, maskedTargets))
        predictions = maskedLogits.argmax(dim=1)
        accuracy = float((predictions == maskedTargets).float().mean())
        macroF1 = MacroF1(predictions, maskedTargets, numClasses=logits.shape[1])
    return loss, accuracy, macroF1


def CaptureMetrics(model: GnnModel, data: Data, metricsInstrument: OversmoothingMetrics) -> dict[str, Any]:
    """One explicit eval-mode forward; detaches layerEmbeddings to CPU at the
    capture site, never inside models. Returns the capture block plus
    bandIndices, for the caller to hoist to the top level once."""
    model.eval()
    with torch.no_grad():
        _, layerEmbeddings = model.Forward(data.x, data.edge_index)
    detached = [h.detach().cpu() for h in layerEmbeddings]
    layerMetrics = metricsInstrument.ComputeAll(detached)
    return {
        "dirichletEnergy": layerMetrics.dirichletEnergy,
        "mad": layerMetrics.mad,
        "frobeniusSquared": layerMetrics.frobeniusSquared,
        "contractionSlope": layerMetrics.contractionSlope,
        "bandIndices": layerMetrics.bandIndices,
    }


def _BuildRunId(model: GnnModel, config: TrainConfig) -> str:
    modelConfig = model.ConfigRecord()
    mitigations = sorted(modelConfig["mitigations"])
    mitigationsStem = "+".join(mitigations) if mitigations else "none"
    return f"{modelConfig['convType']}_{mitigationsStem}_d{modelConfig['numLayers']}_s{config.seed}"


def TrainRun(model: GnnModel, data: Data, config: TrainConfig, metricsInstrument: OversmoothingMetrics) -> dict[str, Any]:
    SetSeed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learningRate, weight_decay=config.weightDecay)

    epoch0Capture = CaptureMetrics(model, data, metricsInstrument)
    bandIndices = epoch0Capture.pop("bandIndices")

    trainingCurve: list[dict[str, Any]] = []
    bestValLoss = float("inf")
    bestEpoch = 0
    bestState = copy.deepcopy(model.state_dict())
    epochsWithoutImprovement = 0
    epochsRun = 0

    for epoch in range(1, config.maxEpochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model.Forward(data.x, data.edge_index)
        trainLoss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        trainLoss.backward()
        optimizer.step()

        valLoss, valAccuracy, _ = EvaluateSplit(model, data, data.val_mask)
        trainingCurve.append(
            {"epoch": epoch, "trainLoss": trainLoss.item(), "valLoss": valLoss, "valAccuracy": valAccuracy}
        )
        epochsRun = epoch

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            bestEpoch = epoch
            bestState = copy.deepcopy(model.state_dict())
            epochsWithoutImprovement = 0
        else:
            epochsWithoutImprovement += 1
            if epochsWithoutImprovement >= config.patience:
                break

    # final-epoch capture on the weights the loop ended with, BEFORE any
    # restore: reversing this would silently make finalMetrics a duplicate of
    # checkpointMetrics on every run
    finalCapture = CaptureMetrics(model, data, metricsInstrument)
    finalCapture.pop("bandIndices")

    model.load_state_dict(bestState)

    checkpointCapture = CaptureMetrics(model, data, metricsInstrument)
    checkpointCapture.pop("bandIndices")

    testLoss, testAccuracy, testMacroF1 = EvaluateSplit(model, data, data.test_mask)
    _, valAccuracyAtCheckpoint, _ = EvaluateSplit(model, data, data.val_mask)
    del testLoss  # part of EvaluateSplit's return signature; not written into the record

    return {
        "runId": _BuildRunId(model, config),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            **model.ConfigRecord(),
            "learningRate": config.learningRate,
            "weightDecay": config.weightDecay,
            "maxEpochs": config.maxEpochs,
            "patience": config.patience,
            "seed": config.seed,
        },
        "bandIndices": bandIndices,
        "results": {
            "testAccuracy": testAccuracy,
            "testMacroF1": testMacroF1,
            "valAccuracy": valAccuracyAtCheckpoint,
            "valLoss": bestValLoss,
            "bestEpoch": bestEpoch,
            "epochsRun": epochsRun,
        },
        "trainingCurve": trainingCurve,
        "epoch0Metrics": epoch0Capture,
        "checkpointMetrics": checkpointCapture,
        "finalMetrics": finalCapture,
        "trajectory": [],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchGeometric": torch_geometric.__version__,
            "device": str(next(model.parameters()).device),
        },
    }
