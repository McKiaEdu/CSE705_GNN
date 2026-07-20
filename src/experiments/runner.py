"""BuildModel, RunOne, RunSweep, ResultPath — construction and orchestration.

The only component that writes to results/ (D-022, D-030). Realizes D-030
(idempotent, atomic writes) and D-039 (record-and-continue failure handling).

ResultPath and RunSweep are arm-agnostic: whoever orchestrates the six arms is
responsible for calling RunSweep(BuildGrid("E"), "results/fidelity") and
RunSweep(BuildGrid("F"), "results/hpsearch") separately from
RunSweep(BuildGrid(arm), "results") for arm in A-D (D-040) -- arm E's
hiddenDim=16 and arm F's varying hyperparameters are not encoded in the C3
filename stem, so run into one flat directory they silently collide with arm A.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from typing import Sequence

import torch
from torch_geometric.data import Data

from data import LoadCora
from metrics import OversmoothingMetrics
from mitigations import JkReadout, PairNormHook, ResidualHook
from models import GatModel, GcniiModel, GcnModel, GnnModel, LastLayerReadout, LayerHook, Readout, SageModel
from train import SetSeed, TrainConfig, TrainRun

from .grid import OUT_DIM, IN_DIM, RunConfig

_CONV_MODEL_CLASSES: dict[str, type[GnnModel]] = {"gcn": GcnModel, "sage": SageModel, "gat": GatModel}

# Chen et al., arXiv:2007.02133, Table 6, Cora row: alpha_l: 0.1, lambda: 0.5
# (verified by fetching, not from memory). GCN2Conv calls lambda "theta".
GCNII_ALPHA = 0.1
GCNII_LAMBDA = 0.5


def _BuildLayerHooks(mitigations: Sequence[str]) -> list[LayerHook]:
    """Residual before PairNorm regardless of input order (D-007), by construction."""
    hooks: list[LayerHook] = []
    if "residual" in mitigations:
        hooks.append(ResidualHook())
    if "pairnorm" in mitigations:
        hooks.append(PairNormHook())
    return hooks


def _BuildReadout(readoutName: str, hiddenDim: int, outDim: int) -> Readout:
    if readoutName == "jk":
        return JkReadout(hiddenDim=hiddenDim, outDim=outDim)
    if readoutName == "lastLayer":
        return LastLayerReadout()
    raise ValueError(f"unknown readout: {readoutName!r}")


def BuildModel(config: RunConfig) -> GnnModel:
    layerHooks = _BuildLayerHooks(config.mitigations)
    readout = _BuildReadout(config.readout, config.hiddenDim, OUT_DIM)
    if config.convType == "gcnii":
        return GcniiModel(
            numLayers=config.numLayers,
            inDim=IN_DIM,
            hiddenDim=config.hiddenDim,
            outDim=OUT_DIM,
            dropout=config.dropout,
            layerHooks=layerHooks,
            readout=readout,
            alpha=GCNII_ALPHA,
            theta=GCNII_LAMBDA,
        )
    ModelClass = _CONV_MODEL_CLASSES[config.convType]
    return ModelClass(
        numLayers=config.numLayers,
        inDim=IN_DIM,
        hiddenDim=config.hiddenDim,
        outDim=OUT_DIM,
        dropout=config.dropout,
        layerHooks=layerHooks,
        readout=readout,
    )


def ResultPath(config: RunConfig, resultsDir: str, includeHyperparams: bool = False) -> str:
    """The sole expression of the C3 filename convention.

    `includeHyperparams` appends learningRate/dropout/weightDecay to the stem
    (D-040) -- needed only for arm F, whose 8 hyperparameter combinations per
    seed are otherwise indistinguishable from each other by convType/mitigations
    /depth/seed alone. Reads the config's own field values directly rather than
    comparing against the DEFAULT_* module constants, which are mutated in
    place once arm F's winner is known and would make the suffix's presence
    depend on call-time global state instead of the config itself.
    """
    mitigationsStem = "+".join(sorted(config.mitigations)) if config.mitigations else "none"
    stem = f"{config.convType}_{mitigationsStem}_d{config.numLayers}_s{config.seed}"
    if includeHyperparams:
        stem += f"_lr{config.learningRate}_do{config.dropout}_wd{config.weightDecay}"
    return os.path.join(resultsDir, f"{stem}.json")


def RunOne(config: RunConfig, data: Data, metricsInstrument: OversmoothingMetrics) -> dict:
    """Builds the model, calls TrainRun, returns the record. Does not write.

    When config.saveEmbeddings is set, the returned dict carries a transient
    "_embeddingsToSave" key (D-031) that RunSweep pops off and writes as .pt
    files before serializing the record -- the written JSON record itself is
    unaffected and still conforms to C3 exactly.
    """
    isJkInMitigations = "jk" in config.mitigations
    isJkReadout = config.readout == "jk"
    if isJkInMitigations != isJkReadout:
        raise ValueError(f"jk/readout consistency violated: mitigations={config.mitigations}, readout={config.readout!r}")

    # TrainRun's own SetSeed call only fixes training-loop randomness (dropout
    # masks etc.); model construction happens before that and draws on whatever
    # RNG state is ambient at the call site, so seeding here too is what makes
    # a run reproducible from its seed alone rather than from call order
    SetSeed(config.seed)
    model = BuildModel(config)
    trainConfig = TrainConfig(
        learningRate=config.learningRate,
        weightDecay=config.weightDecay,
        maxEpochs=config.maxEpochs,
        patience=config.patience,
        seed=config.seed,
    )
    record = TrainRun(model, data, trainConfig, metricsInstrument)

    if config.saveEmbeddings:
        # TrainRun leaves model at its checkpoint weights when it returns
        # (D-020); one more eval-mode forward on those same weights recovers
        # the raw tensors CaptureMetrics only reduced to scalars, without
        # changing train/'s own capture contract (C5)
        model.eval()
        with torch.no_grad():
            _, layerEmbeddings = model.Forward(data.x, data.edge_index)
        bandIndices = record["bandIndices"]
        indicesToSave = sorted({bandIndices[0], bandIndices[-1]})
        record["_embeddingsToSave"] = {index: layerEmbeddings[index].detach().cpu() for index in indicesToSave}

    return record


def _AtomicWrite(path: str, writeFn) -> None:
    """Writes via a temporary file in the same directory, then renames, so a
    file is either complete or absent (D-030)."""
    directory = os.path.dirname(path) or "."
    fd, tmpPath = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as tmpFile:
            writeFn(tmpFile)
        os.replace(tmpPath, path)
    except BaseException:
        os.remove(tmpPath)
        raise


def _AtomicWriteJson(path: str, data: dict) -> None:
    _AtomicWrite(path, lambda f: json.dump(data, f))


def _SaveEmbeddings(runId: str, embeddingsToSave: dict[int, torch.Tensor], resultsDir: str) -> None:
    embeddingsDir = os.path.join(resultsDir, "embeddings")
    os.makedirs(embeddingsDir, exist_ok=True)
    for index, tensor in embeddingsToSave.items():
        torch.save(tensor, os.path.join(embeddingsDir, f"{runId}_l{index}.pt"))


def RunSweep(
    configs: list[RunConfig],
    resultsDir: str,
    force: bool = False,
    includeHyperparamsInPath: bool = False,
) -> None:
    """Iterates configs, skipping any whose result already exists unless
    force=True. Failures are recorded, not raised (D-039): the sweep continues.

    `includeHyperparamsInPath` is passed through to ResultPath (D-040); the
    driver sets it True only when sweeping arm F.
    """
    os.makedirs(resultsDir, exist_ok=True)

    data = LoadCora()
    metricsInstrument = OversmoothingMetrics(data.edge_index, data.num_nodes)

    executed = 0
    skipped = 0
    failed = 0

    for config in configs:
        path = ResultPath(config, resultsDir, includeHyperparams=includeHyperparamsInPath)
        failurePath = path[: -len(".json")] + ".failed.json"

        if not force and (os.path.exists(path) or os.path.exists(failurePath)):
            skipped += 1
            continue

        try:
            record = RunOne(config, data, metricsInstrument)
        except Exception as error:  # noqa: BLE001 -- deliberately broad, D-039
            failed += 1
            failureRecord = {
                "runId": os.path.basename(path)[: -len(".json")],
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            _AtomicWriteJson(failurePath, failureRecord)
            print(f"FAILED: {path} -- {error}")
            continue

        embeddingsToSave = record.pop("_embeddingsToSave", None)
        _AtomicWriteJson(path, record)
        if embeddingsToSave is not None:
            _SaveEmbeddings(record["runId"], embeddingsToSave, resultsDir)

        executed += 1
        print(f"OK: {path}")

    print(f"sweep complete: {executed} executed, {skipped} skipped, {failed} failed, {len(configs)} total")
