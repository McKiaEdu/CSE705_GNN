"""Test plan for experiments_spec.md."""

from __future__ import annotations

import glob
import json
import os

import pytest

from data import LoadCora
from experiments import BuildGrid, BuildModel, ResultPath, RunConfig, RunOne, RunSweep
from metrics import OversmoothingMetrics

ARM_COUNTS = {"A": 150, "B": 200, "C": 50, "E": 10, "F": 24}


@pytest.fixture(scope="module")
def cora():
    return LoadCora()


@pytest.fixture(scope="module")
def coraMetrics(cora):
    return OversmoothingMetrics(cora.edge_index, cora.num_nodes)


@pytest.mark.parametrize("arm,expected", list(ARM_COUNTS.items()))
def test_grid_arithmetic(arm: str, expected: int) -> None:
    assert len(BuildGrid(arm)) == expected


def test_grid_arithmetic_arm_d() -> None:
    assert len(BuildGrid("D", armDMitigation=["residual"])) == 100


def test_arm_d_requires_mitigation() -> None:
    with pytest.raises(ValueError):
        BuildGrid("D")


def test_grid_purity() -> None:
    first = BuildGrid("A")
    second = BuildGrid("A")
    assert first == second
    assert first is not second


def test_no_duplicate_result_paths(tmp_path) -> None:
    # arms E and F write to their own subdirectories, and arm F additionally
    # needs its hyperparameters in the filename itself (D-040): the C3 stem
    # does not encode hiddenDim or the training hyperparameters, so within one
    # flat directory arm E collides exactly with arm A's gcn-depth-2 subset,
    # and arm F's 8 per-seed hyperparameter combinations collapse onto one
    # filename each even within their own subdirectory -- verified by hand
    # before being treated as settled
    paths: list[str] = []
    for arm in ("A", "B", "C"):
        paths.extend(ResultPath(config, str(tmp_path)) for config in BuildGrid(arm))
    paths.extend(ResultPath(config, str(tmp_path)) for config in BuildGrid("D", armDMitigation=["residual"]))
    paths.extend(ResultPath(config, str(tmp_path / "fidelity")) for config in BuildGrid("E"))
    paths.extend(
        ResultPath(config, str(tmp_path / "hpsearch"), includeHyperparams=True) for config in BuildGrid("F")
    )

    assert len(paths) == len(set(paths))
    assert len(paths) == 534


def test_arm_e_and_f_collide_within_a_flat_directory(tmp_path) -> None:
    # the failure mode D-040 fixes, pinned down as a regression test: without
    # separate subdirectories, arm E exactly duplicates 10 of arm A's paths and
    # arm F's 24 configs collapse onto 3 already-claimed paths
    flatPaths = [ResultPath(c, str(tmp_path)) for c in BuildGrid("A") + BuildGrid("E") + BuildGrid("F")]
    assert len(set(flatPaths)) == 150  # arm A's 150 unique paths; E and F add none


def test_arm_f_still_collides_within_its_own_subdirectory_without_hyperparams_in_path(tmp_path) -> None:
    # subdirectory isolation alone is not enough for arm F (D-040's same-day
    # refinement): its 8 hyperparameter combinations per seed are
    # indistinguishable by convType/mitigations/depth/seed alone
    armFPaths = [ResultPath(c, str(tmp_path / "hpsearch")) for c in BuildGrid("F")]
    assert len(set(armFPaths)) == 3  # only the 3 distinct seeds distinguish anything

    armFPathsWithHyperparams = [
        ResultPath(c, str(tmp_path / "hpsearch"), includeHyperparams=True) for c in BuildGrid("F")
    ]
    assert len(set(armFPathsWithHyperparams)) == 24


def test_jk_readout_consistency_guard(cora, coraMetrics) -> None:
    badConfig = RunConfig(
        convType="gcn",
        numLayers=2,
        mitigations=["jk"],
        readout="lastLayer",
        hiddenDim=64,
        dropout=0.5,
        learningRate=0.01,
        weightDecay=5e-4,
        maxEpochs=5,
        patience=5,
        seed=0,
    )
    with pytest.raises(ValueError):
        RunOne(badConfig, cora, coraMetrics)


def test_embedding_flags_across_full_grid() -> None:
    allConfigs = BuildGrid("A") + BuildGrid("B") + BuildGrid("C") + BuildGrid("E") + BuildGrid("F")
    allConfigs += BuildGrid("D", armDMitigation=["residual"])

    flagged = [c for c in allConfigs if c.saveEmbeddings]
    assert len(flagged) == 10
    for config in flagged:
        assert config.convType == "gcn"
        assert config.seed == 0
        assert config.numLayers in (2, 32)

    flaggedMitigationSets = {tuple(sorted(c.mitigations)) for c in flagged}
    expectedArms = {(), ("residual",), ("pairnorm",), ("jk",), ("pairnorm", "residual")}
    assert flaggedMitigationSets == expectedArms


def test_save_embeddings_not_serialized(cora, coraMetrics) -> None:
    base = dict(
        convType="gcn",
        numLayers=2,
        mitigations=[],
        readout="lastLayer",
        hiddenDim=64,
        dropout=0.5,
        learningRate=0.01,
        weightDecay=5e-4,
        maxEpochs=5,
        patience=5,
        seed=0,
    )
    configA = RunConfig(**base, saveEmbeddings=False)
    configB = RunConfig(**base, saveEmbeddings=True)

    recordA = RunOne(configA, cora, coraMetrics)
    recordB = RunOne(configB, cora, coraMetrics)
    recordB.pop("_embeddingsToSave", None)

    recordA.pop("timestamp")
    recordB.pop("timestamp")
    assert json.dumps(recordA, sort_keys=True) == json.dumps(recordB, sort_keys=True)


def test_idempotency(tmp_path, cora, coraMetrics) -> None:
    configs = BuildGrid("F")[:2]
    resultsDir = str(tmp_path)

    RunSweep(configs, resultsDir)
    paths = [ResultPath(c, resultsDir) for c in configs]
    mtimesBefore = [os.path.getmtime(p) for p in paths]
    assert all(os.path.exists(p) for p in paths)

    RunSweep(configs, resultsDir)
    mtimesAfter = [os.path.getmtime(p) for p in paths]
    assert mtimesBefore == mtimesAfter


def test_force_overrides_skip(tmp_path) -> None:
    configs = BuildGrid("F")[:2]
    resultsDir = str(tmp_path)

    RunSweep(configs, resultsDir)
    paths = [ResultPath(c, resultsDir) for c in configs]
    mtimesBefore = [os.path.getmtime(p) for p in paths]

    RunSweep(configs, resultsDir, force=True)
    mtimesAfter = [os.path.getmtime(p) for p in paths]
    assert mtimesBefore != mtimesAfter


def test_record_conformance_end_to_end(tmp_path) -> None:
    config = RunConfig(
        convType="gcn",
        numLayers=2,
        mitigations=[],
        readout="lastLayer",
        hiddenDim=64,
        dropout=0.5,
        learningRate=0.01,
        weightDecay=5e-4,
        maxEpochs=5,
        patience=5,
        seed=0,
    )
    resultsDir = str(tmp_path)
    RunSweep([config], resultsDir)

    path = ResultPath(config, resultsDir)
    with open(path) as f:
        record = json.load(f)

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

    reconstructed = RunConfig(
        convType=record["config"]["convType"],
        numLayers=record["config"]["numLayers"],
        mitigations=record["config"]["mitigations"],
        readout=record["config"]["readout"],
        hiddenDim=record["config"]["hiddenDim"],
        dropout=record["config"]["dropout"],
        learningRate=record["config"]["learningRate"],
        weightDecay=record["config"]["weightDecay"],
        maxEpochs=record["config"]["maxEpochs"],
        patience=record["config"]["patience"],
        seed=record["config"]["seed"],
    )
    assert reconstructed == RunConfig(**{**config.__dict__, "saveEmbeddings": False})


def test_hook_order_independent_of_mitigations_list_order() -> None:
    configSorted = RunConfig(
        convType="gcn",
        numLayers=4,
        mitigations=["pairnorm", "residual"],
        readout="lastLayer",
        hiddenDim=64,
        dropout=0.5,
        learningRate=0.01,
        weightDecay=5e-4,
        maxEpochs=5,
        patience=5,
        seed=0,
    )
    configReversed = RunConfig(**{**configSorted.__dict__, "mitigations": ["residual", "pairnorm"]})

    for config in (configSorted, configReversed):
        model = BuildModel(config)
        names = [type(hook).__name__ for hook in model.layerHooks]
        assert names == ["ResidualHook", "PairNormHook"]


def test_arm_e_width() -> None:
    for config in BuildGrid("E"):
        assert config.hiddenDim == 16
    for arm in ("A", "C", "F"):
        for config in BuildGrid(arm):
            assert config.hiddenDim == 64


def test_embedding_tensor_count(tmp_path) -> None:
    # the ten D-031-flagged configs, sped up with tiny maxEpochs/patience -- this
    # test asserts file-writing mechanics (D-001 C1's band-derivation showing up
    # in the file system), not learning quality, so full training is unneeded
    flaggedConfigs = [c for c in BuildGrid("A") + BuildGrid("B") if c.saveEmbeddings]
    assert len(flaggedConfigs) == 10
    fastConfigs = [RunConfig(**{**c.__dict__, "maxEpochs": 3, "patience": 3}) for c in flaggedConfigs]

    resultsDir = str(tmp_path)
    RunSweep(fastConfigs, resultsDir)

    embeddingFiles = glob.glob(os.path.join(resultsDir, "embeddings", "*.pt"))
    assert len(embeddingFiles) == 16


def test_interrupt_safety_leaves_no_partial_file(tmp_path) -> None:
    from experiments.runner import _AtomicWrite

    path = os.path.join(str(tmp_path), "test.json")

    def _RaisingWrite(f) -> None:
        f.write("{incomplete")
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError):
        _AtomicWrite(path, _RaisingWrite)

    assert not os.path.exists(path)
    assert glob.glob(os.path.join(str(tmp_path), "*.tmp")) == []
