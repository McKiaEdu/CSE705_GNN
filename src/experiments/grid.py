"""RunConfig and BuildGrid — the declarative sweep grid (D-027 through D-029, D-031).

Pure enumeration: no side effects, no model construction, no I/O. The runner
(runner.py) is the only thing that executes what this module describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

IN_DIM = 1433
OUT_DIM = 7

DEPTHS = [2, 4, 8, 16, 32]
SEEDS = list(range(10))

DEFAULT_HIDDEN_DIM = 64  # D-023, every arm except E
FIDELITY_HIDDEN_DIM = 16  # D-023's published-baseline reproduction, arm E only

# provisional D-029 published-baseline values; arm F's winner overwrites these
# three module constants once the search completes and is aggregated
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_DROPOUT = 0.5
DEFAULT_WEIGHT_DECAY = 5e-4

DEFAULT_MAX_EPOCHS = 1000  # D-018
DEFAULT_PATIENCE = 100  # D-018

ARM_B_MITIGATIONS: list[list[str]] = [["residual"], ["pairnorm"], ["jk"], ["pairnorm", "residual"]]

ARM_F_SEEDS = [0, 1, 2]
ARM_F_LEARNING_RATES = [0.005, 0.01]
ARM_F_DROPOUTS = [0.5, 0.6]
ARM_F_WEIGHT_DECAYS = [5e-4, 1e-3]


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    convType: str
    numLayers: int
    mitigations: list[str] = field(default_factory=list)
    readout: str
    hiddenDim: int
    dropout: float
    learningRate: float
    weightDecay: float
    maxEpochs: int
    patience: int
    seed: int
    saveEmbeddings: bool = False


def _ReadoutFor(mitigations: list[str]) -> str:
    return "jk" if "jk" in mitigations else "lastLayer"


def _ShouldSaveEmbeddingsForArmA(convType: str, depth: int, seed: int) -> bool:
    # D-031: baseline half of the ten flagged runs -- gcn, seed 0, depth in {2, 32}
    return convType == "gcn" and seed == 0 and depth in (2, 32)


def _ShouldSaveEmbeddingsForArmB(depth: int, seed: int) -> bool:
    # D-031: mitigated half -- arm B is already gcn-only, so only seed/depth gate
    return seed == 0 and depth in (2, 32)


def _BuildArmA() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for convType in ("gcn", "sage", "gat"):
        for depth in DEPTHS:
            for seed in SEEDS:
                configs.append(
                    RunConfig(
                        convType=convType,
                        numLayers=depth,
                        mitigations=[],
                        readout="lastLayer",
                        hiddenDim=DEFAULT_HIDDEN_DIM,
                        dropout=DEFAULT_DROPOUT,
                        learningRate=DEFAULT_LEARNING_RATE,
                        weightDecay=DEFAULT_WEIGHT_DECAY,
                        maxEpochs=DEFAULT_MAX_EPOCHS,
                        patience=DEFAULT_PATIENCE,
                        seed=seed,
                        saveEmbeddings=_ShouldSaveEmbeddingsForArmA(convType, depth, seed),
                    )
                )
    return configs


def _BuildArmB() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for mitigations in ARM_B_MITIGATIONS:
        for depth in DEPTHS:
            for seed in SEEDS:
                configs.append(
                    RunConfig(
                        convType="gcn",
                        numLayers=depth,
                        mitigations=list(mitigations),
                        readout=_ReadoutFor(mitigations),
                        hiddenDim=DEFAULT_HIDDEN_DIM,
                        dropout=DEFAULT_DROPOUT,
                        learningRate=DEFAULT_LEARNING_RATE,
                        weightDecay=DEFAULT_WEIGHT_DECAY,
                        maxEpochs=DEFAULT_MAX_EPOCHS,
                        patience=DEFAULT_PATIENCE,
                        seed=seed,
                        saveEmbeddings=_ShouldSaveEmbeddingsForArmB(depth, seed),
                    )
                )
    return configs


def _BuildArmC() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for depth in DEPTHS:
        for seed in SEEDS:
            configs.append(
                RunConfig(
                    convType="gcnii",
                    numLayers=depth,
                    mitigations=[],
                    readout="lastLayer",
                    hiddenDim=DEFAULT_HIDDEN_DIM,
                    dropout=DEFAULT_DROPOUT,
                    learningRate=DEFAULT_LEARNING_RATE,
                    weightDecay=DEFAULT_WEIGHT_DECAY,
                    maxEpochs=DEFAULT_MAX_EPOCHS,
                    patience=DEFAULT_PATIENCE,
                    seed=seed,
                    saveEmbeddings=False,
                )
            )
    return configs


def _BuildArmD(armDMitigation: list[str]) -> list[RunConfig]:
    configs: list[RunConfig] = []
    for convType in ("sage", "gat"):
        for depth in DEPTHS:
            for seed in SEEDS:
                configs.append(
                    RunConfig(
                        convType=convType,
                        numLayers=depth,
                        mitigations=list(armDMitigation),
                        readout=_ReadoutFor(armDMitigation),
                        hiddenDim=DEFAULT_HIDDEN_DIM,
                        dropout=DEFAULT_DROPOUT,
                        learningRate=DEFAULT_LEARNING_RATE,
                        weightDecay=DEFAULT_WEIGHT_DECAY,
                        maxEpochs=DEFAULT_MAX_EPOCHS,
                        patience=DEFAULT_PATIENCE,
                        seed=seed,
                        saveEmbeddings=False,
                    )
                )
    return configs


def _BuildArmE() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for seed in SEEDS:
        configs.append(
            RunConfig(
                convType="gcn",
                numLayers=2,
                mitigations=[],
                readout="lastLayer",
                hiddenDim=FIDELITY_HIDDEN_DIM,
                dropout=DEFAULT_DROPOUT,
                learningRate=DEFAULT_LEARNING_RATE,
                weightDecay=DEFAULT_WEIGHT_DECAY,
                maxEpochs=DEFAULT_MAX_EPOCHS,
                patience=DEFAULT_PATIENCE,
                seed=seed,
                saveEmbeddings=False,
            )
        )
    return configs


def _BuildArmF() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for learningRate in ARM_F_LEARNING_RATES:
        for dropout in ARM_F_DROPOUTS:
            for weightDecay in ARM_F_WEIGHT_DECAYS:
                for seed in ARM_F_SEEDS:
                    configs.append(
                        RunConfig(
                            convType="gcn",
                            numLayers=2,
                            mitigations=[],
                            readout="lastLayer",
                            hiddenDim=DEFAULT_HIDDEN_DIM,
                            dropout=dropout,
                            learningRate=learningRate,
                            weightDecay=weightDecay,
                            maxEpochs=DEFAULT_MAX_EPOCHS,
                            patience=DEFAULT_PATIENCE,
                            seed=seed,
                            saveEmbeddings=False,
                        )
                    )
    return configs


def BuildGrid(arm: str, armDMitigation: list[str] | None = None) -> list[RunConfig]:
    """Pure enumeration of one arm's configurations; no I/O, no model construction.

    `armDMitigation` is required only for arm D, since it is not known until
    arm B's aggregate result names the most effective mitigation (open question
    in experiments_spec.md, resolved by the caller, not here).
    """
    if arm == "A":
        return _BuildArmA()
    if arm == "B":
        return _BuildArmB()
    if arm == "C":
        return _BuildArmC()
    if arm == "D":
        if armDMitigation is None:
            raise ValueError(
                "arm D requires armDMitigation, the mitigation arm B found most effective "
                "-- not known until B's results are aggregated"
            )
        return _BuildArmD(armDMitigation)
    if arm == "E":
        return _BuildArmE()
    if arm == "F":
        return _BuildArmF()
    raise ValueError(f"unknown arm: {arm!r}")
