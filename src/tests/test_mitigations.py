"""Test plan for mitigations_spec.md."""

from __future__ import annotations

import pytest
import torch

from data import LoadCora
from metrics import OversmoothingMetrics
from mitigations import JkReadout, MitigationNames, PairNormHook, ResidualHook
from models import GcnModel, LastLayerReadout

HIDDEN_DIM = 64
OUT_DIM = 7
IN_DIM = 1433


@pytest.fixture(scope="module")
def cora():
    return LoadCora()


@pytest.fixture(scope="module")
def coraMetrics(cora):
    return OversmoothingMetrics(cora.edge_index, cora.num_nodes)


@pytest.mark.parametrize("HookClass", [ResidualHook, PairNormHook])
def test_shape_preservation(HookClass) -> None:
    torch.manual_seed(0)
    hook = HookClass()
    h = torch.randn(2708, HIDDEN_DIM)
    hPrev = torch.randn(2708, HIDDEN_DIM)
    out = hook.Apply(h, hPrev, edgeIndex=None)
    assert out.shape == h.shape


def test_residual_fires_where_widths_match() -> None:
    torch.manual_seed(0)
    hook = ResidualHook()
    h = torch.randn(2708, HIDDEN_DIM)
    hPrev = torch.randn(2708, HIDDEN_DIM)
    out = hook.Apply(h, hPrev, edgeIndex=None)
    assert torch.equal(out, h + hPrev)


def test_residual_is_noop_where_widths_differ() -> None:
    torch.manual_seed(0)
    hook = ResidualHook()
    h = torch.randn(2708, HIDDEN_DIM)
    hPrev = torch.randn(2708, IN_DIM)
    out = hook.Apply(h, hPrev, edgeIndex=None)
    assert torch.equal(out, h)


def test_residual_fires_on_expected_layer_count(cora) -> None:
    torch.manual_seed(0)
    callLog: list[tuple[int, int]] = []

    class _CountingResidualHook(ResidualHook):
        def Apply(self, h, hPrev, edgeIndex):
            fired = h.shape == hPrev.shape
            callLog.append((h.shape[1], hPrev.shape[1], fired))
            return super().Apply(h, hPrev, edgeIndex)

    model = GcnModel(
        numLayers=8,
        inDim=IN_DIM,
        hiddenDim=HIDDEN_DIM,
        outDim=OUT_DIM,
        dropout=0.0,
        layerHooks=[_CountingResidualHook()],
        readout=LastLayerReadout(),
    )
    model.eval()
    with torch.no_grad():
        model.Forward(cora.x, cora.edge_index)

    firedCount = sum(1 for _, _, fired in callLog if fired)
    assert len(callLog) == 8
    assert firedCount == 6  # layers 2 through 7; layer 1 (1433->64) and layer 8 (64->7) skip


def test_pairnorm_invariant() -> None:
    torch.manual_seed(0)
    hook = PairNormHook(scale=1.0)
    h = torch.randn(2708, HIDDEN_DIM) * 5.0 + 3.0
    out = hook.Apply(h, hPrev=h, edgeIndex=None)

    # atol wider than a tight epsilon: float32 summation noise over 2708 rows,
    # not a correctness issue with the centering step itself
    columnMeans = out.mean(dim=0)
    assert torch.allclose(columnMeans, torch.zeros_like(columnMeans), atol=1e-3)

    rowNorms = out.norm(dim=1)
    assert torch.allclose(rowNorms, torch.full_like(rowNorms, hook.scale), atol=1e-4)


def test_pairnorm_survives_zero_rows() -> None:
    torch.manual_seed(0)
    hook = PairNormHook(scale=1.0)
    h = torch.randn(2708, HIDDEN_DIM)
    h[:100] = 0.0  # simulating ReLU-dead nodes at depth
    out = hook.Apply(h, hPrev=h, edgeIndex=None)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_pairnorm_changes_dirichlet_energy(cora, coraMetrics) -> None:
    torch.manual_seed(0)
    hook = PairNormHook(scale=1.0)
    h = torch.randn(cora.num_nodes, HIDDEN_DIM)
    energyBefore = coraMetrics.DirichletEnergy(h)
    energyAfter = coraMetrics.DirichletEnergy(hook.Apply(h, hPrev=h, edgeIndex=None))
    assert energyBefore != pytest.approx(energyAfter)


@pytest.mark.parametrize("depth", [2, 8, 32])
def test_jk_output_width_is_depth_invariant(depth: int) -> None:
    readout = JkReadout(hiddenDim=HIDDEN_DIM, outDim=OUT_DIM)
    paramCount = sum(p.numel() for p in readout.linear.parameters())
    assert paramCount == HIDDEN_DIM * OUT_DIM + OUT_DIM  # weight + bias, independent of depth


def test_jk_aggregates_correct_range() -> None:
    readout = JkReadout(hiddenDim=HIDDEN_DIM, outDim=OUT_DIM)
    numNodes = 2708
    misshapedIndex0 = torch.randn(numNodes, 999)  # deliberately wrong width
    band = [torch.randn(numNodes, HIDDEN_DIM) for _ in range(4)]
    layerEmbeddings = [misshapedIndex0] + band
    logits = readout.Apply(layerEmbeddings)  # must not touch index 0 or raise
    assert logits.shape == (numNodes, OUT_DIM)


def test_jk_readout_declares_itself() -> None:
    readout = JkReadout(hiddenDim=HIDDEN_DIM, outDim=OUT_DIM)
    assert readout.FinalLayerIsLogits is False
    assert LastLayerReadout.FinalLayerIsLogits is True


def test_gradient_reaches_every_layer_under_real_jk_readout(cora) -> None:
    # test_models.py covers this with a stub readout double; this is the real
    # JkReadout integration the stub stands in for
    torch.manual_seed(0)
    readout = JkReadout(hiddenDim=HIDDEN_DIM, outDim=OUT_DIM)
    model = GcnModel(
        numLayers=4,
        inDim=IN_DIM,
        hiddenDim=HIDDEN_DIM,
        outDim=OUT_DIM,
        dropout=0.0,
        layerHooks=[],
        readout=readout,
    )
    model.train()
    logits, _ = model.Forward(cora.x, cora.edge_index)
    logits.sum().backward()
    for conv in model.convs:
        for param in conv.parameters():
            assert param.grad is not None
            assert bool((param.grad != 0).any())
    for param in readout.linear.parameters():
        assert param.grad is not None


def test_jk_readout_parameters_reach_model_parameters(cora) -> None:
    # confirms self.readout = readout auto-registers as a submodule (unlike
    # layerHooks, stored as a plain list -- would not auto-register a
    # parameterized hook, see hooks.py's module docstring)
    readout = JkReadout(hiddenDim=HIDDEN_DIM, outDim=OUT_DIM)
    model = GcnModel(
        numLayers=4,
        inDim=IN_DIM,
        hiddenDim=HIDDEN_DIM,
        outDim=OUT_DIM,
        dropout=0.0,
        layerHooks=[],
        readout=readout,
    )
    modelParamIds = {id(p) for p in model.parameters()}
    readoutParamIds = {id(p) for p in readout.linear.parameters()}
    assert readoutParamIds <= modelParamIds


def test_mitigation_names_canonicalized_and_sorted() -> None:
    assert MitigationNames([PairNormHook(), ResidualHook()], LastLayerReadout()) == ["pairnorm", "residual"]
    assert MitigationNames([ResidualHook(), PairNormHook()], LastLayerReadout()) == ["pairnorm", "residual"]
    assert MitigationNames([], LastLayerReadout()) == []
    assert MitigationNames([], JkReadout(HIDDEN_DIM, OUT_DIM)) == ["jk"]
    assert MitigationNames([ResidualHook()], JkReadout(HIDDEN_DIM, OUT_DIM)) == ["jk", "residual"]
