"""Test plan for models_spec.md."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data import LoadCora
from models import BuildConv, GatModel, GcniiModel, GcnModel, LastLayerReadout, SageModel

ARCHITECTURES = {"gcn": GcnModel, "sage": SageModel, "gat": GatModel}
ALL_ARCHITECTURES = {**ARCHITECTURES, "gcnii": GcniiModel}
DEPTHS = [2, 4, 8, 16, 32]
HIDDEN_DIM = 64
OUT_DIM = 7
IN_DIM = 1433

# Arbitrary valid GCN2Conv hyperparameters, used only to exercise GcniiModel's code
# paths in these tests. The real value is an experiments-level decision (D-034),
# not settled here.
GCNII_TEST_ALPHA = 0.1
GCNII_TEST_THETA = 0.5


class _StubJkLikeReadout:
    """Minimal Readout test double with FinalLayerIsLogits = False.

    Stands in for mitigations' real JkReadout, which is not built yet (mitigations
    comes after train/ in the build order per models_spec.md Dependencies). This
    tests that GnnModel behaves correctly for any conforming Readout, not the
    specific JK aggregation logic, which belongs to mitigations' own test plan.
    """

    FinalLayerIsLogits = False

    def __init__(self, hiddenDim: int, outDim: int) -> None:
        self.projection = nn.Linear(hiddenDim, outDim)

    def Apply(self, layerEmbeddings: list[Tensor]) -> Tensor:
        return self.projection(layerEmbeddings[-1])


def _BuildModel(convType: str, numLayers: int, readout=None, dropout: float = 0.0):
    readout = readout or LastLayerReadout()
    if convType == "gcnii":
        return GcniiModel(
            numLayers=numLayers,
            inDim=IN_DIM,
            hiddenDim=HIDDEN_DIM,
            outDim=OUT_DIM,
            dropout=dropout,
            layerHooks=[],
            readout=readout,
            alpha=GCNII_TEST_ALPHA,
            theta=GCNII_TEST_THETA,
        )
    ModelClass = ARCHITECTURES[convType]
    return ModelClass(
        numLayers=numLayers,
        inDim=IN_DIM,
        hiddenDim=HIDDEN_DIM,
        outDim=OUT_DIM,
        dropout=dropout,
        layerHooks=[],
        readout=readout,
    )


@pytest.fixture(scope="module")
def cora():
    return LoadCora()


@pytest.mark.parametrize("convType", ALL_ARCHITECTURES.keys())
@pytest.mark.parametrize("depth", DEPTHS)
def test_shape_contract(cora, convType: str, depth: int) -> None:
    torch.manual_seed(0)
    model = _BuildModel(convType, depth)
    model.eval()
    logits, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    assert logits.shape == (cora.num_nodes, OUT_DIM)
    assert len(layerEmbeddings) == depth + 1
    assert layerEmbeddings[0] is cora.x


def test_final_entry_identity_last_layer_readout(cora) -> None:
    torch.manual_seed(0)
    model = _BuildModel("gcn", 4, readout=LastLayerReadout())
    model.eval()
    logits, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    assert layerEmbeddings[-1] is logits


def test_final_entry_identity_jk_like_readout(cora) -> None:
    torch.manual_seed(0)
    readout = _StubJkLikeReadout(HIDDEN_DIM, OUT_DIM)
    model = _BuildModel("gcn", 4, readout=readout)
    model.eval()
    logits, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    assert layerEmbeddings[-1].shape[1] == HIDDEN_DIM
    assert logits.shape[1] == OUT_DIM
    assert layerEmbeddings[-1] is not logits


def test_width_homogeneity_last_layer_readout(cora) -> None:
    torch.manual_seed(0)
    depth = 8
    model = _BuildModel("gcn", depth, readout=LastLayerReadout())
    model.eval()
    _, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    band = layerEmbeddings[1 : depth - 1 + 1]  # 1 .. numLayers - 1
    widths = {t.shape[1] for t in band}
    assert len(widths) == 1


def test_width_homogeneity_jk_like_readout(cora) -> None:
    torch.manual_seed(0)
    depth = 8
    readout = _StubJkLikeReadout(HIDDEN_DIM, OUT_DIM)
    model = _BuildModel("gcn", depth, readout=readout)
    model.eval()
    _, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    band = layerEmbeddings[1 : depth + 1]  # 1 .. numLayers under JK
    widths = {t.shape[1] for t in band}
    assert len(widths) == 1


def test_activation_consistency_jk_like_readout(cora) -> None:
    torch.manual_seed(0)
    readout = _StubJkLikeReadout(HIDDEN_DIM, OUT_DIM)
    model = _BuildModel("gcn", 4, readout=readout)
    model.eval()
    _, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    assert bool((layerEmbeddings[-1] >= 0).all())


def test_activation_consistency_last_layer_readout(cora) -> None:
    torch.manual_seed(0)
    model = _BuildModel("gcn", 4, readout=LastLayerReadout())
    model.eval()
    logits, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    assert layerEmbeddings[-1] is logits
    assert bool((logits < 0).any())


def test_null_object_path_identity(cora) -> None:
    torch.manual_seed(0)
    model = _BuildModel("gcn", 3, readout=LastLayerReadout())
    model.eval()
    logits, layerEmbeddings = model.Forward(cora.x, cora.edge_index)

    # reference forward that never touches self.layerHooks at all, replicating
    # the loop manually against the same submodules and weights
    h = cora.x
    x0 = cora.x
    referenceEmbeddings = [cora.x]
    for l, conv in enumerate(model.convs, start=1):
        h = conv(h, cora.edge_index, x0)
        isFinalAndLogits = l == model.numLayers
        if not isFinalAndLogits:
            h = torch.relu(h)
        referenceEmbeddings.append(h)
        if not isFinalAndLogits:
            h = model.dropoutLayer(h)  # identity in eval mode

    assert torch.equal(logits, referenceEmbeddings[-1])
    for a, b in zip(layerEmbeddings, referenceEmbeddings, strict=True):
        assert torch.equal(a, b)


def test_gradient_reaches_every_layer_under_jk_like_readout(cora) -> None:
    torch.manual_seed(0)
    readout = _StubJkLikeReadout(HIDDEN_DIM, OUT_DIM)
    model = _BuildModel("gcn", 4, readout=readout)
    model.train()
    logits, _ = model.Forward(cora.x, cora.edge_index)
    loss = logits.sum()
    loss.backward()
    for conv in model.convs:
        for param in conv.parameters():
            assert param.grad is not None
            assert bool((param.grad != 0).any())


def test_tap_point_precedes_dropout(cora) -> None:
    torch.manual_seed(0)
    model = _BuildModel("gcn", 4, readout=LastLayerReadout(), dropout=0.9)
    model.train()

    capturedInputs: dict[int, Tensor] = {}

    def MakeHook(idx: int):
        def Hook(module: nn.Module, args: tuple) -> None:
            capturedInputs[idx] = args[0]

        return Hook

    handles = [model.convs[i].register_forward_pre_hook(MakeHook(i)) for i in range(1, len(model.convs))]
    _, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    for handle in handles:
        handle.remove()

    # layerEmbeddings[1] is layer 1's pre-dropout output; convs[1] receives layer
    # 1's post-dropout output as its input, so at dropout=0.9 the two must differ
    assert not torch.equal(layerEmbeddings[1], capturedInputs[1])


@pytest.mark.parametrize("convType", ARCHITECTURES.keys())
def test_uniform_conv_signature_x0_ignored(cora, convType: str) -> None:
    torch.manual_seed(0)
    model = _BuildModel(convType, 2, readout=LastLayerReadout())
    model.eval()
    conv = model.convs[0]
    outWithRealX0 = conv(cora.x, cora.edge_index, cora.x)
    outWithZeroX0 = conv(cora.x, cora.edge_index, torch.zeros_like(cora.x))
    assert torch.equal(outWithRealX0, outWithZeroX0)


def test_gcnii_requires_equal_width() -> None:
    with pytest.raises(ValueError):
        BuildConv("gcnii", IN_DIM, HIDDEN_DIM, alpha=GCNII_TEST_ALPHA, theta=GCNII_TEST_THETA, layer=1)


def test_gcnii_final_entry_identity_last_layer_readout(cora) -> None:
    torch.manual_seed(0)
    model = _BuildModel("gcnii", 4, readout=LastLayerReadout())
    model.eval()
    logits, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    assert layerEmbeddings[-1] is logits
    assert logits.shape == (cora.num_nodes, OUT_DIM)


def test_gcnii_width_homogeneity_last_layer_readout(cora) -> None:
    torch.manual_seed(0)
    depth = 8
    model = _BuildModel("gcnii", depth, readout=LastLayerReadout())
    model.eval()
    _, layerEmbeddings = model.Forward(cora.x, cora.edge_index)
    band = layerEmbeddings[1:depth]  # 1 .. numLayers - 1
    widths = {t.shape[1] for t in band}
    assert widths == {HIDDEN_DIM}


def test_gcnii_numlayers_is_hop_count(cora) -> None:
    # D-034: numLayers GCN2Conv hops, input/output projections uncounted
    torch.manual_seed(0)
    depth = 5
    model = _BuildModel("gcnii", depth, readout=LastLayerReadout())
    assert len(model.convs) == depth


def test_gcnii_x0_changes_output(cora) -> None:
    # contrast to test_uniform_conv_signature_x0_ignored: x0 is load-bearing for
    # gcnii (the initial-residual term), unlike gcn/sage/gat
    torch.manual_seed(0)
    conv = BuildConv("gcnii", HIDDEN_DIM, HIDDEN_DIM, alpha=GCNII_TEST_ALPHA, theta=GCNII_TEST_THETA, layer=1)
    conv.eval()
    h = torch.randn(cora.num_nodes, HIDDEN_DIM)
    outWithRealX0 = conv(h, cora.edge_index, h)
    outWithZeroX0 = conv(h, cora.edge_index, torch.zeros_like(h))
    assert not torch.equal(outWithRealX0, outWithZeroX0)


@pytest.mark.slow
def test_smoke_two_layer_gcn_reaches_published_accuracy(cora) -> None:
    torch.manual_seed(0)
    model = GcnModel(
        numLayers=2,
        inDim=IN_DIM,
        hiddenDim=16,
        outDim=OUT_DIM,
        dropout=0.5,
        layerHooks=[],
        readout=LastLayerReadout(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        logits, _ = model.Forward(cora.x, cora.edge_index)
        loss = F.cross_entropy(logits[cora.train_mask], cora.y[cora.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model.Forward(cora.x, cora.edge_index)
        predictions = logits.argmax(dim=1)
        testAccuracy = float((predictions[cora.test_mask] == cora.y[cora.test_mask]).float().mean())

    print(f"\n2-layer GCN test accuracy: {testAccuracy:.4f}")
    assert testAccuracy > 0.75


@pytest.mark.slow
def test_smoke_32_layer_gcn_collapses(cora) -> None:
    torch.manual_seed(0)
    model = GcnModel(
        numLayers=32,
        inDim=IN_DIM,
        hiddenDim=HIDDEN_DIM,
        outDim=OUT_DIM,
        dropout=0.5,
        layerHooks=[],
        readout=LastLayerReadout(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        logits, _ = model.Forward(cora.x, cora.edge_index)
        loss = F.cross_entropy(logits[cora.train_mask], cora.y[cora.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model.Forward(cora.x, cora.edge_index)
        predictions = logits.argmax(dim=1)
        testAccuracy = float((predictions[cora.test_mask] == cora.y[cora.test_mask]).float().mean())

    print(f"\n32-layer GCN test accuracy: {testAccuracy:.4f}")
    assert testAccuracy < 0.60
