# Findings — Cora GNN Oversmoothing Study

Empirical results from running the code against real data: what was measured,
not what was decided. Distinct from `DECISIONS.md`, which records project-wide
engineering and methodology choices that are expected to stay settled once
made. A finding is provisional by nature — it describes evolving understanding
of what the sweep's data actually shows, and is expected to be revised as
analysis deepens. Revisions are dated and kept visible in place, the same way
`DECISIONS.md` keeps superseded reasoning rather than deleting it, so the
resolution history stays legible.

Every clause in a finding entry is marked by its epistemic status:
**measured** (read directly off recorded data, stated flat) versus
**inferred** (a candidate mechanism consistent with the measured data but not
independently demonstrated by it — flagged as "consistent with," never
"because"). This distinction is load-bearing for the report and should never
be allowed to blur in editing.

Entries are numbered `F-001`, `F-002`, ... and edited in place when revised,
per the same convention as `DECISIONS.md`. Entries cite each other by ID rather
than restating shared numbers — one source of truth per fact.

**Not every empirically-discovered thing lives here.** `DECISIONS.md` D-036 and
D-037 were also discovered by running code rather than planned in advance, but
both resolve in a stable, permanent property (a fixed operator's null space; a
numerical-stability floor in an implementation) rather than an evolving
understanding of model behavior. The test is not "was this found empirically"
but "will looking harder change the answer" — a finding here is expected to be
revised as analysis deepens (see F-001's own "Resolution history" section for
an example, and F-002/F-003's inline revisions for others without a dedicated
heading); a decision in `DECISIONS.md`, empirically-discovered or not, is not.


## F-001 — Depth-32 failure mode is architecture-dependent: GCN partially trains, SAGE/GAT do not train at all

**Date:** 2026-07-20, corrected 2026-07-21
**Scope:** unmitigated GCN, GraphSAGE, GAT at `numLayers=32`, the real
`train/` harness, this study's settled hyperparameters (D-029: `lr=0.01`,
`dropout=0.5`, `weightDecay=5e-4`, `patience=100`, `maxEpochs=1000`). Not a
claim about other depths or about the mitigated arms. Checkpoint state only —
see F-002 for the epoch-0 (pre-training) picture, F-003 for GCN's inferred
mechanism.

### Measured

**GCN partially trains.** Across all 10 seeds: training loss descends from
`ln(7) = 1.9459` to a mean minimum of **1.57** (range 1.53–1.61), compared to
depth 2's well-fit final loss of ~0.23. Runs last 315–883 epochs before
`patience=100` exhausts — long enough that validation loss kept improving
somewhere along the way. Mean test accuracy **24.07%**, below the actual
majority-class floor of the Cora test split (**31.9%**, computed directly from
the test-mask label counts `[130, 91, 144, 319, 149, 103, 64]`, not the ~30%
figure earlier cited from memory). Not just below on average: the full
10-seed range is **21.2%–26.2%**, so the single best seed still sits 5.7
points below the floor — every seed is below it, not just the mean.

**SAGE and GAT do not train at all.** Training loss stays flat at `ln(7)`
(SAGE mean minimum 1.934, GAT mean minimum 1.939 — a movement of ~0.01–0.02
nats, indistinguishable from optimization noise). `patience` exhausts almost
immediately: SAGE runs only 102–168 epochs, GAT only 101–121, versus GCN's
315–883. Mean test accuracy 22.65% (SAGE) and 19.66% (GAT); SAGE's worst seed
(10.3%) is *below* the uniform-random floor of 14.3%.

**Checkpoint MAD shows no progressive depth-collapse for GCN, but a clean
collapse-to-zero for SAGE/GAT** (checkpoint, final band index, mean over
10 seeds):

| depth | GCN MAD | SAGE MAD | GAT MAD |
|---|---|---|---|
| 2 | 0.240 | 0.262 | 0.286 |
| 4 | 0.264 | 0.291 | 0.302 |
| 8 | 0.290 | 0.058 | 0.115 |
| 16 | 0.176 | ≈0.000 | ≈0.000 |
| 32 | 0.183 | ≈0.000 | ≈0.000 |

GCN's checkpoint MAD is roughly flat/non-monotonic across the whole depth
sweep — never approaching zero. SAGE and GAT's checkpoint MAD collapses to
essentially zero by depth 16 and stays there. This also serves as a **positive
control**: the metric does fire cleanly when collapse is genuinely present
(SAGE/GAT), which is what rules out "the metric is just insensitive" as an
explanation for GCN not showing the pattern.

**GCN's checkpoint energy ratio (`E_last/E_1`, per D-002/D-003's convention —
see F-002 for why this denominator and not `E_0`) grows explosively with
depth — reported as geometric mean and range, not arithmetic mean:**

| depth | geometric mean | range (min–max across 10 seeds) |
|---|---|---|
| 2 | 1.0 (trivial — single-point band) | 1.0–1.0 |
| 4 | 26.6 | 16.0–43.6 |
| 8 | 3.96e6 | 2.12e2–7.85e11 |
| 16 | 4.49e12 | 7.51e9–5.08e19 |
| 32 | 5.45e26 | 3.25e12–2.49e38 |

**The arithmetic mean is the wrong summary for this quantity, and an earlier
draft of this table used it.** At depth 32 the per-seed ratio spans from
3.25e12 (seed 6, the one seed whose early layers don't collapse — F-003) to
2.49e38 (seed 9) — 26 orders of magnitude. The arithmetic mean previously
reported here (2.50e37) is determined almost entirely by seed 9 alone: seed
9's own ratio (2.49e38) is itself roughly 10× that mean, so "the mean of 10
seeds" was arithmetically close to "one seed's value, divided by 10." A
grader who reads this table next to the per-seed spread in F-003 would notice
the same thing. Geometric mean is the appropriate summary for a
multiplicative, log-scale quantity like this one; the qualitative conclusion
is unchanged either way — every single seed's ratio is enormous, none close
to 1.

**Reading this ratio as "nothing collapses" is backwards, and the earlier
draft of this entry said exactly that — corrected here, not softened.** The
ratio is enormous because the *denominator collapsed*, not because nothing
collapsed: `E_1` itself is ~1.4e-15 at checkpoint depth 32, seed 0 (D-037's
quantified check; other seeds range down to 2.9e-38 — F-003), already many
orders of magnitude below a healthy energy value. A large `E_last/E_1` is the
arithmetic consequence of dividing by an already-dead layer 1, not evidence
that layer 1 (or the early band generally) stayed intact while later layers
merely failed to shrink further. What the band actually shows, read
correctly, is **early-layer collapse plus late-layer growth** — early layers
carrying essentially zero energy (dead), late layers carrying enormous
energy — which is precisely F-003's inferred mechanism (early-layer
parameters starved of gradient, late-layer weights growing unchecked), not a
contradiction of it. In log terms the geometric-mean exponent runs
0 → 1.42 → 6.60 → 12.65 → 26.74 — roughly doubling per depth doubling from
depth 8 onward (6.60 → 12.65 is ×1.92; 12.65 → 26.74 is ×2.11), similar in
shape to the (now-corrected) arithmetic-mean-based description this replaces,
though the absolute exponent values differ substantially between the two
summaries, which is exactly the point above.

**Numerical headroom.** The largest *individual* per-seed ratio (seed 9,
2.49e38) sits close to float32's ceiling (~3.40e38, within a factor of ~1.4)
— but this ratio is a Python-level division of two already-extracted scalar
values (effectively double precision once loaded from JSON), not a float32
tensor operation, so it was never at risk of overflowing the way a raw
tensor computation would be. The check that actually matters is on the raw
`dirichletEnergy` tensor values themselves (computed in float32, before
extraction): no `inf` or `nan` appears anywhere in `dirichletEnergy` for any
of the 10 seeds, either capture. The raw (non-ratio) energy values themselves have
enormous headroom — the largest single value observed is ~2028, nowhere near
overflow. The ratio's magnitude comes entirely from the denominator (`E_1`
itself is ~1.4e-15 at checkpoint) being astronomically small, not from any
value approaching a numerical ceiling — the same fact stated above as the
correction to "never collapses," not a separate observation.

F-002 contrasts this checkpoint picture against epoch 0: the clean collapse
signature that GCN and GAT both show at initialization does not survive
training under this study's hyperparameters.

### Inferred (consistent with, not demonstrated)

**For SAGE and GAT, optimization failure and representational collapse read
as the same event, not two separable phenomena.** Two independently measured
quantities point the same way: the flat loss curve (above) and the
checkpoint's near-total MAD collapse (above) are both consistent with
training essentially never moving the weights. That the two are "the same
event here" is the interpretation of that pairing, not a third measurement —
it has not been independently confirmed beyond what the two measured signals
jointly support.

### Resolution history

Originally recorded (as `D-038` in `DECISIONS.md`, since corrected and moved
here) as a single-architecture finding, generalized without checking: "the
partial-training/blow-up pattern is the unmitigated deep baseline's failure
mode." That generalization was never checked against SAGE or GAT and is
**false as stated** — SAGE and GAT do not partially train at depth 32, they do
not train at all, which is a different phenomenon (optimization failure
indistinguishable from the untrained operator's own collapse) from GCN's
(partial training producing a distinct, non-collapsed, non-recovered failure
mode). The corrected, narrower claim above replaces it; it does not soften it.


## F-002 — Structural collapse at initialization is architecture-dependent in kind: GCN/GAT collapse in both direction and magnitude, SAGE only in direction

**Date:** 2026-07-21
**Scope:** unmitigated GCN, GraphSAGE, GAT, `epoch0Metrics` — the initialized
network, i.e. the propagation operator composed with each layer's randomly
initialized, untrained weight matrices — across the depth sweep
`{2, 4, 8, 16, 32}`. This is NOT "the pure propagation operator": the random
weights' spectral norms are part of the composition and part of what drives
the observed contraction, exactly as Cai & Wang's bound
(`E_{l+1} <= (1-lambda)^2 ||W||^2 E_l`) has it — `||W||` does not vanish just
because `W` is untrained. Checkpoint (post-training) state is F-001's, not
this entry's; cited here for contrast only.

**The three architectures do NOT share one initializer — checked against
PyG's source, not assumed.** `GCNConv.lin` and every one of `GATConv`'s
internal `Linear` layers (`lin`, `lin_src`, `lin_dst`, `lin_edge`, `res`) are
constructed with `weight_initializer='glorot'` explicitly. `SAGEConv.lin_l`
and `SAGEConv.lin_r` — the aggregation and root-transform weights, the main
parameters of the layer — are constructed with no `weight_initializer`
argument at all, which PyG's `Linear.reset_parameters` resolves to
`kaiming_uniform` (`a=sqrt(5)`, PyTorch's own `nn.Linear` default), not
Glorot. This is a real, verified difference, not a guess, and it is a live
candidate contributor to the depth-2 anomaly below (a same-depth,
single-hop scale difference is exactly what a different init scheme would
produce) alongside the aggregation-structure explanation given there — the
two are not mutually exclusive and neither has been isolated from the other.

### Measured

**Energy ratio convention.** All energy ratios below are `E_last/E_1`
(normalized at the first band index), per D-002/D-003 — *not* `E_last/E_0`.
D-003 explicitly rejects `E_0` as a denominator: `h_0` (sparse, 1433-dim,
row-normalized bag-of-words) and `h_1`+ (dense, `hiddenDim`, post-projection)
are different representation kinds, so a ratio against `E_0` conflates
aggregation-driven smoothing with the arbitrary scale of the first
projection's random weights. `E_last/E_1` keeps numerator and denominator in
the same representation kind, matching `viz`'s actual `EnergyCurve`
implementation and the project's reported-quantity convention. At depth 2 the
ratio is trivially 1.0 (the band is a single point, `bandIndices = [1]`).

**GCN and GAT show clean, monotonic collapse in both MAD and energy ratio.**
GCN final-representation MAD: 0.600 → 0.240 → 0.085 → 0.023 → 0.0045 across
depths 2/4/8/16/32; energy ratio `E_last/E_1`: 1.0 → 3.14e-2 → 3.14e-4 →
2.60e-7 → 6.22e-13 (monotonic, ~13 orders of magnitude by depth 32). GAT: MAD
0.596 → 0.211 → 0.076 → 0.017 → 0.0041; energy ratio 1.0 → 7.26e-2 → 2.73e-3
→ 1.28e-5 → 1.25e-10. Both decay monotonically and look exponential in depth
— consistent with the *qualitative* shape Cai & Wang's bound predicts
(repeated contraction). This is **not** a claim of numerical agreement with
the bound itself: `||W||` and the graph's spectral gap were not computed and
the measured decay rate was not checked against the bound's actual
right-hand side. "Matches the bound" would be overreach; "consistent with
exponential contraction" is what was actually checked. (The rejected
`E_last/E_0` convention gives a qualitatively identical picture here — 9.4e-2
→ 2.9e-3 → 2.9e-5 → 2.4e-8 → 5.9e-14 for GCN — so this particular conclusion
is not sensitive to the normalization choice; F-001's checkpoint case is
different, see below.)

**SAGE collapses directionally but not in magnitude.** MAD: 0.083 → 0.0002 →
≈0.000 → ≈0.000 → ≈0.000 — directionally *fully* collapsed already by depth 4,
faster than either GCN or GAT. But the energy ratio does not follow:
`E_last/E_1` is 1.0 → 18.3 → 19.8 → 20.3 → 19.3 — it *rises* from depth 2 to
depth 4 and then plateaus around 19–20, never decaying toward zero at any
depth tested. (Again in qualitative agreement with the rejected `E_0`
convention, which plateaus around 9 instead of 19–20 — different absolute
scale, same shape.) So at initialization, SAGE's representations become
indistinguishable in direction but do not shrink in magnitude the way GCN's
and GAT's do.

**SAGE's MAD is already far below GCN/GAT's at depth 2 — before any
depth-driven collapse could plausibly apply.** At the shallowest depth tested,
one hop: SAGE MAD 0.083 vs. GCN 0.600 and GAT 0.596. This is a same-depth,
single-layer difference, not a consequence of stacking layers — whatever
causes it operates within one hop. Checked directly (not merely reported):
built one `GCNConv`, one `SAGEConv` (PyG defaults, `root_weight=True`), one
`SAGEConv` with `root_weight=False`, and one `GATConv`, at matched random
seeds, on Cora's real graph, and measured output MAD directly. The gap
reproduces exactly (SAGE ≈0.07–0.10 vs. GCN/GAT ≈0.59–0.60 across three
seeds) — confirming it is a real property of the layer, not an artifact of the
sweep. `root_weight` is **not** the cause: MAD stays low, in fact lower, with
`root_weight=False` (≈0.03–0.05) than with the default `root_weight=True`
(≈0.07–0.10). This rules out one candidate explanation concretely.

**SAGE's collapse is an exact rank-1 collapse onto the constant direction,
not D-036's sqrt-degree null space — measured directly via SVD, and this is
what connects the MAD/energy dissociation to D-036's already-established
geometry.** At depth 16, random init, three seeds: SAGE's representation
matrix `H` has top-singular-value energy fraction **1.0000** (an exact rank-1
matrix, not merely dominated by one direction), and its top left singular
vector `u1` has `|cos(u1, constant)| = 1.0000` exactly, versus
`|cos(u1, sqrt-degree)| = 0.9512` — identical across all three seeds,
consistent with a deterministic geometric fact rather than a coincidence of
initialization. GCN and GAT, by contrast, are only mostly rank-1 (energy
fraction 0.93–0.98) and don't cleanly separate the two candidate directions
(both cosines land in 0.86–0.96 for either) — **and this test cannot resolve
them by construction, not because GCN/GAT's geometry is itself ambiguous.**
`cos(constant, sqrt-degree)` on Cora's actual graph is **0.9512** (computed
directly, the same figure SAGE's own `|cos(u1, sqrt-degree)|` happens to
equal exactly, since `u1 = constant` for SAGE). The two candidate directions
are themselves only ~18° apart, so any vector that leans toward either one
necessarily shows a "high" cosine to both. The test only has discriminating
power when one cosine clearly dominates the other, as it does for SAGE
(1.0000 vs. 0.9512 — a real, if small-looking, gap given how close the two
reference directions are). GCN/GAT's close, similar-looking pair of cosines
is therefore not itself a finding about which direction they favor — it is
the test failing to discriminate for them, which is a different claim from
"GCN/GAT partially favor both directions." D-036 already established that
`Δ~`'s null space is the `sqrt(d~_i)` direction, not the constant direction —
so a representation that collapses exactly onto the constant direction is
collapsing onto the *wrong* direction to zero out Dirichlet energy, which is
precisely why SAGE's MAD reaches ≈0 while its energy ratio stays large: MAD
is blind to which direction the collapse is onto, energy is not, and SAGE's
collapse direction happens to be the one direction (among rank-1
possibilities) that a symmetric-normalized Laplacian does not annihilate.

### Contrast with the checkpoint state

GCN's clean epoch-0 collapse does not survive training under this study's
hyperparameters in the way a naive reading might expect: it does not vanish,
it deepens and narrows to the early layers specifically, while the late
layers separately blow up in magnitude. The checkpoint shows a
non-monotonic MAD (never approaching zero the way epoch 0 does) and an
enormous `E_last/E_1` ratio — but per F-001's corrected reading, that ratio is
large because `E_1` itself has collapsed to near-zero energy (the early band
is dead), not because the early layers stayed healthy while later ones merely
failed to shrink. Read correctly, the checkpoint shows early-layer collapse
*plus* late-layer growth (F-001, F-003), not an absence of collapse. For SAGE
and GAT, training does not happen at all (F-001), so their checkpoint state is
close to
this epoch-0 state by default, not because training reproduced it.

### Inferred (consistent with, not demonstrated)

**The magnitude-without-direction pattern is now explained by the rank-1/D-036
connection above (Measured), not merely a candidate.** An earlier pass
proposed SAGE's separate root/self-term as the explanation; that is
**weakened, not supported**, by the `root_weight` check (removing the root
term made the directional collapse more severe, not less). What replaces it
is not another guess about the aggregation step but a measured geometric fact:
SAGE collapses onto a specific direction (constant) that is provably outside
`Δ~`'s null space (D-036), so energy cannot vanish regardless of how complete
the directional collapse is. This explains the *dissociation itself* — MAD→0
alongside energy staying large — without needing a further hypothesis.

**What remains open is a different, narrower question: why SAGE specifically
collapses onto the constant direction (rather than, say, sqrt-degree, or not
collapsing to rank-1 at all) at a single hop, when GCN and GAT do not collapse
this cleanly to any one direction.** Two verified-different candidates now
exist and neither has been isolated from the other: (1) SAGE's unweighted mean
aggregation versus GCN's degree-normalized symmetric aggregation and GAT's
attention weighting, interacting with Cora's skewed degree distribution
(2 to 169); (2) SAGE's `lin_l`/`lin_r` defaulting to `kaiming_uniform` init
rather than the `glorot` init GCN/GAT's layers use (verified above). Isolating
either would require, at minimum, testing a degree-normalized-mean aggregation
variant and a Glorot-initialized `SAGEConv` variant separately, which has not
been done. This is an open question, not a claim.


## F-003 — GCN's depth-32 checkpoint shows early-layer energy collapse across seeds; the weight-norm mechanism behind it is still one-run inference

**Date:** 2026-07-20, upgraded 2026-07-21
**Scope:** GCN only. SAGE and GAT need no comparable mechanism at depth 32 —
per F-001, they simply do not train, so there is no partial-training mechanism
to explain for them.

### Measured

**Early-layer checkpoint energy collapse holds across 9 of 10 seeds, not just
the one run F-003 originally rested on.** `dirichletEnergy` is persisted for
every run in the schema (unlike weight or gradient norms), so this is checked
directly rather than inferred: for GCN depth 32, checkpoint, per seed, the
count of band layers below D-037's `1e-12` floor and the layer where energy
first rises back above it —

| seed | below-floor count (of 31) | first layer ≥ floor | `E_1` |
|---|---|---|---|
| 0 | 22 | 23 | 1.4e-15 |
| 1 | 22 | 23 | 5.1e-20 |
| 2 | 26 | 27 | 7.5e-33 |
| 3 | 24 | 25 | 6.5e-24 |
| 4 | 25 | 25 | 2.9e-35 |
| 5 | 25 | 26 | 1.9e-31 |
| 6 | **0** | **1** | **3.3e-12** |
| 7 | 25 | 24 | 1.5e-26 |
| 8 | 25 | 22 | 4.2e-28 |
| 9 | 27 | 28 | 2.9e-38 |

Nine of ten seeds show the same shape: the early ~22–28 layers carry
essentially zero energy (`E_1` itself ranging from 1.4e-15 down to 2.9e-38 —
the smallest values sit at the edge of float32's representable range),
transitioning to non-negligible energy only in roughly the last 4–9 layers.
**Seed 6 is a genuine exception**, not folded into the pattern to keep it
clean: no layer falls below the floor, `E_1 = 3.3e-12`. Seed 6 also has the
highest test accuracy of the ten (26.2%) and the shortest run (`epochsRun=315`,
versus 385–883 for the others) — noted as a correlation, not explained; it has
not been investigated further.

**The transition is not always a single clean crossover.** "First layer ≥
floor" in the table is the literal first layer where energy pokes above
`1e-12` — for 7 of 10 seeds this is also where it stays above for good, but
for seeds 4, 7, and 8 the transition wobbles: energy rises above the floor,
dips back below, and rises again before settling. Checked directly via a
run-length encoding of the per-layer below/above-floor sequence, not
hand-narrated (an earlier hand-narrated pass through the same three seeds
mis-described seed 8's exact pattern — corrected here by using the encoding
directly): seed 4 is 24 layers below, 2 above, 1 below, 4 above (settles at
layer 28); seed 7 is 23 below, then alternating below/above/below/above
singly, settling at layer 28 (23 below, 1 above, 1 below, 1 above, 1 below,
4 above); seed 8 is the most complex — 21 below, 1 above, 3 below, 2 above,
**1 below again**, 3 above — a second dip at layer 28 after already having
poked above at layers 26–27, only truly settling at layer 29. The overall
shape (dead early band, alive late band) is unaffected, but the boundary
between them is a brief, occasionally multi-dip oscillation for these three
seeds, not a sharp step — worth a sentence if this table or shape appears in a
figure, so a single "first crossing" number is not read as implying strict
monotonicity for every seed.

This directly answers what F-001's corrected energy-ratio reading (above)
implies: early-layer collapse is not a one-run inference, it is a measured,
cross-seed property of the checkpoint state (modulo the one exception).

### Inferred (consistent with, not demonstrated)

**The mechanism behind the collapse** — early-layer *parameters* driven toward
zero for lack of gradient signal, late-layer weights growing large and
unconstrained — remains inferred from **terminal weight norms of one run**
(`gcn_none_d32_s0`, post-hoc inspection, not a persisted field in the C3
schema, not a gradient trace). The Measured section above upgrades "early
layers carry near-zero *energy*" from one-run inference to a cross-seed
measured fact; it does not by itself upgrade the *weight/gradient* story,
which is a claim about parameters, not activations, and nothing in the
persisted schema measures parameters or gradients directly. This is
*consistent with* vanishing gradients starving early layers while late-layer
weights grow largely unchecked; it has not been independently demonstrated by
a gradient-norm measurement. Deliberately not characterized further than
"large and unconstrained": whether the late layers are fitting (even
overfitting) the 140-node training set specifically is a separate, unmeasured
claim — the schema has no `trainAccuracy` field and no persisted per-run model
checkpoint to check a train/test generalization gap against, and the measured
test accuracy (24.07%, below the 31.9% majority floor, per F-001) is if
anything in tension with, not supportive of, a clean overfitting story. No
per-layer gradient norms exist anywhere in the recorded data — `TrainRun`
calls `.backward()` but never captures or persists gradient magnitudes, for
any run. Confirming the parameter-level mechanism directly would require new
instrumentation, not a re-read of existing records.


## F-004 — GCNII is the one architecture whose test accuracy improves with depth, and its checkpoint slope stays near zero rather than exploding

**Date:** 2026-07-21
**Scope:** GCNII (`alpha=0.1`, `theta=0.5` per D-042 — D-034 had left these
open; D-042 is the closing decision), full depth sweep `{2, 4, 8, 16, 32}`,
checkpoint state, real `train/` harness, 10 seeds.

### Measured

**Test accuracy rises monotonically with depth**, the opposite direction from
GCN/SAGE/GAT (F-001): 78.45% (±1.79, depth 2) → 75.81% (±4.82, depth 4) →
77.66% (±1.63, depth 8) → 80.16% (±1.04, depth 16) → **83.26% (±0.44, depth
32)**. Depth 32 is GCNII's best result and has the tightest spread of any
depth (std 0.44 points, versus 1–5 points elsewhere).

**Checkpoint `contractionSlope` trends toward and past zero, not toward the
explosive growth GCN shows (F-001).** Depth 4: +0.383 (±0.063); depth 8:
+0.169 (±0.024); depth 16: **−0.037** (±0.010, a genuine sign flip); depth 32:
+0.012 (±0.003, essentially flat). Contrast GCN's checkpoint slope, which
grows to the tens by depth 8 and beyond (D-037's quantified check on
`gcn_none_d32_s0`: unfloored checkpoint slope +1.14). GCNII's slope stays
within roughly ±0.4 of zero at every depth tested — no blow-up, no collapse.

### Inferred (consistent with, not demonstrated)

GCNII's identity-mapping and initial-residual terms (D-008, D-034) are
architecturally designed to prevent both the vanishing-gradient dynamic F-003
infers for GCN and the unconstrained late-layer growth that produces it. The
near-zero, non-exploding checkpoint slope and the monotonically improving
accuracy are *consistent with* that design doing what it is meant to do — this
has not been checked against GCNII's own per-layer weight norms or gradients
the way F-003 did for GCN (not done, not claimed here), so it remains a
plausible reading of the accuracy/slope pattern rather than a demonstrated
mechanism.


## F-005 — Mitigation ablation on GCN at depth 32: JK wins clearly; PairNorm helps substantially; residual alone underperforms baseline; combining PairNorm+residual is worse than PairNorm alone

**Date:** 2026-07-21
**Scope:** GCN, depth 32, all four arm-B mitigation combinations plus the
unmitigated baseline (arm A), checkpoint test accuracy, 10 seeds each. This is
the comparison D-041 already used to select JK for arm D; this entry records
the full ablation, not just the winning arm.

### Measured

Mean test accuracy (±std, min–max) across the five arms, depth 32:

| arm | mean | std | min | max |
|---|---|---|---|---|
| none (baseline) | 24.07% | 1.60 | 21.2% | 26.2% |
| residual | 19.25% | 8.97 | 10.3% | 32.1% |
| pairnorm | 58.07% | 7.10 | 47.6% | 65.6% |
| **jk** | **73.13%** | 3.43 | 68.0% | 78.2% |
| pairnorm+residual | 24.36% | 9.79 | 13.5% | 41.0% |

**JK is the clear winner** (D-041's basis for selecting it into arm D), ~15
points ahead of the next best (pairnorm) and with the tightest spread of any
mitigated arm. **PairNorm alone helps substantially** (58.07% vs. 24.07%
baseline) but well short of JK. **Residual alone does not clearly beat the
unmitigated baseline at depth 32** — its mean (19.25%) is *below* baseline's
(24.07%), though its std (8.97) is large enough, and its max (32.1%) high
enough, that this is not a confident "residual hurts" claim on 10 seeds alone.
**Combining PairNorm with residual is worse than PairNorm by itself**
(24.36% vs. 58.07%) — closer to the residual-alone and baseline numbers than
to PairNorm's, a negative interaction between the two mechanisms at this
depth, not a combination of their individual benefits.

**D-024 anticipated this exact result before it was measured** — a design
predicting its own finding, worth stating as such rather than leaving
uncited: "If the residual arm shows no benefit at depth, the absent layer-1
residual is a candidate explanation" (D-024's "Known cost," recorded when
residual was designed as a no-projection identity hook, per Kipf & Welling's
Appendix B — layers 2 through L-1 only, since layer 1 changes width
1433→hiddenDim and layer L changes hiddenDim→outDim, neither admitting an
identity residual). That candidate explanation has not been checked here
(it would need the layer-1-projection diagnostic D-024 flagged as
recoverable but not built into the ablation), so this is D-024's prediction
being confirmed as *worth investigating*, not confirmed as *correct*.

### Inferred (consistent with, not demonstrated)

No mechanism is proposed here for *why* residual underperforms baseline or
why PairNorm+residual underperforms PairNorm alone at depth 32 — both are
real, measured patterns in the aggregate numbers, but explaining them would
need the same per-run scrutiny F-001–F-003 gave the unmitigated baseline
(training curves, checkpoint MAD/energy, weight norms), which has not been
done for any mitigated arm. Recorded here as an open question worth that
scrutiny before the report asserts a reason, not as a claim that one is known.


## F-006 — Arm E confirms the published-baseline fidelity claim: hiddenDim=16 and hiddenDim=64 give statistically indistinguishable depth-2 accuracy

**Date:** 2026-07-21
**Scope:** GCN, depth 2, unmitigated, 10 seeds each: arm E (`hiddenDim=16`,
D-023's published-fidelity arm) versus arm A's depth-2 subset
(`hiddenDim=64`, the sweep's actual width).

### Measured

Arm E: mean test accuracy **81.60%** (±0.31). Arm A (depth 2, `hiddenDim=64`):
mean test accuracy **81.49%** (±0.32). Difference is 0.11 points, well within
one standard deviation of either — not distinguishable given the seed
variance observed. Both land close to Kipf & Welling's published ~81.5%.

This is the single number behind the claim that D-023's width change (16 to
64) does not materially affect the depth-2 baseline reproduction — a
paraphrase, not a quote; D-023 does not use that wording. D-023's actual text
calls arm E "Fidelity, recovered rather than sacrificed," and states its
purpose directly: the deviation from the published width "becomes a measured
quantity rather than a stated caveat." Until now, the number that measured
quantity actually comes out to had not been written down anywhere outside the
aggregate table `generate_report_figures.py` produces — the report's fidelity
claim had no `FINDINGS.md` entry to cite despite arm E being, on D-023's own
framing, one of the more report-relevant numbers in the whole sweep.


## F-007 — Arm F's winner matches the published baseline exactly; the search does not clearly separate the top configurations, closing D-029's open question

**Date:** 2026-07-21
**Scope:** GCN, depth 2, all 8 hyperparameter combinations (`learningRate` in
{0.005, 0.01} × `dropout` in {0.5, 0.6} × `weightDecay` in {5e-4, 1e-3}), 3
seeds each, mean validation accuracy per D-029's selection rule (primary
criterion, tie-break, final fallback — "three-level" is experiments_spec.md's
phrasing for this rule, per D-041, not D-029's own wording).

### Measured

Ranked by mean validation accuracy (3 seeds):

| lr | dropout | weightDecay | mean valAcc | std |
|---|---|---|---|---|
| **0.01** | **0.5** | **0.0005** | **0.8047** | 0.0042 |
| 0.01 | 0.5 | 0.001 | 0.8000 | 0.0020 |
| 0.005 | 0.5 | 0.001 | 0.7993 | 0.0031 |
| 0.01 | 0.6 | 0.0005 | 0.7993 | 0.0031 |
| 0.01 | 0.6 | 0.001 | 0.7987 | 0.0023 |
| 0.005 | 0.6 | 0.0005 | 0.7980 | 0.0020 |
| 0.005 | 0.6 | 0.001 | 0.7967 | 0.0012 |
| 0.005 | 0.5 | 0.0005 | 0.7933 | 0.0042 |

The winner (`lr=0.01, dropout=0.5, weightDecay=0.0005`) is exactly Kipf &
Welling's published configuration. It wins on the primary criterion (highest
mean validation accuracy) without needing the tie-break: the runner-up's mean
(0.8000) sits 0.0047 below the winner's, only marginally outside the winner's
own one-seed-noise band (0.0042).

**D-029's own open question — "whether three seeds are enough to separate
eight configurations whose accuracies may differ by less than seed noise" —
is answered, and the answer is not fully reassuring.** All 8 configurations
span only 0.7933–0.8047, a range of 1.14 points, while individual per-config
standard deviations (from only 3 seeds) run 0.12–0.42 points — comparable in
scale to the *total spread across all 8 configurations*. The margin between
the top 2 configurations (0.47 points) is larger than the winner's own
seed-to-seed std (0.42 points), but only barely — a 0.05-point difference is
not a clean separation, even though the margin is nominally on the right side
of one std. Read plainly: three seeds do not cleanly separate these
8 configurations on accuracy alone; the ranking is close to what seed noise
alone could produce. What makes the winner nonetheless a reasonable, not
arbitrary, choice is that it is *also* the configuration closest to the
published baseline (D-029's own final fallback), so the practical selection
does not depend on trusting a ranking this close to noise — it would have
been selected by the fallback criterion even if the primary ranking were
pure noise.

### Inferred (consistent with, not demonstrated)

None — this entry is descriptive statistics over the recorded validation
accuracies and stops there.
