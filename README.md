# Sampling_IRIS

**A certified sampling layer for IRIS / GCS convex-region libraries — and the
first measurement of what those libraries actually contain.**

### ▶ [Video results — px-cps.github.io/Sampling_IRIS](https://px-cps.github.io/Sampling_IRIS/)

Five annotated result videos: an aerial planner flying through walls on the
library as shipped versus the certified route beside it, a dual-arm
pick-and-place delivered twice, the worst task violation in full, a certified
positive control, and a frame-by-frame forensic audit of one benchmark query.

Every planner in the graphs-of-convex-sets (GCS) family rests on one sentence:
*points inside the regions are collision-free*. The regions come from a
sampling-based generator (IRIS-ZO / IRIS-NP) whose contract is explicitly
**probabilistic** — at confidence `1-δ`, the colliding volume fraction is at
most `ε`, with `ε > 0` by construction. The planning side consumes that
contract as if it were deterministic, and never re-checks the trajectory it
returns.

This repository contains the code that measures the gap, explains why it
cannot be closed at the library level, and closes it at the answer level.

---

## The finding

| quantity | value | what it measures |
|---|---|---|
| interface-sample contamination (14-DOF library) | **3.2 %** | uniform measure — darts thrown into the regions |
| random-query reference optima colliding | **12.5 %** (2/16) | argmin, generic endpoints |
| task-query reference optima colliding | **62 %** (18/29) | argmin, real pick-and-place endpoints |
| worst penetration of a delivered "optimal" path | **−91.3 mm** | median over colliding pairs: −39.4 mm |

A 3.2 % volume error becomes a 62 % answer error because the planner is not a
sampler: it is an `argmin` over the modeled set, i.e. an active search for
exactly the model surplus that shortens paths. Three mechanisms compound —
taut paths patrol the boundary shell where the residual lives; whenever the
through-pocket shortcut is genuinely shorter *every* optimum of the modeled
problem is infeasible; and task configurations (grasp poses hugging shelves)
live in the dirtiest real estate of the library.

**Fixing the library does not work.** Three independent routes to volume-level
certainty were run to completion; all three destroy the connectivity that
planning needs before they deliver soundness where the tasks are:

| route | certainty delivered | measured cost to the graph |
|---|---|---|
| strict Bernoulli gate at generation (`ε = 0.1 %`, enforced) | 10× tighter, probabilistic | 219 → **47** regions, 2757 → **0** portals |
| SOS certificates (C-IRIS, 2-DOF testbed) | deterministic | 7 disconnected islands, **0/30** queries solved |
| uniform margins (erosion sweep) | none (3.4 % → 2.6 %) | queries disconnect first |

## The layer

Certainty about a *d*-dimensional volume must be paid wherever the volume
touches obstacles. Certainty about the *1*-dimensional path the robot actually
executes is paid only along that path. The layer composes two lineages on top
of the IRIS skeleton:

* **Safety certificates** (Bialkowski/Karaman/Frazzoli; Safe Bubble Cover) —
  *one distance query, one ball exempt*: clearance `φ(q)` and the robot's
  Lipschitz constant `L` make the ball of radius `φ(q)/L` provably safe.
  Overlapping balls chained along a segment are a **continuous** proof: no
  resolution parameter, hence no resolution loophole. Long strides in open
  space, fine steps near obstacles.
* **Lazy collision checking** (LazyPRM, LazySP) — *check only the roads the
  route uses*: most edges are never used by any shortest path, so validate the
  returned path only, repair or blacklist what fails, re-search.

Pipeline: interface classification → area/contour sampling → informed
(admissible) pruning → anytime search → **validate only the candidate** →
in-region repair (convexity keeps via-points legal) or edge blacklist →
corridor SOCP polish → **re-validate** (a polish is an `argmin` too) →
certified path with an ε-optimality certificate, or an honest "no path".

### Head-to-head (29 task queries, same library, same machine)

| | reference (GCS\*) | ours |
|---|---|---|
| delivered answers failing certification | **21 / 29** (18 probe-confirmed) | **0** |
| certified answers / honest no-paths | 0 / 0 | 15 / 14 |
| time to first answer | 1.59 s (unvalidated) | **1.14 s** (certified) |
| cost ratio on pairs both solve cleanly | 1.0000 | 1.0000 (digit-for-digit) |
| price of truth where the reference is wrong | — | median **+14 %** |

Certificates cost ~200 clearance queries (2.3 ms each) per path, 0.02–0.06 s —
cheap precisely because the library is 97 % clean. Sparsity makes validation
affordable; adversarial placement of the residual makes it necessary.

## Video

One two-minute result video, [`docs/media/icra_video.mp4`](docs/media/icra_video.mp4),
plays on the project page, **<https://px-cps.github.io/Sampling_IRIS/>**.
Every scene shows the reference planner's answer on the left and the
certified answer on the right, with clearance read from the benchmark's own
collision checker and the carried objects inside the collision model: the
aerial benchmark (reference optimum through building walls, certified reroute
0.9 % shorter), a 7-DOF pick-and-place (reference hand and bottle 68 mm
through a shelf board, certified path +19 mm clear), and the 14-DOF dual-arm
transfer and handover (reference gripper 49 mm into the shelf top, ours
certified throughout).

## Layout

```
docs/      the project page (GitHub Pages: /docs on main), the result video, the paper PDF
src/       experiment and library code (flat, so the imports run as-is)
slurm/     cluster batch scripts for the long runs
figures/   main figure (draw.io / pptx sources) and paper figures
notes/     two technical notes (PDF): the mathematics, and the lab record
```

### Map of `src/`

| file | role |
|---|---|
| `e6_sampling.py` | interface classification, samplers (contour / contour-in / area / center), roadmap construction, traversability rule |
| `e6_step2_anytime.py` | anytime interface planner: informed pruning hooks, monotone incumbents, physical-audit callback |
| `e6_build_iris2d.py`, `e6_fig*.py`, `e6_step*.py` | 2-D IRIS testbed, verification figures, 3-D port |
| `e7_highd_audit.py` | 7-DOF / 14-DOF port: kNN sparsification with connectivity chain, lazy rejection, repair-first, corridor polish |
| `e8_validate.py` | **continuous certificate**: clearance + Lipschitz ball chain, path validation, in-region repair |
| `e8_repair_lib.py`, `e8_erosion.py` | library-repair studies: pocket excision surgery, margin-erosion sweep |
| `e9_verify_gcsstar.py` | audit protocol: bit-exact cost reproduction → chain-SOCP argmin → dual-mechanism validation → colliding-pair classification |
| `e9b_depth_profiles.py` | full min-clearance profiles of every colliding reference answer |
| `e10_ours_on_tasks.py`, `e10_regen219.py` | our layer on the violating task pairs; strict regeneration with the Bernoulli gate enforced at generation |
| `e13*_fig2d_*.py` | 2-D story figures: contamination map, grown (not injected) pockets, floor-plan variant |
| `e14_fig_why_pockets.py` | four-panel derivation figure: why pockets must exist |
| `e15_ciris_2dof.py`, `e15b_ciris_dense.py` | C-IRIS (SOS-certified) library composed with this layer |
| `e16_headtohead.py` | the closing experiment: reference vs ours on all task queries |
| `e17_uav_compare.py`, `e9_*_video.py`, `e1[012]_*.py` | renderers for the comparison videos |
| `e18_master_slide.py`, `e19_drawio_main.py` | editable main-figure generators (PPTX, draw.io) |
| `baseline_gcsstar.py` | faithful reimplementation of the search-based baseline used as the reference arm |
| `stageB_bimanual_*.py`, `panda_scene.py`, `ipgcs_core.py` | scenes, region libraries and the query-aware GCS core the layer sits on |

## Requirements

Python 3.12, [Drake](https://drake.mit.edu) 1.54 (`IrisZo`, `CspaceFreePolytope`,
`SceneGraphCollisionChecker`), NumPy/SciPy, CVXPY with Clarabel (MOSEK optional,
required only for the SOS and full-graph GCS arms), MuJoCo + imageio for the
renderers, and [`gcs-science-robotics`](https://github.com/mpetersen94/gcs) on
`PYTHONPATH` for the bimanual scene assets.

Long runs (region regeneration, audits, head-to-head) are Slurm jobs; see
`slurm/`. The 2-D testbed, the figures and the videos run on a laptop.

## Notes

* `notes/soundness_note.pdf` — the mathematics: why pockets must exist (a
  five-step chain from the universally quantified soundness claim to the
  Bernoulli acceptance test), the certificate lemmas, contamination immunity
  of ε-optimality certificates, and the measurement.
* `notes/certainty_experiments.pdf` — the lab record: recipes, raw tables,
  cluster job structure, the amplification law and the connectivity law,
  every number with its generating artifact.
