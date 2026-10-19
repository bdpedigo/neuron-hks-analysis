# %%
"""Plan step 0c: the paper's H01 picture next to the deployment's, on fixed H01
cells.

Two panels per cell, in `figures/h01_paper_compare/`:

- `{root_id}.png`, predictions. Columns: the paper model on the paper run's
  features, with `distance_to_nucleus` as the Fig 6 script scored them and with
  a null distance as the cloud run scored them (its saved predictions match
  these), then deployment models with no H01 training rows.
- `{root_id}_structure.png`, a check of the drawing: each domain in a random
  color, and one HKS column, for the paper run and the deployment. Clean domains
  and smooth features next to patchy predictions mean the patches come from the
  model, not from the drawing.

Rows: the whole cell, and the mesh within `SOMA_RADIUS` of the nucleus.
"""

import numpy as np
import polars as pl

from analysis import Tables
from analysis.io import FIG_PATH
from analysis.models import load_ssa_model, predict
from analysis.products.paper_h01 import load_paper_root, paper_vertex_posteriors
from analysis.products.root_cache import cached
from analysis.products.roots import NO_DOMAIN, domain_frame, load_root_cell
from analysis.products.sources import H01_DATASTACK
from analysis.viz import Channel, View, add_prediction, cell_to_scene, render_panel

# The 5 retrain cells with the most soma labels, and the cell with the largest
# share of no-domain vertices near the soma (see compare_paper_pipeline.py).
ROOTS = [
    864691132319203596,
    864691132354795050,
    864691132320274558,
    864691132353618514,
    864691132369739175,
    864691132455841074,
]
# The cell of the published Fig 6 (validation cohort). Its paper picture is in the
# paper, so it is the check that this script draws the paper run as it was.
FIG6_ROOT = 864691132388416057
MODELS = {
    "hks_nuc": "ssa_model_hks_nuc_v1",
    "hks": "ssa_model_hks_v1",
    "full": "ssa_model_full_v1",
}
STRUCTURE_FEATURE = "hks_20"
SOMA_RADIUS = 30_000  # nm
OUT_PATH = FIG_PATH / "h01_paper_compare"
OUT_PATH.mkdir(parents=True, exist_ok=True)

models = {name: load_ssa_model(folder) for name, folder in MODELS.items()}
nuclei = Tables["h01_root_nuclei"].collect()
load = cached(load_root_cell, max_roots=len(ROOTS) + 2)


def random_colors(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """A random color for each label (rows of `labels`), NaN for a negative label."""
    colors = np.random.default_rng(seed).uniform(0.15, 0.95, size=(labels.max() + 1, 3))
    out = np.full((len(labels), 3), np.nan)
    out[labels >= 0] = colors[labels[labels >= 0]]
    return out


def panels(root_id: int, allow_held_out: bool = False) -> None:
    cell = load(root_id, H01_DATASTACK, allow_held_out=allow_held_out)
    features = domain_frame(cell)
    posteriors = {
        name: predict(model, features, keys=("domain_id",)) for name, model in models.items()
    }
    scene = cell_to_scene(cell, posteriors, [STRUCTURE_FEATURE])
    paper = load_paper_root(root_id)
    add_prediction(scene, "paper", "vertices", paper_vertex_posteriors(paper))
    add_prediction(scene, "paper_no_dist", "vertices", paper_vertex_posteriors(paper, distance=False))

    scene.add(Channel("paper/domain", "vertices", "rgb", random_colors(paper.labels)))
    scene.add(
        Channel(
            "paper/feature",
            "vertices",
            "scalar",
            paper.features[STRUCTURE_FEATURE].reindex(paper.labels).to_numpy(np.float32),
        )
    )
    domain_ids = cell.annotations["domains"].vertex_index
    n_real = int((domain_ids != NO_DOMAIN).sum())
    scene.add(Channel("new/domain", "domains", "rgb", random_colors(np.arange(n_real), seed=1)))

    nucleus = (
        nuclei.filter(pl.col("root_id") == root_id)
        .select("nucleus_x", "nucleus_y", "nucleus_z")
        .to_numpy()[0]
    )
    views = [View("whole cell"), View("soma", center=np.asarray(nucleus), radius=SOMA_RADIUS)]
    render_panel(
        scene,
        ["paper/pred_label", "paper_no_dist/pred_label"] + [f"{name}/pred_label" for name in MODELS],
        views,
        str(OUT_PATH / f"{root_id}.png"),
        titles=["paper", "paper, no distance"] + [f"new {name}" for name in MODELS],
    )
    render_panel(
        scene,
        ["paper/domain", "new/domain", "paper/feature", f"feature/{STRUCTURE_FEATURE}"],
        views,
        str(OUT_PATH / f"{root_id}_structure.png"),
        titles=["paper domains", "new domains", f"paper {STRUCTURE_FEATURE}", f"new {STRUCTURE_FEATURE}"],
    )
    print(f"wrote {root_id}")


# %%
panels(FIG6_ROOT, allow_held_out=True)

# %%
for root_id in ROOTS:
    panels(root_id)
