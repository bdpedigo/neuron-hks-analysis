"""The model comparison viewer: every variant in `MODEL_SPECS`, trained on MICrONS
only or on MICrONS + the H01 retrain rows, on whole cells of any datastack in
`products.roots.ROOT_SOURCES`.

One mesh at a time. The checkboxes on the left choose the training rows, the
variant, and what colors the mesh (the predicted label, the posterior blend,
p(spine), or a feature from the slider). Every synapse shows the predicted label
of its domain; a checkbox shows or hides them. On H01, the `paper` training
option shows the paper's model on the paper run's features (`products.paper_h01`)
and ignores the variant. Keys: `n` for the next cell, `b`
for the one before. Right-click flies the camera to the clicked point.

The cells come in sorted order from a random start, so one session can visit all
of them. A worker process loads and scores the next cell while one shows (about a
minute from the cloud). The last few cells stay in `data/root_cache/`.

The scripts that run it name the datastack and the cell pools:
`scripts/h01/view_retrain_models.py` and `scripts/model/view_microns_models.py`.
"""

import argparse
from collections.abc import Callable, Sequence
from functools import partial

from .models import MODEL_SPECS, load_ssa_model, predict
from .products.paper_h01 import load_paper_root, paper_vertex_posteriors
from .products.root_cache import cached
from .products.roots import domain_frame, feature_columns, load_root_cell
from .sources import H01_DATASTACK
from .viz import CellOrder, Slider, Viewer, add_prediction, cell_to_scene

# The retrained models come from the last cell of `scripts/h01/retrain_h01.py`.
TRAINING = {"no retrain": "", "retrain": "_h01"}
VARIANTS = {name.removeprefix("ssa_model_"): name for name in MODEL_SPECS}
MODELS = {
    f"{training}/{variant}": f"{name}{suffix}_v1"
    for training, suffix in TRAINING.items()
    for variant, name in VARIANTS.items()
}
COLORS = ["pred_label", "blend", "p(spine)", "feature"]
# The `training` option that shows the paper's model on the paper run's features
# (`products.paper_h01`), for the datastacks that have one. It ignores `variant`.
PAPER = "paper"
PAPER_DATASTACKS = (H01_DATASTACK,)
MAX_CACHED_ROOTS = 5

# Set in the worker process by `start_worker`.
_worker = {}


def start_worker(datastack: str, use_cache: bool = True) -> None:
    """Runs once in the worker: load every model, and set up the cell loader."""
    _worker["datastack"] = datastack
    _worker["models"] = {key: load_ssa_model(name) for key, name in MODELS.items()}
    _worker["load"] = (
        cached(load_root_cell, max_roots=MAX_CACHED_ROOTS) if use_cache else load_root_cell
    )


def scene_for_root(root_id: int):
    """Runs in the worker: the cell of `root_id`, scored by every model."""
    datastack = _worker["datastack"]
    cell = _worker["load"](root_id, datastack)
    features = domain_frame(cell)
    posteriors = {
        key: predict(model, features, keys=("domain_id",))
        for key, model in _worker["models"].items()
    }
    scene = cell_to_scene(cell, posteriors, feature_columns(datastack))
    if datastack in PAPER_DATASTACKS:
        add_prediction(scene, PAPER, "vertices", paper_vertex_posteriors(load_paper_root(root_id)))
    return scene


def resolve(state: dict):
    model = PAPER if state["training"] == PAPER else f"{state['training']}/{state['variant']}"
    if state["color"] == "feature":
        channel = state["feature"]
    else:
        channel = f"{model}/{state['color']}"
    return channel, f"{model}/pred_label"


def make_viewer(
    datastack: str,
    roots: Sequence[int],
    *,
    start=None,
    seed=None,
    use_cache: bool = True,
    prefetch: bool = True,
    off_screen: bool = False,
) -> Viewer:
    return Viewer(
        scene_for_root,
        CellOrder(roots, start=start, seed=seed),
        selectors={
            "training": list(TRAINING) + ([PAPER] if datastack in PAPER_DATASTACKS else []),
            "variant": list(VARIANTS),
            "color": COLORS,
        },
        resolve=resolve,
        sliders={
            "feature": Slider(
                [f"feature/{c}" for c in feature_columns(datastack)],
                visible=lambda state: state["color"] == "feature",
            )
        },
        prefetch=prefetch,
        worker_initializer=partial(start_worker, datastack, use_cache),
        off_screen=off_screen,
    )


def main(datastack: str, pools: dict[str, Callable[[], Sequence[int]]], description: str) -> None:
    """Parse the command line and run the viewer on one of `pools` (name to a
    function that gives the root ids; the first is the default)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pool", choices=list(pools), default=next(iter(pools)))
    parser.add_argument("--start", type=int, help="the place in the order to start at")
    parser.add_argument("--seed", type=int, help="the seed of the random start")
    parser.add_argument("--screenshot", help="write an image of the first cell and exit")
    parser.add_argument("--no-cache", action="store_true", help="do not read or write data/root_cache")
    args = parser.parse_args()

    viewer = make_viewer(
        datastack,
        pools[args.pool](),
        start=args.start,
        seed=args.seed,
        use_cache=not args.no_cache,
        prefetch=args.screenshot is None,
        off_screen=args.screenshot is not None,
    )
    viewer.show(screenshot=args.screenshot)
