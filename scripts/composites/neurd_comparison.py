# %%
"""The NEURD comparison figure (reviewer R2.1, ssa-backlog TASK-38).

A: macro F1 per cell type on the validation synapses
(scripts/validation/compare_to_neurd.py). B to D: the column cells
(scripts/validation/neurd_downstream.py). B is the share of the cells of each type
with a labeled input. C is the per-cell input spine proportion, HKS on the
NEURD-labeled synapses against NEURD. D is the Spearman rho of that proportion
with cell covariates, within each cell type.

It reads only figure files, so run the two panel scripts first.

    uv run python scripts/composites/neurd_comparison.py
"""

import os

# NOTE: cairosvg needs the Homebrew cairo library, as in the meshrep composites.
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:" + os.environ.get(
    "DYLD_LIBRARY_PATH", ""
)

from panel_mosaic import PanelMosaic  # noqa: E402

from analysis import FIG_PATH  # noqa: E402

mosaic = """
AAABBB
CCDDDD
"""
panel_mapping = {
    "A": FIG_PATH / "compare_to_neurd" / "compare_neurd_hks_by_cell_type.svg",
    "B": FIG_PATH / "neurd_downstream" / "composite_cell_coverage.svg",
    "C": FIG_PATH / "neurd_downstream" / "composite_spine_proportion_scatter.svg",
    "D": FIG_PATH / "neurd_downstream" / "composite_trend_rho_by_cell_type.svg",
}
label_mapping = {
    "A": "A) Accuracy on the validation synapses",
    "B": "B) Column cells with labeled inputs",
    "C": "C) Input spine proportion per cell",
    "D": "D) Trends of the input spine proportion within each cell type",
}

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping=label_mapping,
    figsize=(20, 14),
    layout="constrained",
    label_fontsize=20,
    label_pos=(0, 1.0),
    panel_borders=False,
    label_dodge="top",
    label_dodge_factor=0.04,
)

# %%
out = FIG_PATH / "composites"
out.mkdir(parents=True, exist_ok=True)
pm.write(out / "neurd_comparison", dpi=300)
