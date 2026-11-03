# %%
"""The NEURD comparison figure (reviewer R2.1, ssa-backlog TASK-38).

A: macro F1 per cell type on the validation synapses
(scripts/validation/compare_to_neurd.py). B to E come from
scripts/validation/neurd_downstream.py. B is the share of the column cells of each
type with a labeled input. C is the share of the output synapses of the proofread
cells of each type with a label. D is the per-cell input spine proportion, HKS on
the NEURD-labeled synapses against NEURD. E is D for the outputs of the proofread
cells to excitatory targets.

It reads only figure files, so run the two panel scripts first.

    uv run python scripts/composites/neurd_comparison.py
"""

import os

# NOTE: cairosvg needs the Homebrew cairo library, as in the meshrep composites.
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:" + os.environ.get(
    "DYLD_LIBRARY_PATH", ""
)

from panel_mosaic import PanelMosaic  # noqa: E402

from analysis import FIG_PATH, capped_dpi  # noqa: E402

# NOTE: one column is 1 inch of `figsize`. The panels are about 3.8 by 6.4 inches
# (A to C) and 6 inches square (D and E), so each fills its cell with little
# scaling.
mosaic = """
AAAABBBBCCCC
DDDDDDEEEEEE
"""
panel_mapping = {
    "A": FIG_PATH / "compare_to_neurd" / "compare_neurd_hks_by_cell_type_dots.svg",
    "B": FIG_PATH / "neurd_downstream" / "composite_cell_coverage.svg",
    "C": FIG_PATH / "neurd_downstream" / "composite_output_coverage.svg",
    "D": FIG_PATH / "neurd_downstream" / "composite_spine_proportion_scatter.svg",
    "E": FIG_PATH / "neurd_downstream" / "composite_output_spine_proportion_scatter.svg",
}
label_mapping = {
    "A": "A) Accuracy on\nvalidation synapses",
    "B": "B) Label coverage of\ninputs to column cells",
    "C": "C) Label coverage of\noutputs of proofread cells",
    "D": "D) Input spine proportion per cell",
    "E": "E) Output spine proportion per cell",
}

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping=label_mapping,
    figsize=(12, 13),
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
pm.write(out / "neurd_comparison", dpi=capped_dpi(pm.fig.get_figwidth()))
