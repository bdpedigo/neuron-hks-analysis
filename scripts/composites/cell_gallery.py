# %%
"""Figure 3: the model predictions on example cells, and the validation accuracy.

A is one cell of each column type, coloured by the predicted postsynaptic structure,
with a zoom inset on a region of putative spines. B is the validation accuracy per cell
type. A comes from scripts/explanatory/cell_gallery.py and B from
scripts/validation/validation.py.

Ported from the PanelMosaic block of meshrep/clean_paper_code/show_posterior_on_mesh.py,
which both rendered the cells and assembled them. The render is slow and the assembly
is fast, so they are separate here, as with the other composites.

It reads only figure files, so run the two panel scripts first.

    uv run python scripts/composites/cell_gallery.py
"""

import os

# NOTE: cairosvg needs the Homebrew cairo library, as in the meshrep composites.
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:" + os.environ.get(
    "DYLD_LIBRARY_PATH", ""
)

from panel_mosaic import PanelMosaic  # noqa: E402

from analysis import FIG_PATH, cell_type_mapper  # noqa: E402
from analysis.colors import CELL_TYPE_PALETTE  # noqa: E402

gallery = FIG_PATH / "cell_gallery"

# The published order: excitatory by layer on the top row, inhibitory on the bottom.
EXCITATORY = ["23P", "4P", "5P-IT", "5P-ET", "5P-NP", "6P-CT", "6P-IT"]
INHIBITORY = ["BC", "BPC", "MC", "NGC"]

mosaic = """
abcdefg
hijklll
"""
keys = "abcdefghijk"
panel_mapping = {
    key: gallery / f"composite_{cell_type}.png"
    for key, cell_type in zip(keys, EXCITATORY + INHIBITORY)
}
panel_mapping["l"] = FIG_PATH / "validation" / "validation_accuracy_by_cell_type.svg"

# NOTE: cell_type_mapper gives the published names (23P -> 2/3P-IT, 4P -> 4P-IT).
label_mapping = {
    key: cell_type_mapper(cell_type)
    for key, cell_type in zip(keys, EXCITATORY + INHIBITORY)
}
label_mapping["l"] = ""

LABEL_FONTSIZE = 24

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping=label_mapping,
    figsize=(7 * 3, 12),
    label_dodge=False,
    label_fontsize=LABEL_FONTSIZE,
    layout="tight",
)

axs = pm.axs
axs["a"].set_ylabel(
    "Excitatory", fontsize=LABEL_FONTSIZE, color=CELL_TYPE_PALETTE["excitatory"]
)
axs["h"].set_ylabel(
    "Inhibitory", fontsize=LABEL_FONTSIZE, color=CELL_TYPE_PALETTE["inhibitory"]
)
axs["a"].text(
    -0.25, 0.93, "A)", transform=axs["a"].transAxes, fontsize=LABEL_FONTSIZE + 10
)
axs["l"].text(
    0.02, 0.93, "B)", transform=axs["l"].transAxes, fontsize=LABEL_FONTSIZE + 10
)

# %%
out = FIG_PATH / "composites"
out.mkdir(parents=True, exist_ok=True)
pm.write(out / "cell-gallery", dpi=300, verbose=True)
