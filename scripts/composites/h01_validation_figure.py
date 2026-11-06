# %%
"""Figure 6: the H01 validation scores, and the predictions on the mesh of an H01 cell.

A-C are the MICrONS comparison, the confusion matrix and the calibration curve, from
scripts/h01/h01_validation.py. D is the mesh panel from scripts/h01/h01_mesh_panel.py.

It reads only figure files, so run the two panel scripts first.

    uv run python scripts/composites/h01_validation_figure.py
"""

import os

# NOTE: cairosvg needs the Homebrew cairo library, as in the meshrep composites.
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:" + os.environ.get(
    "DYLD_LIBRARY_PATH", ""
)

from panel_mosaic import PanelMosaic  # noqa: E402

from analysis import FIG_PATH, capped_dpi  # noqa: E402

H01_PATH = FIG_PATH / "h01_validation"
ROOT_ID = 864691132388416057
name = f"h01_{ROOT_ID}_validation"

pm = PanelMosaic(
    """
    AD
    BD
    CD
    """,
    panel_mapping={
        "A": H01_PATH / "h01_microns_score_comparison_short.png",
        "B": H01_PATH / "h01_confusion_matrix.svg",
        "C": H01_PATH / "h01_calibration_curve_binary.png",
        "D": H01_PATH / f"composite_{name}.png",
    },
    label_mapping={"A": "A)", "B": "B)", "C": "C)", "D": "D)"},
    gridspec_kw={
        "height_ratios": [5, 4, 4],
        "width_ratios": [1, 2],
        "wspace": 0,
        "hspace": 0.0,
    },
    figsize=(17, 15),
    label_fontsize=50,
    layout="constrained",
    panel_borders=False,
)

# %%
out = FIG_PATH / "composites"
out.mkdir(parents=True, exist_ok=True)
# NOTE: the name is what main.tex includes, so `just publish-figures` copies it as is.
pm.write(out / "h01_validation_composite", dpi=capped_dpi(pm.fig.get_figwidth()))
