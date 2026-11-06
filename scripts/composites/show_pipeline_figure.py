# %%
"""Figure 2: the pipeline, and the comparison of the proposed method to its ablations.

A is the pipeline mosaic from scripts/explanatory/explain_pipeline.py. B is blank for
now. C is the confusion matrix of the proposed arm from scripts/ablation/compare_ablations.py.

Ported from meshrep/clean_paper_code/show_pipeline_figure.py. The figure size and the
grid are unchanged. B takes the width of the old B-D and C takes the old E.

It reads only figure files, so run the two panel scripts first.

    uv run python scripts/composites/show_pipeline_figure.py
"""

import os

# NOTE: cairosvg needs the Homebrew cairo library, as in the meshrep composites.
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:" + os.environ.get(
    "DYLD_LIBRARY_PATH", ""
)

from panel_mosaic import PanelMosaic  # noqa: E402

from analysis import FIG_PATH, capped_dpi  # noqa: E402

mosaic = """
AAAAAAAAA
BBBBBBCCC
"""
# TODO: B is left out of the mapping, so it draws as a blank panel.
panel_mapping = {
    "A": FIG_PATH / "explain_pipeline" / "explain_pipeline_mosaic_simple.svg",
    "C": FIG_PATH / "ablation" / "confusion_proposed.svg",
}

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping={"A": "A", "B": "B", "C": "C"},
    gridspec_kw={"height_ratios": [12, 8.5], "wspace": 0, "hspace": 0.0},
    figsize=(28, 22),
    label_fontsize=50,
    layout="constrained",
    panel_borders=False,
)
pm.show()

# %%
out = FIG_PATH / "composites"
out.mkdir(parents=True, exist_ok=True)
pm.write(out / "show_pipeline_figure", dpi=capped_dpi(pm.fig.get_figwidth()))
