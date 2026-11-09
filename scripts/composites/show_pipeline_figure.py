# %%
"""Figure 2: the pipeline, and the comparison of the proposed method to its ablations.

A is the pipeline mosaic from scripts/explanatory/explain_pipeline.py. B is the time
against the storage of each neuron in each ablation arm, C is a temporary narrow panel of
the accuracy of each arm, and D is the confusion matrix of the proposed arm, all from
scripts/ablation/compare_ablations.py.

Ported from meshrep/clean_paper_code/show_pipeline_figure.py. The figure size and the
grid are unchanged.

It reads only figure files, so run the two panel scripts first.

    uv run python scripts/composites/show_pipeline_figure.py
"""

import os
import xml.etree.ElementTree as ET

# NOTE: cairosvg needs the Homebrew cairo library, as in the meshrep composites.
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:" + os.environ.get(
    "DYLD_LIBRARY_PATH", ""
)

from panel_mosaic import PanelMosaic  # noqa: E402

from analysis import FIG_PATH, capped_dpi  # noqa: E402

mosaic = """
AAA
BCD
"""
# TODO: C is a temporary version.
panel_mapping = {
    "A": FIG_PATH / "explain_pipeline" / "explain_pipeline_mosaic_simple.svg",
    "B": FIG_PATH / "ablation" / "time_storage_scatter_annotated.svg",
    "C": FIG_PATH / "ablation" / "f1_weighted_comparison_horizontal.svg",
    "D": FIG_PATH / "ablation" / "confusion_proposed.svg",
}



def svg_width(path) -> float:
    """The width of a matplotlib SVG, in points."""
    return float(ET.parse(path).getroot().get("width").removesuffix("pt"))


# NOTE: B-D are saved at about one height and one font size. Each column is as wide as its
# panel, so the three draw at one scale and their text at one size.
width_ratios = [svg_width(panel_mapping[label]) for label in "BCD"]

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping={"A": "A", "B": "B", "C": "C", "D": "D"},
    gridspec_kw={
        "height_ratios": [12, 8.5], "width_ratios": width_ratios, "wspace": 0, "hspace": 0.0
    },
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
