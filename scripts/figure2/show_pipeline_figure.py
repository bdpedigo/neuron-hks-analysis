# %%
import os

os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:$DYLD_LIBRARY_PATH"

from pathlib import Path

from panel_mosaic import PanelMosaic

figure_path = Path("/Users/ben.pedigo/code/meshrep/meshrep/figures")

panel_mapping = {
    "A": figure_path / "explain_pipeline" / "explain_pipeline_mosaic_simple.svg",
    "B": figure_path / "profile_approaches2" / "time_comparison.svg",
    "C": figure_path / "profile_approaches2" / "storage_comparison.svg",
    "D": figure_path / "profile_approaches2" / "test_accuracy_comparison_zoom.svg",
    "E": figure_path / "profile_approaches2" / "confusion_matrix.svg",

    #     "B": figure_path / "explain_pipeline" / "single_heat_trace.svg",
    #     "C": figure_path
    #     / "diagram_heat_on_labeled_cell"
    #     / "multipanel_solve_on_neuron.svg",
    #     "D": figure_path
    #     / "diagram_heat_on_labeled_cell"
    #     / "many_label_examples_on_neuron_zoom.svg",
    #     "E": figure_path / "diagram_heat_on_labeled_cell" / "hks_curves_by_label.svg",
    # }
}


mosaic = """
AAAAAAAAA
BBCCDDEEE
"""

panel = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping={"A": "A", "B": "B", "C": "C", "D": "D", "E": "E"},
    gridspec_kw={"height_ratios": [12, 8.5], 'wspace':0, 'hspace':0.0},
    figsize=(28, 22),
    label_fontsize=50,
    layout="constrained",
    panel_borders=False,
)
panel.show()
panel.write(figure_path / "show_pipeline_figure" / "show_pipeline_figure")

# %%
