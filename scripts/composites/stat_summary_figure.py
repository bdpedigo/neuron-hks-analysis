# %%
"""Figure 4: the target structures of the inputs to and outputs from the curated cells.

A is the inputs onto the curated cells, B the same proportion against every neuron in
the dataset, C the outputs onto excitatory targets and D the outputs onto inhibitory
targets. The plot panels come from scripts/census/input_compartments.py (4A),
scripts/census/whole_dataset.py (4B) and scripts/census/outputs.py (4C, 4D). The three
cell diagrams are hard copies in figures/legacy/diagram_cells (see its README).

Ported from meshrep/clean_paper_code/stat_summary_figure.py. The mosaic, the labels and
the panel keys are unchanged; only the folders moved.

It reads only figure files, so run the three panel scripts first.

    uv run python scripts/composites/stat_summary_figure.py
"""

import os

# NOTE: cairosvg needs the Homebrew cairo library, as in the meshrep composites.
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:" + os.environ.get(
    "DYLD_LIBRARY_PATH", ""
)

from panel_mosaic import PanelMosaic  # noqa: E402

from analysis import FIG_PATH  # noqa: E402

census = FIG_PATH / "census"
diagrams = FIG_PATH / "legacy" / "diagram_cells"

mosaic = """
AABBCCCDDD
FFGGGHHIII
"""
# NOTE: D and the diagrams stay .png. The .svg of D is 8 MB, which skunk composites
# slowly and cairosvg then rasterizes anyway.
panel_mapping = {
    "A": diagrams / "composite_synapses_onto_exc.png",
    "B": census / "spine_histograms.svg",
    "C": census / "cell_type_input_spine_proportion.svg",
    "D": census / "whole_dataset_soma_depth_vs_post_p_spine_synapse.png",
    "F": diagrams / "composite_synapses_from_exc_to_exc.png",
    "G": census / "cell_type_output_proportion_to_excitatory_scatter.svg",
    "H": diagrams / "composite_synapses_from_exc_to_inh.png",
    "I": census / "cell_type_output_proportion_to_inhibitory_scatter.svg",
}
label_mapping = {
    "A": "A) Synaptic inputs to proofread neurons",
    "B": "",
    "C": "",
    "D": "B) Synaptic inputs to all neurons",
    "E": "",
    "F": "C) Proofread synaptic outputs to excitatory neurons",
    "G": "",
    "H": "D) Proofread synaptic outputs to inhibitory neurons",
}

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping=label_mapping,
    figsize=(28, 16),
    layout="constrained",
    label_fontsize=25,
    label_pos=(0, 0.5),
    panel_borders=False,
    label_dodge="top",
    label_dodge_factor=0.03,
)

# %%
out = FIG_PATH / "composites"
out.mkdir(parents=True, exist_ok=True)
pm.write(out / "stat_summary_figure", dpi=300, verbose=True)
