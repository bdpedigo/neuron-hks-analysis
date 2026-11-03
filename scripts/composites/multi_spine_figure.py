# %%
"""Figure 5: multi-input spine targeting for the proofread cells.

A is the inputs onto multi-input spines, B their distribution across excitatory
neurons, C their size and location, D the outputs onto them, E the spine targets of
excitatory axons, F the spine targets of inhibitory axons, G the pairing of the two
inputs and H the synapse size against spine volume. The plot panels come from
scripts/census/input_multi_spines.py (5A, 5C, 5E left),
scripts/census/whole_dataset.py (5B), scripts/census/outputs.py (5D),
scripts/census/inhibitory_targeting.py (5E right, 5F) and
scripts/census/multi_synapse_spines.py (5G, 5H). The three cell diagrams are hard
copies in figures/legacy/diagram_cells (see its README).

Ported from the second PanelMosaic block of meshrep/clean_paper_code/multi_spine_figure.py,
which is the one that built the published figure. The first block there builds an older
six-panel layout. The mosaic, the labels and the panel keys are unchanged; only the
folders moved.

It reads only figure files, so run the five panel scripts first.

    uv run python scripts/composites/multi_spine_figure.py
"""

import os

# NOTE: cairosvg needs the Homebrew cairo library, as in the meshrep composites.
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:" + os.environ.get(
    "DYLD_LIBRARY_PATH", ""
)

from panel_mosaic import PanelMosaic  # noqa: E402

from analysis import FIG_PATH, capped_dpi  # noqa: E402

census = FIG_PATH / "census"
diagrams = FIG_PATH / "legacy" / "diagram_cells"

upper_left = [
    ["A", "A", "B", "B", "B"],
]
upper_right = [
    ["E", "C", "D"],
]
middle_left = [["F", "F", "G", "G", "G"]]
middle_right = [
    ["H", "H", "hh", "hh", "I", "I"],
]
bottom_left = [["J", "K"]]
bottom_right = [
    [
        "L",
    ]
]

mosaic = [
    [upper_left, upper_right],
    [middle_left, middle_right],
    [bottom_left, bottom_right],
]

# NOTE: E and the diagrams stay .png. The .svg of E is 6 MB, which skunk composites
# slowly and cairosvg then rasterizes anyway.
# NOTE: L is multi_synapse_size_vs_spine_volume, not spine_synapse_area_corr_by_multi.
# The latter is supplement S8; see the markers in scripts/census/multi_synapse_spines.py.
panel_mapping = {
    "A": diagrams / "composite_multi_spine_synapses_onto_exc.png",
    "B": census / "cell_type_input_proportion_multi_spine.svg",
    "C": census / "spine_size_vs_p_multi_rotated.svg",
    "D": census / "multi_spine_prop_by_euc_distance_to_nuc.svg",
    "E": census / "excitatory_post_p_spine_site_is_multi_distribution_with_p_spine.png",
    "F": diagrams / "composite_synapses_from_exc_to_multi.png",
    "G": census / "cell_type_output_proportion_multi_spine_to_excitatory.svg",
    "H": census / "exc_vs_th_size_by_multi_rotated.svg",
    "hh": census / "multi_spine_prop_by_euc_distance_to_nuc_by_exc_broad_type.svg",
    "I": census / "bc_vs_mc_multi_spine_prop_by_euc_distance_to_nuc.svg",
    "J": census / "multi_spine_pairing.svg",
    "K": diagrams / "composite_1E1I_synapse.png",
    "L": census / "multi_synapse_size_vs_spine_volume.svg",
}
label_mapping = {
    "A": "A) Synaptic inputs onto multi-input spines",
    "B": "",
    "C": "C) Multi-input spines on excitatory neurons",
    "E": "B) Distribution across\n     excitatory neurons",
    "F": "D) Synaptic outputs onto excitatory multi-input spines",
    "H": "E) Spine targets\n    of excitatory neurons",
    "I": "F) Spine targets\n    of inhibitory neurons",
    "J": "G) Double-input spine synapse pairing",
    "L": "H) Synapse size and spine volume correlations for double-input spines",
}

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping=label_mapping,
    figsize=(28, 24),
    gridspec_kw={"wspace": 0.06, "hspace": 0.0},
    layout="constrained",
    label_fontsize=25,
    label_pos=(0, 0.7),
    panel_borders=False,
    label_dodge="top",
    label_dodge_factor=0.04,
)

# %%
out = FIG_PATH / "composites"
out.mkdir(parents=True, exist_ok=True)
pm.write(
    out / "multi_spine_figure", dpi=capped_dpi(pm.fig.get_figwidth()), verbose=True
)
