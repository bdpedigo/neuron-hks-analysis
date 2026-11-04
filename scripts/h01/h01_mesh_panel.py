# %%
"""Fig 6D: the deployment predictions on the mesh of an
H01 cell of the validation cohort.

Ported from meshrep/sandbox/h01_validation.py. Each vertex takes the `pred_label`
of its domain in the deployment `h01` job; a vertex in no domain is `unknown`.
The cameras are those of the published figure. The composite is
scripts/composites/h01_validation_figure.py.
"""

import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

from analysis import (
    FIG_PATH,
    plot_zoom_indicator,
    render,
    save_matplotlib_figure,
    set_matplotlib_theme,
    set_pyvista_theme,
)
from analysis.colors import COMPARTMENT_PALETTE_MUTED
from analysis.products.roots import NO_DOMAIN, deployment_posteriors, load_root_cell
from analysis.sources import H01_DATASTACK

SUBFOLDER = "h01_validation"
FIGURE_PATH = FIG_PATH / SUBFOLDER
ROOT_ID = 864691132388416057
INTERACTIVE = False

set_matplotlib_theme()
set_pyvista_theme()

# %%
# NOTE: the root is in the held-out validation cohort; this is its one look.
cell = load_root_cell(ROOT_ID, H01_DATASTACK, allow_held_out=True)
vertex_domain = np.asarray(cell.mesh.map_index_to_layer("domains"))
domain_label = deployment_posteriors(cell).to_pandas().set_index("domain_id")["pred_label"]
pred_labels = (
    pd.Series(vertex_domain)
    .map(domain_label)
    .where(vertex_domain != NO_DOMAIN)
    .fillna("unknown")
    .to_numpy()
)
print(pd.Series(pred_labels).value_counts(normalize=True))
colors = np.array([COMPARTMENT_PALETTE_MUTED[label] for label in pred_labels], dtype=np.uint8)
mesh = pv.make_tri_mesh(cell.mesh.vertices, cell.mesh.faces)


def add_cell(plotter):
    plotter.add_mesh(mesh, scalars=colors, rgb=True, show_scalar_bar=False)


# %%
cpos = [
    (2878547.0786078013, 932324.7857342574, 747462.9265241964),
    (2988822.597355306, 880368.5846934861, 104071.52193589504),
    (0.8138798508621828, -0.5511285704496345, 0.1840024108393515),
]
zoom_cposes = [
    [(3083460.252571646, 902456.7085839356, 141675.18488479272),
     (3079915.332538704, 900494.8473509091, 120925.84017969745),
     (0.8370716745993685, -0.5392991948147503, -0.0920184223705852)],
    [(2993682.8619351415, 915194.2740472725, 116025.7183485223),
    (3007461.213245588, 917333.9373609251, 108900.1364244689),
    (0.43208245778048426, -0.6284796886255994, 0.646775100528341)],
    [
        (2955513.629277105, 929265.3024383206, 85084.00739455571),
        (2952106.642677461, 934648.834441879, 84674.0923590454),
        (0.05134428147586535, -0.04347647167766007, -0.9977342136912962),
    ],
]

#%%
INTERACTIVE = False

name = f"h01_{ROOT_ID}_validation"
plotter = pv.Plotter(window_size=(600, 800), off_screen=not INTERACTIVE)
add_cell(plotter)
plotter.camera_position = cpos
for zoom_cpos in zoom_cposes:
    plot_zoom_indicator(plotter, zoom_cpos, plotter.camera_position, shift=20_000)
render(plotter, name, FIGURE_PATH, interactive=INTERACTIVE)

for i, zoom_cpos in enumerate(zoom_cposes):
    plotter = pv.Plotter(window_size=(400, 400), off_screen=not INTERACTIVE)
    add_cell(plotter)
    plotter.camera_position = zoom_cpos
    plotter.camera.clipping_range = (10, 70_000)
    render(plotter, f"{name}_zoom_{i}", FIGURE_PATH, interactive=INTERACTIVE, circle_crop=True)

# %%
# 6D: the whole cell with the three numbered insets.
inset_bounds = [
    (0.33, 0.72, 0.4, 0.4),
    (0.66, 0.3, 0.4, 0.4),
    (0.5, -0.05, 0.4, 0.4),
]
label_positions = [
    (0.255, 0.71),
    (0.373, 0.489),
    (0.43, 0.305),
]
fig, ax = plt.subplots(1, 1, figsize=(6, 8), dpi=500, gridspec_kw=dict(hspace=0, wspace=0))
ax.set_facecolor("none")
ax.imshow(plt.imread(FIGURE_PATH / f"{name}.png"), interpolation="none", aspect=None)
ax.axis("off")
for i, (bounds, label_position) in enumerate(zip(inset_bounds, label_positions)):
    sub_ax = ax.inset_axes(bounds)
    sub_ax.set_facecolor("none")
    sub_ax.imshow(plt.imread(FIGURE_PATH / f"{name}_zoom_{i}_circular.png"), aspect=None)
    sub_ax.axis("off")
    sub_ax.text(
        0.13, 0.75, f"{i + 1}", color="black", ha="left", va="top", transform=sub_ax.transAxes
    )
    ax.add_artist(
        ConnectionPatch(
            xyA=(0.5, 0.5),
            xyB=label_position,
            coordsA="axes fraction",
            coordsB="axes fraction",
            axesA=sub_ax,
            axesB=ax,
            color="black",
            lw=2,
        )
    )
save_matplotlib_figure(fig, f"composite_{name}", subfolder=SUBFOLDER, formats=["png"])
