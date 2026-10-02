# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from caveclient import CAVEclient
from fast_simplification import simplify

from meshmash import HeatSolver, chunked_hks_pipeline, project_points_to_mesh
from analysis import (
    COMPARTMENT_PALETTE_HEX,
    COMPARTMENT_PALETTE_MUTED,
    COMPARTMENT_PALETTE_MUTED_HEX,
    DATA_PATH,
    FIG_PATH,
    FONT_PATH,
    make_spheres_from_points,
    mask_dendrite_by_client_skeleton,
    save_matplotlib_figure,
    save_pyvista_figure,
    set_matplotlib_theme,
    set_pyvista_theme,
)
from panel_mosaic import PanelMosaic

font_file = FONT_PATH

figure_out_path = FIG_PATH / "diagram_heat_on_labeled_cell"

set_pyvista_theme()

SAVE = False

client = CAVEclient("minnie65_phase3_v1", version=1300)
cv = client.info.segmentation_cloudvolume(progress=False)

query_kwargs = {
    "desired_resolution": [1, 1, 1],
    "split_positions": True,
    "log_warning": False,
}

# %%

if False:
    targets = client.materialize.query_table(
        "vortex_compartment_targets", **query_kwargs
    )
    targets.groupby("post_pt_root_id").size().sort_values(ascending=False).head(10)
    root_targets = targets.query("post_pt_root_id == @root_id").copy()
    root_targets.to_csv(DATA_PATH / "sample_targets.csv")

# %%

root_id = 864691136025099065
raw_mesh = cv.mesh.get(
    root_id, remove_duplicate_vertices=True, deduplicate_chunk_boundaries=False
)[root_id]

# %%
mesh = (raw_mesh.vertices, raw_mesh.faces)
mesh = simplify(*mesh, target_reduction=0.7)

try:
    mesh = mask_dendrite_by_client_skeleton(
        mesh,
        root_id,
        client,
        distance_threshold=5,
    )
    print("Masked dendrite successfully")
except Exception as e:
    print(f"Error masking dendrite for root_id {root_id}: {e}")

# %%


root_targets = pd.read_csv(
    DATA_PATH / "sample_targets.csv",
    index_col=0,
)
root_targets["tag"] = root_targets["tag"].replace({"soma_spine": "spine"})
root_targets = root_targets.query("tag.isin(['soma', 'spine', 'shaft'])")

points = root_targets[
    ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
].values

mesh_indices = project_points_to_mesh(points, mesh, distance_threshold=1000)
root_targets["mesh_index"] = mesh_indices
root_targets = root_targets.query("mesh_index != -1")

points = root_targets[
    ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
].values

# %%

spheres = make_spheres_from_points(points, root_targets["tag"].values)
rgb = np.array([COMPARTMENT_PALETTE_MUTED[c] for c in root_targets["tag"]]) / 255

window_size = np.floor(np.array((6.44, 13.92)) * 100).astype(int)

plotter = pv.Plotter(window_size=window_size)

mesh_params = dict(color="darkgrey")

sphere_params = dict(
    scalars="rgb",
    rgb=True,
    ambient=1,
    diffuse=0,
    specular=0,
    metallic=0,
    opacity=0.5,
)

glowing_point_params = dict(
    point_size=2,
    style="points_gaussian",
    emissive=True,
    scalars=rgb,
    rgb=True,
)
plotter.add_mesh(pv.make_tri_mesh(*mesh), **mesh_params)

plotter.add_mesh(spheres, **sphere_params)

plotter.add_points(points, **glowing_point_params)

wide_cpos = [
    (696386.7915661471, 428009.9887418418, 1409355.1204906167),
    (757098.5343463086, 495593.5471083228, 853471.0010635282),
    (0.007377127303158819, -0.992759470032648, -0.11989250457492864),
]
plotter.camera_position = wide_cpos

save_pyvista_figure(
    plotter,
    "many_label_examples_on_neuron",
    figure_out_path,
    formats=["svg", "png"],
    scale=5,
)
plotter.show(jupyter_backend="static")
plotter.close()

# %%

# %%

root_targets = root_targets.sort_values("target_id")

n_samples = 5

sample_indices = [
    180698808,
    180699673,
    180699067,
    180408094,
    180698460,
    # shaft
    187703650,
    181499944,
    166248414,
    177623769,
    161206469,
    # spine
    172759448,
    177592987,
    172564612,
    172658229,
    171027561,
]
samples = (
    root_targets.set_index("target_id").loc[sample_indices].reset_index(drop=False)
)
samples
# %%
plotter = pv.Plotter(
    shape=(3, n_samples),
    window_size=np.floor(np.array((4.00, 3.85)) * 250).astype(int),
    border_width=8,
    border_color=(0.1, 0.1, 0.1),
)

poly = pv.make_tri_mesh(*mesh)
poly.compute_normals(
    cell_normals=False, point_normals=True, consistent_normals=True, inplace=True
)
normals = poly.point_data["Normals"]

row_map = {"soma": 0, "shaft": 1, "spine": 2}
distance_map = {"soma": 40_000, "shaft": 10_000, "spine": 7_000}

for tag, tag_samples in samples.groupby("tag"):
    for i, (_, sample) in enumerate(tag_samples.iterrows()):
        plotter.subplot(row_map[tag], i)

        plotter.add_mesh(poly, **mesh_params)
        point = (
            sample[["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]]
            .values.reshape(1, -1)
            .astype(float)
        )
        plotter.add_points(
            point,
            render_points_as_spheres=True,
            point_size=100,
            color=COMPARTMENT_PALETTE_MUTED[tag],
            ambient=1,
            diffuse=0,
            specular=0,
            metallic=0,
            opacity=1,
        )

        plotter.enable_depth_peeling()

        plotter.camera.focal_point = point[0]
        plotter.camera.position = (
            point[0] + normals[sample["mesh_index"]] * distance_map[tag]
        )

        plotter.camera.clipping_range = (1, 2 * distance_map[tag])
        plotter.camera.up = [0, 1, 0]

plotter.subplot(0, 0)
plotter.add_text(
    " Soma",
    position="upper_left",
    font_size=100,
    font_file=font_file,
    color=COMPARTMENT_PALETTE_MUTED["soma"],
)
plotter.subplot(1, 0)
plotter.add_text(
    " Shaft",
    position="upper_left",
    font_size=100,
    font_file=font_file,
    color=COMPARTMENT_PALETTE_MUTED["shaft"],
    shadow=True,
)
plotter.subplot(2, 0)
plotter.add_text(
    " Spine",
    position="upper_left",
    font_size=100,
    font_file=font_file,
    color=COMPARTMENT_PALETTE_MUTED["spine"],
)

if SAVE:
    save_pyvista_figure(
        plotter,
        f"vortex_labels_zoom_root_id={root_id}",
        figure_out_path,
        formats=["svg", "png"],
        scale=5,
    )
plotter.show(jupyter_backend="static")


# %%
t_max = 20000000.0
t_min = 50000.0
timescales = [1e5, 1e6, 1e7]
hs = HeatSolver(timescales=timescales)

hs.fit(mesh)

initial_nodes = samples["mesh_index"].values
solutions = hs.solve(initial_nodes)

# %%

tag_nodes = {"soma": 0, "shaft": 5, "spine": 10}

set_pyvista_theme()

window_size = np.floor(np.array((6.57, 6.92)) * 250).astype(int)
plotter = pv.Plotter(
    shape=(len(tag_nodes), len(hs.timescales)),
    border_color=(1.0, 1.0, 1.0),
    border_width=0,
    window_size=window_size,
)

# distance = 15000
distance_map = {"soma": 30_000, "shaft": 10_000, "spine": 10_000}

row_map = {"soma": 0, "shaft": 1, "spine": 2}

for tag, node in tag_nodes.items():
    # distance = distance_map[tag]
    for timescale_index in range(len(hs.timescales)):
        plotter.subplot(row_map[tag], timescale_index)
        maxs_at_time = solutions[:, timescale_index, :].max(axis=0).max()
        maxs_at_time = np.log(solutions[:, timescale_index, :] + 1).max(axis=0).max()
        plotter.add_mesh(
            pv.make_tri_mesh(*mesh),
            color="darkgrey",
            # opacity=0.15,
            opacity=0.35,
            # ambient=1,
        )
        plotter.add_mesh(
            pv.make_tri_mesh(*mesh),
            # scalars=np.log(solutions[:, timescale_index, node] + 1),
            scalars=np.log(solutions[:, timescale_index, node] + 1),
            cmap="Reds",
            clim=[0, 0.4 * maxs_at_time],
            # clim = [0, 0.08653839118620565],
            # ambient=0.5,
            emissive=True,
        )

        plotter.camera.focal_point = mesh[0][initial_nodes[node]]
        distance = distance_map[tag]
        plotter.camera.position = (
            mesh[0][initial_nodes[node]] + normals[initial_nodes[node]] * distance
        )

        plotter.camera.clipping_range = (1, 1.5 * distance)
        plotter.camera.up = [0, 1, 0]

font_size = 25
plotter.subplot(0, 0)
plotter.add_text(
    " Soma",
    position="upper_left",
    font_size=font_size,
    font_file=font_file,
    color=COMPARTMENT_PALETTE_MUTED["soma"],
)
plotter.subplot(1, 0)
plotter.add_text(
    " Shaft",
    position="upper_left",
    font_size=font_size,
    font_file=font_file,
    color=COMPARTMENT_PALETTE_MUTED["shaft"],
)
plotter.subplot(2, 0)
plotter.add_text(
    " Spine",
    position="upper_left",
    font_size=font_size,
    font_file=font_file,
    color=COMPARTMENT_PALETTE_MUTED["spine"],
)

plotter.subplot(2, 0)
plotter.add_text(
    " t = 1 (AU)",
    position="lower_left",
    font_size=font_size,
    font_file=font_file,
    color="black",
)
plotter.subplot(2, 1)
plotter.add_text(
    " t = 10 (AU)",
    position="lower_left",
    font_size=font_size,
    font_file=font_file,
    color="black",
)

plotter.subplot(2, 2)
plotter.add_text(
    " t = 100 (AU)",
    position="lower_left",
    font_size=font_size,
    font_file=font_file,
    color="black",
)

if SAVE:
    save_pyvista_figure(
        plotter,
        "heat_implicit_solve_on_neuron",
        figure_out_path,
        formats=["svg", "png"],
    )
plotter.show(jupyter_backend="static")
plotter.close()

# %%

tag_nodes = {"soma": 0, "shaft": 5, "spine": 10}

set_pyvista_theme()

window_size = np.floor(np.array((4.45, 4.64)) * 250).astype(int)

# distance = 15000
distance_map = {"soma": 30_000, "shaft": 10_000, "spine": 10_000}

row_map = {"soma": 0, "shaft": 1, "spine": 2}

for tag, node in tag_nodes.items():
    # distance = distance_map[tag]
    for timescale_index in range(len(hs.timescales)):
        plotter = pv.Plotter(
            window_size=window_size,
        )
        maxs_at_time = solutions[:, timescale_index, :].max(axis=0).max()
        maxs_at_time = np.log(solutions[:, timescale_index, :] + 1).max(axis=0).max()
        plotter.add_mesh(
            pv.make_tri_mesh(*mesh),
            color="darkgrey",
            # opacity=0.15,
            opacity=0.35,
            # ambient=1,
        )
        plotter.add_mesh(
            pv.make_tri_mesh(*mesh),
            # scalars=np.log(solutions[:, timescale_index, node] + 1),
            scalars=np.log(solutions[:, timescale_index, node] + 1),
            cmap="Reds",
            clim=[0, 0.4 * maxs_at_time],
            # clim = [0, 0.08653839118620565],
            # ambient=0.5,
            emissive=True,
        )

        plotter.camera.focal_point = mesh[0][initial_nodes[node]]
        distance = distance_map[tag]
        plotter.camera.position = (
            mesh[0][initial_nodes[node]] + normals[initial_nodes[node]] * distance
        )

        plotter.camera.clipping_range = (1, 1.5 * distance)
        plotter.camera.up = [0, 1, 0]

        if SAVE:
            save_pyvista_figure(
                plotter,
                f"heat_implicit_solve_on_neuron_{tag}_t={timescale_index}",
                figure_out_path,
                formats=["svg", "png"],
            )
        plotter.show(jupyter_backend="static")
        plotter.close()

# %%

mosaic = """
ABC
DEF
GHI
"""

panel_mapping = {
    "A": figure_out_path / "heat_implicit_solve_on_neuron_soma_t=0.svg",
    "B": figure_out_path / "heat_implicit_solve_on_neuron_soma_t=1.svg",
    "C": figure_out_path / "heat_implicit_solve_on_neuron_soma_t=2.svg",
    "D": figure_out_path / "heat_implicit_solve_on_neuron_shaft_t=0.svg",
    "E": figure_out_path / "heat_implicit_solve_on_neuron_shaft_t=1.svg",
    "F": figure_out_path / "heat_implicit_solve_on_neuron_shaft_t=2.svg",
    "G": figure_out_path / "heat_implicit_solve_on_neuron_spine_t=0.svg",
    "H": figure_out_path / "heat_implicit_solve_on_neuron_spine_t=1.svg",
    "I": figure_out_path / "heat_implicit_solve_on_neuron_spine_t=2.svg",
}

fontsize = 30
pm = PanelMosaic(
    mosaic=mosaic,
    panel_mapping=panel_mapping,
    figsize=(13.37, 13.92),
    layout="constrained",
    label_pos=None,
)

pm.axs["A"].set_ylabel(
    "Soma", fontsize=fontsize, color=COMPARTMENT_PALETTE_MUTED_HEX["soma"]
)
pm.axs["D"].set_ylabel(
    "Shaft", fontsize=fontsize, color=COMPARTMENT_PALETTE_MUTED_HEX["shaft"]
)
pm.axs["G"].set_ylabel(
    "Spine", fontsize=fontsize, color=COMPARTMENT_PALETTE_MUTED_HEX["spine"]
)
pm.axs["H"].set_xlabel(r"Increasing timescale (AU) $\rightarrow$ ", fontsize=fontsize)
pm.show()
pm.write(figure_out_path / "multipanel_solve_on_neuron")


# %%

full_mesh = (raw_mesh.vertices, raw_mesh.faces)

result = chunked_hks_pipeline(full_mesh, verbose=True, n_jobs=-2)

# %%
t_min = 50000.0
t_max = 20000000.0
timesteps = 32
timescales = np.geomspace(t_min, t_max, timesteps)

# %%

simple_mesh = result.simple_mesh

# root_targets = targets.query("post_pt_root_id == @root_id").copy()

root_targets["tag"] = root_targets["tag"].replace({"soma_spine": "spine"})
root_targets = root_targets.query("tag.isin(['soma', 'spine', 'shaft'])")

points = root_targets[
    ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
].values

mesh_indices = project_points_to_mesh(points, simple_mesh, distance_threshold=1000)
root_targets["mesh_index"] = mesh_indices
root_targets = root_targets.query("mesh_index != -1")

np.random.seed(8)
n_samples = 100
samples = root_targets.groupby("tag").sample(n_samples)

features = result.simple_features.copy().values

scaled_features = features / np.nansum(features, axis=0)[None, :]

# %%

set_matplotlib_theme(font_scale=1.5)

hks_df = pd.DataFrame(
    scaled_features[samples["mesh_index"].values, :],
)
hks_df["label"] = pd.Categorical(
    samples["tag"].values, categories=["soma", "shaft", "spine"]
)
hks_df.index.name = "node"

fig, axs = plt.subplots(1, 3, figsize=(10, 6), sharex=True, sharey=True)

melt_hks_df = (
    hks_df.drop(columns="label")
    .reset_index()
    .melt(var_name="timescale_index", value_name="hks", id_vars=["node"])
)

timescale_map = dict(zip(range(timesteps), timescales / timescales[0]))

for i, (label, label_hks_df) in enumerate(hks_df.groupby("label")):
    ax = axs[i]
    melt_label_hks_df = (
        label_hks_df.drop(columns="label")
        .reset_index()
        .melt(var_name="timescale_index", value_name="hks", id_vars=["node"])
    )
    melt_label_hks_df["timescale"] = melt_label_hks_df["timescale_index"].map(
        timescale_map
    )

    sns.lineplot(
        data=melt_hks_df,
        x="timescale",
        y="hks",
        ax=axs[i],
        # label=label,
        alpha=0.05,
        color="dimgrey",
        estimator=None,
        units="node",
        linewidth=1,
        zorder=-1,
    )
    sns.lineplot(
        data=melt_label_hks_df,
        x="timescale",
        y="hks",
        ax=axs[i],
        # label=label,
        alpha=0.5,
        estimator=None,
        units="node",
        color=COMPARTMENT_PALETTE_HEX[label],
    )
    axs[i].set_title(label.capitalize(), y=0.97, color=COMPARTMENT_PALETTE_HEX[label])
    ax.set_yscale("log")
    ax.set_xlabel("Time (AU)")
    # ax.set_xticks([0, 15, 31])
    # ax.set_xticklabels([1, 16, 32])
    ax.set_ylabel("Scaled HKS (AU)")
    ax.set_ylim(5e-7, 5e-6)

if SAVE:
    save_matplotlib_figure(fig, "hks_curves_by_label", figure_out_path)

# %%

fontsize = 30
set_matplotlib_theme(font_scale=1.0)

hks_df = pd.DataFrame(
    scaled_features[samples["mesh_index"].values, :],
)
hks_df["label"] = pd.Categorical(
    samples["tag"].values, categories=["soma", "shaft", "spine"]
)
hks_df.index.name = "node"

fig, axs = plt.subplots(
    3, 1, figsize=(6.44, 13.92), sharex=True, sharey=True, layout="constrained"
)

melt_hks_df = (
    hks_df.drop(columns="label")
    .reset_index()
    .melt(var_name="timescale_index", value_name="hks", id_vars=["node"])
)
timescale_map = dict(zip(range(timesteps), timescales / timescales[0]))

melt_hks_df["timescale"] = melt_hks_df["timescale_index"].map(timescale_map)

for i, (label, label_hks_df) in enumerate(hks_df.groupby("label")):
    ax = axs[i]
    melt_label_hks_df = (
        label_hks_df.drop(columns="label")
        .reset_index()
        .melt(var_name="timescale_index", value_name="hks", id_vars=["node"])
    )
    melt_label_hks_df["timescale"] = melt_label_hks_df["timescale_index"].map(
        timescale_map
    )

    sns.lineplot(
        data=melt_hks_df,
        x="timescale",
        y="hks",
        ax=axs[i],
        # label=label,
        alpha=0.05,
        color="dimgrey",
        estimator=None,
        units="node",
        linewidth=1,
        zorder=-1,
    )
    sns.lineplot(
        data=melt_label_hks_df,
        x="timescale",
        y="hks",
        ax=axs[i],
        # label=label,
        alpha=0.5,
        estimator=None,
        units="node",
        color=COMPARTMENT_PALETTE_MUTED_HEX[label],
    )
    axs[i].set_title(
        label.capitalize(),
        y=0.97,
        color=COMPARTMENT_PALETTE_MUTED_HEX[label],
        fontsize=fontsize,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Timescale (AU)", fontsize=fontsize)
    # ax.set_xticks([0, 15, 31])
    # ax.set_xticklabels([1, 16, 32])
    ax.set_ylabel("Scaled HKS (AU)", fontsize=fontsize)
    ax.set_ylim(4e-8, 4e-6)
    ax.set_xscale("log")

SAVE = True
if SAVE:
    save_matplotlib_figure(fig, "hks_curves_by_label", figure_out_path)

# %%


figsize = (28, 14)
label_fontsize = 50
panel_borders = False
layout = "constrained"
mosaic = """
AAAABBCC
"""
panel_mapping = {
    "A": figure_out_path / "multipanel_solve_on_neuron.svg",
    "B": figure_out_path / "many_label_examples_on_neuron_zoom.svg",
    "C": figure_out_path / "hks_curves_by_label.svg",
}

panel = PanelMosaic(
    mosaic,
    panel_mapping,
    figsize=figsize,
    panel_borders=panel_borders,
    layout=layout,
    label_fontsize=label_fontsize,
)
ax: plt.Axes = panel.axs["B"]
ax.annotate(
    "Labeled\npostsynapses",
    xy=(0.47, 0.65),
    xytext=(-0.0, 0.75),
    textcoords="axes fraction",
    fontsize=fontsize,
    color="black",
    arrowprops=dict(facecolor="black", arrowstyle="-|>", lw=2),
    horizontalalignment="left",
    verticalalignment="top",
)
panel.show()
panel.write(figure_out_path / "diagram_heat_on_labeled_cell")
# %%
panel.show_dummies()


# %%

# figsize = (28, 14)
# label_fontsize = 50
# panel_borders = True
# layout = "constrained"
# mosaic = """
# AAAABBCC
# """
# panel_mapping = {
#     "B": figure_out_path / "many_label_examples_on_neuron_zoom.svg",
#     "A": figure_out_path / "heat_implicit_solve_on_neuron.svg",
#     "C": figure_out_path / "hks_curves_by_label.svg",
# }

# panel = PanelMosaic(
#     mosaic,
#     panel_mapping,
#     figsize=figsize,
#     panel_borders=panel_borders,
#     layout=layout,
#     label_fontsize=label_fontsize,
# )

# panel.show()
# panel.show_dummies()
