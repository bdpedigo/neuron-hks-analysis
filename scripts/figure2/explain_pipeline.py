# %%

import numpy as np
import pyvista as pv
import seaborn as sns
from caveclient import CAVEclient
from fast_simplification import simplify

from meshmash import MeshStitcher
from analysis import (
    COMPARTMENT_PALETTE_MUTED_HEX,
    FIG_PATH,
    FONT_PATH,
    load_model,
    mask_dendrite_by_client_skeleton,
    save_matplotlib_figure,
    save_pyvista_figure,
    set_matplotlib_theme,
    set_pyvista_theme,
)

# COMPARTMENT_PALETTE["shaft"] = (221, 205, 37)
font_file = FONT_PATH

figure_out_path = FIG_PATH / "explain_pipeline"

set_pyvista_theme()

client = CAVEclient("minnie65_phase3_v1", version=1300)
cv = client.info.segmentation_cloudvolume(progress=False)

query_kwargs = {
    "desired_resolution": [1, 1, 1],
    "split_positions": True,
    "log_warning": False,
}

# %%

root_id = 864691136025099065
raw_mesh = cv.mesh.get(
    root_id, remove_duplicate_vertices=True, deduplicate_chunk_boundaries=False
)[root_id]

# %%
mesh = (raw_mesh.vertices, raw_mesh.faces)
simple_mesh = simplify(*mesh, target_reduction=0.7)

try:
    simple_mesh = mask_dendrite_by_client_skeleton(
        simple_mesh,
        root_id,
        client,
        distance_threshold=5,
    )
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

base_size = (8.04, 5.83)

window_size = np.floor(np.array(base_size) * 100).astype(int)
plotter = pv.Plotter(shape=(1, 2), border_color="white", window_size=window_size)

mesh_params = dict(
    color="darkgrey",
    show_edges=True,
    edge_color="black",
    line_width=15,
)

plotter.subplot(0, 0)
plotter.add_mesh(pv.make_tri_mesh(*mesh), **mesh_params)

plotter.subplot(0, 1)
plotter.add_mesh(pv.make_tri_mesh(*simple_mesh), **mesh_params)

plotter.link_views()
plotter.camera_position = [
    (793529.5368801871, 376050.93514928065, 853973.1183615582),
    (797608.4761118466, 376719.4115159685, 849445.1221647501),
    (0.6724925787645127, -0.5171395794948519, 0.5294529127566905),
]
plotter.show(jupyter_backend="static")

save_pyvista_figure(
    plotter,
    filename="mesh_simplification",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)
# %%

np.random.seed(8888)

ms = MeshStitcher(
    simple_mesh,
    verbose=True,
)
ms.split_mesh()

# %%

wide_cpos = [
    (696386.7915661471, 428009.9887418418, 1409355.1204906167),
    (757098.5343463086, 495593.5471083228, 853471.0010635282),
    (0.007377127303158819, -0.992759470032648, -0.11989250457492864),
]

plotter = pv.Plotter(window_size=window_size)

colors = sns.color_palette("husl", len(ms.submeshes))
rgb = np.stack([colors[i] for i in ms.submesh_mapping])

plotter.add_mesh(
    pv.make_tri_mesh(*simple_mesh),
    scalars=rgb,
    rgb=True,
)

plotter.camera_position = [
    (792870.2282990322, 368229.46879586804, 1381074.3946477291),
    (760289.8939015344, 529237.536903177, 861652.1558706994),
    (0.05474924852423706, -0.9527550613108655, -0.29876464471647346),
]
plotter.camera_position = [
    (761153.2680168052, 509647.1074853752, 908522.2031611485),
    (770113.8973525754, 535899.8879023538, 854077.5397970545),
    (-0.030963872323981602, -0.8983162284683616, -0.4382569934194806),
]
plotter.camera_position = [
    (763964.1691384615, 566308.360592099, 903015.5739960294),
    (779389.9684833995, 542885.541278852, 859759.017366439),
    (0.9224876144050025, -0.08702061905461216, 0.37609043211536164),
]
plotter.camera_position = [
    (763813.1846672246, 585337.1103602368, 887288.2747169487),
    (779781.1755592477, 544200.1784752707, 858761.6450168767),
    (0.9049416618614838, 0.05903143134870292, 0.42142126042749595),
]
plotter.camera_position = [
    (842704.7356809109, 528811.466726638, 852952.9455932118),
    (752215.2308465753, 531061.233506078, 848930.4540050552),
    (-0.016628526254827462, -0.9841822882670842, -0.176376629903053),
]
plotter.camera.clipping_range = (30000, 121000)

save_pyvista_figure(
    plotter,
    filename="mesh_subdivision",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)
# %%

from meshmash import spectral_geometry_filter

font_size = 20

i = 5
submesh = ms.submeshes[i]
evals, evecs = spectral_geometry_filter(submesh, max_eigenvalue=1e-6, drop_first=False)

# %%
select_evals = [5, 20, 30, 151]
n_evals = len(select_evals)
window_size = np.floor(np.array((8.04, 5.83)) * 100).astype(int)
plotter = pv.Plotter(shape=(1, n_evals), window_size=window_size, border_color="white")

for i, k in enumerate(select_evals):
    x = evecs[:, k]
    clim = max(np.abs(np.percentile(x, (1, 99))))
    plotter.subplot(0, i)
    plotter.add_mesh(
        pv.make_tri_mesh(*submesh),
        scalars=evecs[:, k],
        cmap="coolwarm",
        clim=[-clim, clim],
    )
    # plotter.add_text(f"Eigenvector {k}", font_file=font_file, font_size=font_size)

plotter.link_views()
plotter.camera_position = [
    (734932.8126930452, 504203.72146433586, 909203.0301436597),
    (722542.4866154874, 541726.2457023762, 888787.3015576294),
    (-0.5385994521805743, 0.2584761308601863, 0.8019356083167358),
]
plotter.zoom_camera(1.5)
plotter.enable_fly_to_right_click()
# plotter.show(jupyter_backend="static")
plotter.show()
save_pyvista_figure(
    plotter,
    filename="mesh_eigenvectors",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)


# %%

evals, evecs = spectral_geometry_filter(submesh, max_eigenvalue=5e-5, drop_first=True)

# %%

node_index = 8964

times = [0, 1e6, 1e7, 1e8]

# plotter = pv.Plotter(shape=(1, len(times)), window_size=(2000, 1000))
eff_dpi = 100
window_size = (eff_dpi * np.array((13.37, 7.92))).astype(int)
window_size = (eff_dpi * np.array((15.36, 7.92))).astype(int)
font_size = 20

plotter = pv.Plotter(shape=(1, len(times)), window_size=window_size, border_width=0)

mesh_poly = pv.make_tri_mesh(*submesh)
mesh_poly.compute_normals(cell_normals=False, point_normals=True)
normals = mesh_poly.point_normals

heat_vals = []

text = False
arrows = False
point_size = 500

for i, t in enumerate(times):
    plotter.subplot(0, i)

    coefs = np.exp(-t * evals)
    out = (evecs[node_index, :] * coefs) @ evecs.T
    out[out < 0] = 1e-12
    if t == 0:
        out = np.zeros_like(out)
        out[node_index] = 1
        out += 1e-12

    heat_vals.append(out[node_index])

    x = np.log(out)
    clim = np.percentile(x, (1, 99))
    if i == 0:
        if text:
            plotter.add_text(
                "Initial heat\nat point",
                font_file=font_file,
                font_size=font_size,
                position=(50, 500),
            )
    if i == 1:
        if text:
            plotter.add_text(
                "Heat remaining\nat point",
                font_file=font_file,
                font_size=font_size,
                position=(100, 650),
            )

    plotter.add_points(
        np.stack([submesh[0][node_index], submesh[0][node_index]]).squeeze(),
        scalars=np.array([x[node_index].reshape(1, 1), x[node_index].reshape(1, 1)]),
        render_points_as_spheres=True,
        point_size=point_size,
        cmap="Reds",
        clim=[-20, -14],
        opacity=0.7,
        emissive=True,
    )
    if arrows:
        vec = np.array([-0.49855718, 0.35485756, 0.7908963]).reshape(1, 3)
        vec = -np.array([0.5, 0.5, -0]).reshape(1, 3)
        plotter.add_arrows(
            submesh[0][node_index].reshape(1, 3) + vec * 2000,
            -vec * 200,
            mag=8,
            line_width=10,
            color="black",
        )
    plotter.add_points(
        submesh[0][node_index].reshape(1, 3),
        render_points_as_spheres=True,
        point_size=point_size + 10,
        # style="points_gaussian",
        emissive=True,
        color="black",
        diffuse=1,
        specular=1,
        opacity=0.7,
    )
    plotter.add_mesh(
        pv.make_tri_mesh(*submesh),
        scalars=x,
        cmap="Reds",
        clim=[-20, -14],
        # clim=clim,
    )

    if text:
        plotter.add_text(
            f" t = {t / 1e6:.0f} (AU)",
            font_file=font_file,
            font_size=25,
            position="lower_left",
            color="black",
        )

plotter.link_views()
plotter.camera_position = [
    (707408.532850284, 503900.38123590423, 909704.4018922036),
    (700866.8937209327, 516762.87477289204, 889591.9275506003),
    (-0.8038888428426355, -0.5841211241460463, -0.11209478435452733),
]
# plotter.camera.zoom(1.75)

plotter.show(jupyter_backend="static")
save_pyvista_figure(
    plotter,
    filename="multiscale_heat_kernel",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)

# %%


plotter = pv.Plotter(window_size=[300, 600])

mesh_poly = pv.make_tri_mesh(*submesh)
sample_indices = np.random.choice(len(submesh[0]), size=5000, replace=False)
plotter.add_mesh(pv.make_tri_mesh(*submesh), color="lightgrey")
plotter.add_points(
    submesh[0][sample_indices],
    color="black",
    render_points_as_spheres=True,
    point_size=5,
)

plotter.camera_position = [
    (707408.532850284, 503900.38123590423, 909704.4018922036),
    (700866.8937209327, 516762.87477289204, 889591.9275506003),
    (-0.8038888428426355, -0.5841211241460463, -0.11209478435452733),
]
plotter.show(jupyter_backend="static")
# %%
window_size = np.floor(np.array((8.04, 5.83)) * 100).astype(int)

t = 1e6
node_index = 8965
coefs = np.exp(-t * evals)
out = (evecs[node_index, :] * coefs) @ evecs.T
out[out < 0] = 1e-12
x = np.log(out)
clim = np.percentile(x, (1, 99))

plotter = pv.Plotter()
plotter.add_points(
    submesh[0][node_index].reshape(1, 3),
    color="black",
    point_size=500,
    render_points_as_spheres=True,
)
plotter.add_mesh(
    pv.make_tri_mesh(*submesh),
    color="grey",
    opacity=0.2,
)
plotter.add_mesh(
    pv.make_tri_mesh(*submesh),
    scalars=x,
    cmap="Reds",
    clim=[-20, -14],
)
plotter.camera_position = [
    (690532.8424083234, 519634.5831712736, 878074.5599839322),
    (699562.9880630554, 515132.88803256705, 889345.9762878821),
    (0.10495369363765703, -0.8917289123226668, -0.44023206280284355),
]
# plotter.show()
save_pyvista_figure(
    plotter,
    filename="single_heat",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)

# %%

import os

os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/cairo/lib:$DYLD_LIBRARY_PATH"

from matplotlib import patheffects as pe

from panel_mosaic import PanelMosaic

mosaic = """
AABBCC
DDEEFF
GGHHII
"""
label_mapping = {
    "A": "Input: mesh",
    "B": "Mesh simplification",
    "C": "Mesh subdivision",
    "D": "Eigendecomposition",
    "E": "Heat kernel",
    "F": "Feature agglomeration",
    "G": "Classification",
    "H": "Connected components",
    "I": "Morphometry",
}
panel_mapping = {
    "B": figure_out_path / "mesh_simplification.png",
    "C": figure_out_path / "mesh_subdivision.png",
    "D": figure_out_path / "mesh_eigenvectors.png",
    "E": figure_out_path / "single_heat.png",
    "F": figure_out_path / "agglomeration.png",
    "G": figure_out_path / "classification.png",
}

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping=label_mapping,
    figsize=(28, 20),
    label_pos=(0.0, 0.9),
    layout="constrained",
    label_fontsize=50,
    panel_borders=True,
    label_dodge="top",
    label_dodge_factor=0.015,
)

axs = pm.axs

for label in label_mapping.keys():
    arrow_ax = pm.draw_arrow(label, "right", size=0.15, scale=50, dummy=label == "I")

ax = pm.axs["A"]

# set background color to light grey
# ax.set_facecolor("lightgrey")
# from matplotlib.patches import FancyBboxPatch, Rectangle

pm.lock_axes()
pm.show()
# ax.autoscale(False)
# rect = Rectangle(
#     (0, 0),
#     1.15,
#     1.15,
#     facecolor="lightgrey",
#     clip_on=False,
#     zorder=-1,
# )
# rect = FancyBboxPatch(
#     (-0.05, -0.05),
#     1.15,
#     1.15,
#     boxstyle="round,pad=0.02,rounding_size=0.05",
#     facecolor="lightgrey",
#     clip_on=False,
#     zorder=-1,
# )
# ax.add_patch(rect)

# pm.lock_axes()

# draw arrows between panels
# for ax in axs.values():
#     ax.annotate(
#         "",
#         xy=(1.2, 0.5),
#         xytext=(0.5, 0.5),
#         textcoords="axes fraction",
#         arrowprops=dict(
#             facecolor="black", arrowstyle="-|>", lw=2, transform=ax.transAxes
#         ),
#         horizontalalignment="center",
#         verticalalignment="center",
#         clip_on=True,
#     )
#
# pm.show_dummies()
# plt.show()

# %%


mosaic = """
AABBCC
DDEEFF
"""
label_mapping = {
    "A": "Mesh simplification",
    "B": "Mesh subdivision",
    "C": "Eigendecomposition",
    "D": "Heat kernel computation",
    "E": "Feature agglomeration",
    "F": "Classification",
}
panel_mapping = {
    "A": figure_out_path / "mesh_simplification.png",
    "B": figure_out_path / "mesh_subdivision.png",
    "C": figure_out_path / "mesh_eigenvectors.png",
    "D": figure_out_path / "single_heat.png",
    "E": figure_out_path / "agglomeration.png",
    "F": figure_out_path / "classification.png",
}

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping=label_mapping,
    figsize=(28, 20 * 0.66),
    label_pos=(0.0, 0.9),
    layout="constrained",
    label_fontsize=30,
    panel_borders=False,
    label_dodge="top",
    label_dodge_factor=0.015,
    gridspec_kw={"hspace": 0.1},
)

axs = pm.axs

for label in label_mapping.keys():
    arrow_ax = pm.draw_arrow(label, "right", size=0.15, scale=50, dummy=label == "F")

ax = pm.axs["A"]

pm.lock_axes()
pm.show()

pm.write(
    figure_out_path / "explain_pipeline_mosaic_simple",
)

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.rcParams["font.size"] = 20
mpl.rcParams["text.usetex"] = False
mpl.rcParams["text.latex.preamble"] = r"\usepackage{{amsmath}}"

fig, ax = plt.subplots(1, 1, figsize=(7.43 / 2, 7.92))


def draw_bracket(ax, start, end, axis="x", color="black"):
    lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 500)
    tan = np.tan(lx)
    curve = np.hstack((tan[::-1], tan))
    x = np.linspace(start, end, 1000)
    if axis == "x":
        ax.plot(x, -curve, color=color)
    elif axis == "y":
        ax.plot(-curve / 200 + 0.1, x, color=color)


draw_bracket(ax, 0, 1, axis="y")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# ax.text(0.6, 0.6, r"$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$")

# %%

# poly = pv.make_tri_mesh(submesh[0].astype(np.float32), submesh[1].astype(np.int32))
# poly['evec'] = evecs[:, 1].astype(np.float32)
# poly.save("evec1.vtk", binary=False)

# from meshio import Mesh

# m = Mesh(
#     points=submesh[0].astype(np.float32),
#     cells={"triangle": submesh[1].astype(np.int32)},
# )
# m.write("evec1.vtk", binary=False)


# %%
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

cmap_name = "Reds"
cmap = colormaps[cmap_name]
norm = Normalize(vmin=-20, vmax=-14)
sm = ScalarMappable(norm=norm, cmap=cmap)


# times = (
#     [0] + np.geomspace(1e6, 1e7, 10).tolist() + np.geomspace(1e7, 1e8, 10).tolist()[1:]
# )
times = [0] + np.geomspace(1e5, 1e8, 20).tolist()
times = np.array(times)


def compute_for_time(t):
    coefs = np.exp(-t * evals)
    out = (evecs[node_index, :] * coefs) @ evecs.T
    out[out < 0] = 1e-12
    if t == 0:
        out = np.zeros_like(out)
        out[node_index] = 1
        out += 1e-12
    return out[node_index]


heat_vals = []
for i, t in enumerate(times):
    heat = compute_for_time(t)
    heat_vals.append(heat)
    x = np.log(out)
heat_vals = np.array(heat_vals)

highlight_times = np.array([0, 1e6, 1e7, 1e8])
highlight_heats = []
for t in highlight_times:
    highlight_heats.append(compute_for_time(t))
highlight_heats = np.array(highlight_heats)

scale = 1e6
times = times / scale
highlight_times = highlight_times / scale


# %%
set_matplotlib_theme()
fontsize = 30
fig, ax = plt.subplots(figsize=(7.43, 7.92))
sns.lineplot(x=times, y=heat_vals, ax=ax, linewidth=2, color="black")
sns.scatterplot(
    x=highlight_times,
    y=highlight_heats,
    hue=np.log(highlight_heats),
    ax=ax,
    legend=False,
    palette=cmap,
    zorder=10,
    linewidth=1,
    s=150,
    edgecolor="black",
)
ax.set_yscale("log")
ax.set_xlabel("Time (AU)", fontsize=fontsize)
ax.set_ylabel("Heat remaining (AU)", fontsize=fontsize)

ax.text(
    0.5,
    0.7,
    "HKS vector",
    transform=ax.transAxes,
    fontsize=fontsize,
)
ax.text(
    0.5,
    0.6,
    r"$\left[h_1, \ldots, h_k\right]$",
    transform=ax.transAxes,
    fontsize=fontsize,
)
# add annotation arrows from some of the points to the text
end_pos = [1, (0.55, 0.58), (0.66, 0.58), (0.76, 0.58)]
for i in range(1, 4):
    ax.annotate(
        "",
        xy=end_pos[i],
        xytext=(highlight_times[i], highlight_heats[i]),
        textcoords="data",
        xycoords="axes fraction",
        arrowprops=dict(
            facecolor="black", arrowstyle="-|>", lw=2, shrinkA=15, shrinkB=5
        ),
        horizontalalignment="right",
        verticalalignment="top",
    )
save_matplotlib_figure(
    fig,
    filename="single_heat_trace",
    out_path=figure_out_path,
    formats=["svg", "png"],
)

# %%
from meshmash import compute_hks


# %%


chks_params = {
    "drop_first": True,
    "max_eigenvalue": 1e-05,
    "mollify_factor": 1e-05,
    "n_components": 32,
    "robust": True,
    "t_max": 20000000.0,
    "t_min": 50000.0,
    "truncate_extra": True,
}
features = compute_hks(submesh, **chks_params)

# %%
from meshmash import agglomerate_mesh, shuffle_label_mapping

labels = agglomerate_mesh(submesh, np.log(features), distance_thresholds=3.0).squeeze()

labels = shuffle_label_mapping(labels)

# %%

plotter = pv.Plotter(window_size=window_size)
colors = sns.color_palette("husl", np.max(labels) + 1).as_hex()
plotter.add_mesh(
    pv.make_tri_mesh(*submesh),
    scalars=labels,
    cmap=colors,
    interpolate_before_map=False,
    show_edges=False,
)
plotter.camera_position = [
    (691673.2824964552, 505369.27117339516, 905212.8317382315),
    (690760.625297867, 502177.7451042422, 896293.809805252),
    (-0.8978140554931682, 0.4356941523220054, -0.06403536047622342),
]
plotter.enable_fly_to_right_click()
plotter.show(jupyter_backend="static")
# plotter.show()

save_pyvista_figure(
    plotter,
    filename="agglomeration",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)

# %%

VERSION = 1412
N_JOBS = -1
VERBOSE = True
PARAMETER_NAME = "absolute-solo-yak"
DATASTACK = "minnie65_phase3_v1"
N_PER_BATCH = 1000

model = load_model("simple_hks_model")

# %%
import pandas as pd

mean_features = pd.DataFrame(features).groupby(by=labels).mean().values
posteriors = model.predict_proba(np.log(mean_features))
posteriors = posteriors[labels]
plotter = pv.Plotter(window_size=window_size)
plotter.add_mesh(
    pv.make_tri_mesh(*submesh),
    scalars=posteriors[:, 1] > 0.5,
    interpolate_before_map=False,
    cmap=[
        COMPARTMENT_PALETTE_MUTED_HEX["shaft"],
        COMPARTMENT_PALETTE_MUTED_HEX["spine"],
    ],
    show_edges=False,
)
plotter.camera_position = [
    (691950.9583624994, 500231.76529024437, 884896.4950618432),
    (692935.685593285, 506349.5209391108, 894429.3798137115),
    (0.9590183311048961, -0.27293547159452947, 0.0760925025889659),
]
plotter.enable_fly_to_right_click()
# plotter.show()
plotter.show(jupyter_backend="static")

save_pyvista_figure(
    plotter,
    filename="classification",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)

# %%
from tqdm import tqdm

from meshmash import subset_mesh_by_indices

all_borders = []
unique_labels = np.unique(labels)
for label in tqdm(unique_labels):
    label_mesh = subset_mesh_by_indices(submesh, np.where(labels == label)[0])
    if len(label_mesh[1]) == 0:
        continue
    poly = pv.make_tri_mesh(*label_mesh)
    poly["index"] = np.arange(poly.n_points)
    edges = poly.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    all_borders.append(edges)

# %%
plotter = pv.Plotter(window_size=window_size)
colors = sns.color_palette("husl", np.max(labels) + 1).as_hex()
plotter.add_mesh(
    pv.make_tri_mesh(*submesh),
    scalars=labels,
    cmap=colors,
    interpolate_before_map=False,
    show_edges=False,
)
for border in all_borders:
    if border.n_points > 0:
        plotter.add_mesh(border, color="black", line_width=5)
        plotter.add_points(
            border.points, color="black", point_size=5, render_points_as_spheres=True
        )
plotter.camera_position = [
    (691673.2824964552, 505369.27117339516, 905212.8317382315),
    (690760.625297867, 502177.7451042422, 896293.809805252),
    (-0.8978140554931682, 0.4356941523220054, -0.06403536047622342),
]
plotter.enable_fly_to_right_click()
plotter.show(jupyter_backend="static")
# plotter.show()

# save_pyvista_figure(
#     plotter,
#     filename="agglomeration",
#     out_path=figure_out_path,
#     formats=["svg", "png"],
#     scale=5,
# )


# %%

panel_mapping = {
    "A": figure_out_path / "multiscale_heat_kernel.png",
    "B": figure_out_path / "single_heat_trace.svg",
}
label_mapping = {
    "A": "A",
    "B": "B",
    "C": "",
}

pm = PanelMosaic(
    mosaic,
    panel_mapping=panel_mapping,
    label_mapping=label_mapping,
    figsize=(28, 8),
    label_fontsize=50,
    layout="constrained",
    panel_borders=False,
)
# pm.show_dummies()
ax = pm.axs["A"]
fontsize = 30
# add an annotation arrow to the plot
ann = ax.annotate(
    "Initial heat\nat point",
    xy=(0.13, 0.65),
    xytext=(0.0, 0.9),
    textcoords="axes fraction",
    fontsize=fontsize,
    color="black",
    arrowprops=dict(facecolor="black", arrowstyle="-|>", lw=2),
    horizontalalignment="left",
    verticalalignment="top",
)
ann.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
ann = ax.annotate(
    "Heat remaining\nat point",
    xy=(0.375, 0.65),
    xytext=(0.225, 0.9),
    textcoords="axes fraction",
    fontsize=fontsize,
    color="black",
    arrowprops=dict(facecolor="black", arrowstyle="-|>", lw=2),
    horizontalalignment="left",
    verticalalignment="top",
)
ann.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

for i in range(4):
    text = ax.text(
        i / 4,
        0.02,
        f"t={int(highlight_times[i]):,} (AU)",
        fontsize=fontsize,
        color="black",
    )
    text.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

ax = pm.axs["C"]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# trigger a draw from underlying renderer
pm.lock_axes()
ax = pm.axs["C"]
ax.annotate(
    "",
    xy=(-0.0, 0.75),
    xytext=(-0.6, 0.75),
    textcoords="axes fraction",
    arrowprops=dict(facecolor="black", arrowstyle="-|>", lw=2),
    horizontalalignment="center",
    verticalalignment="top",
    clip_on=False,
)
ax.text(
    0.05,
    0.75,
    "Rescaling",
    fontsize=fontsize + 1,
    ha="left",
    va="center",
    clip_on=True,
    transform=ax.transAxes,
    zorder=10,
)

pm.show()
pm.write(figure_out_path / "explain_hks")
pm.close()

# %%
