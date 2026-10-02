# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from caveclient import CAVEclient
from fast_simplification import simplify
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from meshmash import (
    MeshStitcher,
    agglomerate_mesh,
    compute_hks,
    shuffle_label_mapping,
    spectral_geometry_filter,
)
from panel_mosaic import PanelMosaic

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

cpos1 = [
    (690532.8424083234, 519634.5831712736, 878074.5599839322),
    (699562.9880630554, 515132.88803256705, 889345.9762878821),
    (0.10495369363765703, -0.8917289123226668, -0.44023206280284355),
]
cpos2 = [
    (691673.2824964552, 505369.27117339516, 905212.8317382315),
    (690760.625297867, 502177.7451042422, 896293.809805252),
    (-0.8978140554931682, 0.4356941523220054, -0.06403536047622342),
]
cpos3 = [
    (691950.9583624994, 500231.76529024437, 884896.4950618432),
    (692935.685593285, 506349.5209391108, 894429.3798137115),
    (0.9590183311048961, -0.27293547159452947, 0.0760925025889659),
]
cpos5 = [
    (687534.3121194638, 507134.3830310004, 907036.5741242096),
    (691021.5197133131, 501929.8393810764, 895942.1087349937),
    (-0.8718083448560751, 0.2770312428408315, -0.40398502488257576),
]

cpos = cpos3

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
plotter.camera_position = cpos
plotter.camera_position = [
    (693989.7152591199, 505345.7094142076, 889785.6072208692),
    (694613.3971637692, 507585.2429081523, 893363.4222122625),
    (0.9544426223112032, -0.2977246344442238, 0.019983061846656134),
]
# plotter.show(jupyter_backend="static")
# plotter.show()
save_pyvista_figure(
    plotter,
    filename="mesh_simplification",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
    show=True
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
plotter.show()
save_pyvista_figure(
    plotter,
    filename="mesh_eigenvectors",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)
# %%
select_evals = [5, 31, -50]
select_evals = [5, 31, -1400]
n_evals = len(select_evals)
window_size = np.floor(np.array((8.04, 5.83)) * 100).astype(int)
plotter = pv.Plotter(shape=(n_evals, 1), window_size=window_size, border_color="white")

cpos4 = [
    (669445.9697108954, 504979.9262651722, 882766.5465656943),
    (692935.685593285, 506349.5209391108, 894429.3798137115),
    (0.39907005177099764, -0.5411461701524402, -0.7402053203732325),
]
for i, k in enumerate(select_evals):
    x = evecs[:, k]
    clim = max(np.abs(np.percentile(x, (1, 99))))
    plotter.subplot(i, 0)
    plotter.add_mesh(
        pv.make_tri_mesh(*submesh),
        scalars=evecs[:, k],
        cmap="coolwarm",
        clim=[-clim, clim],
    )
    # plotter.add_text(f"Eigenvector {k}", font_file=font_file, font_size=font_size)


plotter.link_views()
# plotter.camera_position = [
#     (734932.8126930452, 504203.72146433586, 909203.0301436597),
#     (722542.4866154874, 541726.2457023762, 888787.3015576294),
#     (-0.5385994521805743, 0.2584761308601863, 0.8019356083167358),
# ]

plotter.camera_position = cpos
plotter.camera_position = [
    (693691.4351564733, 500428.835000885, 885687.2010647365),
    (692935.685593285, 506349.5209391108, 894429.3798137115),
    (0.9171912528176493, -0.28858419282194975, 0.2747350895100485),
]
plotter.zoom_camera(1.2)
plotter.enable_fly_to_right_click()
plotter.show()
save_pyvista_figure(
    plotter,
    filename="mesh_eigenvectors",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
    show=True,
)

# %%

evals, evecs = spectral_geometry_filter(submesh, max_eigenvalue=5e-5, drop_first=True)

# %%

node_index = 8964

times = [0, 1e6, 1e7, 1e8]

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

plotter.show(jupyter_backend="static")
save_pyvista_figure(
    plotter,
    filename="multiscale_heat_kernel",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)


# %%
window_size = np.floor(np.array((8.04, 5.83)) * 100).astype(int)


focal_point = cpos[1]

node_index = np.argmin(np.linalg.norm(submesh[0] - focal_point, axis=1))
# node_index += 245

t = 1e6

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

plotter.camera_position = cpos
save_pyvista_figure(
    plotter,
    filename="single_heat",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
    show=True,
)


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
plotter.camera_position = cpos


# plotter.enable_fly_to_right_click()
# plotter.show(jupyter_backend="static")

save_pyvista_figure(
    plotter,
    filename="agglomeration",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
    show=True,
)

# %%

VERSION = 1412
N_JOBS = -1
VERBOSE = True
PARAMETER_NAME = "absolute-solo-yak"
DATASTACK = "minnie65_phase3_v1"
N_PER_BATCH = 1000

model = load_model("simple_hks_model")

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
plotter.camera_position = cpos

#  [
#     (691950.9583624994, 500231.76529024437, 884896.4950618432),
#     (692935.685593285, 506349.5209391108, 894429.3798137115),
#     (0.9590183311048961, -0.27293547159452947, 0.0760925025889659),
# ]
plotter.enable_fly_to_right_click()

plotter.show(jupyter_backend="static")

save_pyvista_figure(
    plotter,
    filename="classification",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)

# %%
# base_size = (8.04, 5.83)

# window_size = np.floor(np.array(base_size) * 100).astype(int)
plotter = pv.Plotter(shape=(2, 1), border_color="white", window_size=window_size)

mesh_params = dict(
    color="darkgrey",
    show_edges=True,
    edge_color="black",
    line_width=15,
)

plotter.subplot(0, 0)
plotter.add_mesh(pv.make_tri_mesh(*submesh), **mesh_params)


plotter.subplot(1, 0)
plotter.add_mesh(pv.make_tri_mesh(*simple_mesh), **mesh_params)

plotter.link_views()
# plotter.camera_position = [
#     (793529.5368801871, 376050.93514928065, 853973.1183615582),
#     (797608.4761118466, 376719.4115159685, 849445.1221647501),
#     (0.6724925787645127, -0.5171395794948519, 0.5294529127566905),
# ]
plotter.camera_position = cpos
plotter.show(jupyter_backend="static")

save_pyvista_figure(
    plotter,
    filename="mesh_simplification",
    out_path=figure_out_path,
    formats=["svg", "png"],
    scale=5,
)

# %%

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

pm.lock_axes()
pm.show()


# %%


mosaic = """
AABBCC
DDEEFF
"""
label_mapping = {
    "A": "Mesh simplification",
    "B": "Mesh subdivision",
    "C": "Eigendecomposition",
    "D": "Diffused feature generation",
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

cmap_name = "Reds"
cmap = colormaps[cmap_name]
norm = Normalize(vmin=-20, vmax=-14)
sm = ScalarMappable(norm=norm, cmap=cmap)


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
