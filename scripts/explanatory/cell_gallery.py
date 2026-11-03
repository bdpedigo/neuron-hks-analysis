# %%
"""Fig 3A: the deployment predictions on the mesh of one cell of each column type.

One wide view per cell, with a circular zoom inset on a region of putative spines.
Each vertex takes the `pred_label` of its domain in the deployment run; a vertex in no
domain is `unknown`. The composite with Fig 3B is in
scripts/composites/cell_gallery.py.

Ported from meshrep/clean_paper_code/show_posterior_on_mesh.py. The 11 example cells
are the published ones, remapped from CAVE v1412 to v1822 by nucleus id; four of them
changed root id. ZOOM_CAMERAS holds the camera states of the published figure, captured
by hand against the v1412 meshes.

Despite the old name, the published panels colour each vertex by the argmax label, not
by a blend of the posteriors. This keeps that.

    uv run python scripts/explanatory/cell_gallery.py [--only CELL_TYPE ...]
"""

import argparse

import numpy as np
import pandas as pd
import pyvista as pv
from caveclient import CAVEclient

from analysis import (
    FIG_PATH,
    make_composite_figure,
    mask_dendrite_by_client_skeleton,
    plot_zoom_indicator,
    render,
    save_matplotlib_figure,
    set_pyvista_theme,
)
from analysis.colors import COMPARTMENT_PALETTE_MUTED
from analysis.products.roots import NO_DOMAIN, deployment_posteriors, load_root_cell
from analysis.sources import DATASTACK, VERSION

SUBFOLDER = "cell_gallery"
FIGURE_PATH = FIG_PATH / SUBFOLDER
INTERACTIVE = False

# The published order: excitatory by layer, then inhibitory.
ROOTS = {
    "23P": 864691136311267930,
    "4P": 864691135942250753,
    "5P-IT": 864691136108584921,
    "5P-ET": 864691136022520056,
    "5P-NP": 864691135014600950,
    "6P-CT": 864691136579193108,
    "6P-IT": 864691135662408944,
    "BC": 864691135946994785,
    "BPC": 864691135948215076,
    "MC": 864691136238652476,
    "NGC": 864691135518510218,
}

# The camera of the zoom inset, as (position, focal point, view up) in nm. The focal
# point also places the zoom indicator on the wide view. Captured interactively against
# the v1412 meshes, so check each one after a mesh changes.
ZOOM_CAMERAS = {
    "23P": [
        (706957.4547530242, 497003.330468549, 935707.9602166384),
        (694793.0766407662, 489576.2471957305, 924908.0262976695),
        (0.09585137130336815, -0.867296611561448, 0.4884763066856924),
    ],
    "4P": [
        (756100.4852379228, 522405.54894267436, 864629.8813648089),
        (740982.9282843418, 520951.87623179913, 870141.7569414034),
        (0.06696933819504837, -0.9946506950357398, -0.0786454233038203),
    ],
    "5P-IT": [
        (687492.6062115746, 732327.1416876956, 844533.7082207475),
        (682690.824536613, 728496.6487777109, 828027.7767359439),
        (-0.46727483301179323, -0.8215767275063866, 0.3265974758838436),
    ],
    "5P-ET": [
        (737504.6619938738, 741970.3761488856, 891419.6474784609),
        (713013.0, 747684.0, 879123.0),
        (-0.1170052521253632, -0.9690805371122859, -0.21723877086110474),
    ],
    "5P-NP": [
        (705074.496807942, 798064.1409513036, 932117.3039965727),
        (699291.6239528132, 793564.0639027812, 921556.3138441102),
        (0.26509425369252854, -0.930850253328277, 0.25148129659655316),
    ],
    "6P-CT": [
        (760145.339601375, 870932.906041457, 891458.7260250819),
        (750377.8586699424, 874752.4523663872, 881691.187787929),
        (-0.24568678849891185, -0.9605990171776393, -0.1299520302051666),
    ],
    "6P-IT": [
        (660630.0142616393, 789503.5056722766, 850340.5642668084),
        (650049.114108209, 796526.0421727186, 855421.4169481356),
        (-0.6329113814677552, -0.5965848683583552, -0.4934669979384862),
    ],
    "BC": [
        (744954.2963798414, 939757.427596519, 930309.191278093),
        (734107.6468943651, 948874.0342239509, 919130.404286086),
        (-0.12779613531828923, -0.8257264128049021, -0.5494033481824242),
    ],
    "BPC": [
        (762882.9704709331, 466451.2279330103, 868807.7543918795),
        (768727.8514781679, 461220.3936543533, 859844.4709080263),
        (0.8551010769795003, 0.07689981192464918, 0.5127266007088499),
    ],
    "MC": [
        (730334.1114414813, 856874.4559178757, 843334.6153535503),
        (733704.9927029627, 873327.1566922538, 843454.3290911687),
        (0.17037910812150017, -0.04207111887037336, 0.9844800559039869),
    ],
    "NGC": [
        (729568.6470767022, 412691.83411276806, 831296.3753259111),
        (733760.0950203188, 404412.2873600472, 822849.8665938233),
        (-0.7156745282646855, 0.28719096024990454, -0.6366563609545246),
    ],
}

ZOOM_RADIUS = 7500
ZOOM_SHIFT = 30000
CAMERA_DISTANCE = 300000
UPSHIFT = 30000
# NOTE: the published framing shifts the focal point right of the soma centroid.
RIGHT_SHIFT = 30000
PANEL_SIZE = (3, 6)
INSET_BOUNDS = (0.4, 0.7, 0.6, 0.3)


def vertex_colors(cell) -> tuple[np.ndarray, np.ndarray]:
    """The argmax label per vertex, and its colour."""
    vertex_domain = np.asarray(cell.mesh.map_index_to_layer("domains"))
    domain_label = (
        deployment_posteriors(cell).to_pandas().set_index("domain_id")["pred_label"]
    )
    labels = (
        pd.Series(vertex_domain)
        .map(domain_label)
        .where(vertex_domain != NO_DOMAIN)
        .fillna("unknown")
        .to_numpy()
    )
    colors = np.array(
        [COMPARTMENT_PALETTE_MUTED.get(label, [0, 0, 0]) for label in labels],
        dtype=np.uint8,
    )
    return labels, colors


def render_cell(cell_type: str, root_id: int, client) -> None:
    cell = load_root_cell(root_id, DATASTACK)
    labels, colors = vertex_colors(cell)

    # NOTE: the published panels show the soma and dendrite alone. The old script got
    # that by masking the morph to level 2 skeleton compartments 1 and 3, which
    # propagated to its mesh. Without this the axon dominates the frame.
    vertices, faces = cell.mesh.vertices, cell.mesh.faces
    (vertices, faces), keep = mask_dendrite_by_client_skeleton(
        (vertices, faces), root_id, client, distance_threshold=5, return_mask=True
    )
    labels, colors = labels[keep], colors[keep]
    share = pd.Series(labels).value_counts(normalize=True).round(3).to_dict()
    print(f"{cell_type} ({root_id}): {len(vertices):,} dendrite vertices, {share}")

    mesh = pv.make_tri_mesh(vertices, faces)
    zoom_cpos = ZOOM_CAMERAS[cell_type]

    # The wide view, framed on the centroid of the vertices predicted soma.
    center = vertices[labels == "soma"].mean(axis=0)
    center = np.array([center[0] - RIGHT_SHIFT, center[1] - UPSHIFT, center[2]])
    plotter = pv.Plotter(window_size=(400, 800), off_screen=not INTERACTIVE)
    plotter.add_mesh(mesh, scalars=colors, rgb=True, show_scalar_bar=False)
    plotter.camera.up = (0, -1, 0)
    plotter.camera.position = center + np.array(
        [CAMERA_DISTANCE, -CAMERA_DISTANCE / 5, CAMERA_DISTANCE]
    )
    plotter.camera.focal_point = center
    plot_zoom_indicator(
        plotter,
        zoom_cpos,
        plotter.camera_position,
        radius=ZOOM_RADIUS,
        shift=ZOOM_SHIFT,
    )
    render(plotter, cell_type, FIGURE_PATH, interactive=INTERACTIVE)

    # The zoom inset, at the hand-captured camera.
    plotter = pv.Plotter(window_size=(400, 400), off_screen=not INTERACTIVE)
    plotter.add_mesh(mesh, scalars=colors, rgb=True, show_scalar_bar=False)
    plotter.camera_position = zoom_cpos
    render(
        plotter,
        f"{cell_type}_zoom",
        FIGURE_PATH,
        circle_crop=True,
        interactive=INTERACTIVE,
    )

    fig, _, _ = make_composite_figure(
        cell_type,
        FIGURE_PATH,
        size=PANEL_SIZE,
        inset_bounds=INSET_BOUNDS,
        flip_inset=True,
        zoom=True,
    )
    save_matplotlib_figure(
        fig, f"composite_{cell_type}", FIGURE_PATH, formats=["png"], pad_inches=0
    )


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", default=[], metavar="CELL_TYPE")
    args = parser.parse_args()
    set_pyvista_theme()
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    chosen = args.only or list(ROOTS)
    unknown = [name for name in chosen if name not in ROOTS]
    if unknown:
        raise SystemExit(f"unknown cell type(s): {', '.join(unknown)}")
    # NOTE: only for the level 2 skeleton that masks the axon away.
    client = CAVEclient(DATASTACK, version=VERSION)
    for cell_type in chosen:
        render_cell(cell_type, ROOTS[cell_type], client)
