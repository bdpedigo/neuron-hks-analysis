import pickle
import warnings

import numpy as np
import polars as pl
import pytest
import pyvista as pv
from ossify import load_cell, save_cell

from analysis.products import root_cache
from analysis.products.roots import NO_DOMAIN, build_root_cell, deployment_posteriors, domain_frame
from analysis.viz import CLASS_ORDER, CellOrder, Slider, Viewer, cell_to_scene, label_codes
from analysis.viz.style import LABEL_PALETTE, categorical_kwargs, posterior_rgb

ROOT_ID = 864691135000000001
CLASSES = ["shaft", "soma", "spine"]


def fake_cell(root_id=ROOT_ID):
    """A sphere in 4 domains by octant of x and y, with no domain for z > 0.8,
    3 mapped synapses (2 labeled) and 1 unmapped synapse."""
    sphere = pv.Sphere(theta_resolution=12, phi_resolution=12)
    vertices = np.asarray(sphere.points, dtype=np.float64)
    faces = np.asarray(sphere.regular_faces)
    domain = (vertices[:, 0] > 0).astype(int) * 2 + (vertices[:, 1] > 0).astype(int)
    domain_id = pl.Series(domain, dtype=pl.Int32).scatter(
        np.flatnonzero(vertices[:, 2] > 0.4), None
    )
    vertex_domains = pl.DataFrame(
        {"mesh_index": np.arange(len(vertices), dtype=np.int32)[::-1], "domain_id": domain_id[::-1]}
    )
    domain_ids = np.arange(4, dtype=np.int32)
    domain_features = pl.DataFrame(
        {
            "domain_id": domain_ids,
            "hks_0": [0.0, 1.0, 2.0, 3.0],
            "curvature_0": [-1.0, 0.0, 1.0, 0.5],
            "domain_x": [-1.0, -1.0, 1.0, 1.0],
            "domain_y": [-1.0, 1.0, -1.0, 1.0],
            "domain_z": [0.0, 0.0, 0.0, 0.0],
            "domain_n_vertices": [10, 10, 10, 10],
        }
    )
    posteriors = np.eye(3, dtype=np.float32)[[0, 1, 2, 2]]
    domain_predictions = pl.DataFrame(
        {
            "domain_id": domain_ids,
            **{f"posterior_{c}": posteriors[:, i] for i, c in enumerate(CLASSES)},
            "pred_label": [CLASSES[i] for i in posteriors.argmax(axis=1)],
            "posterior_entropy": np.zeros(4, dtype=np.float32),
        }
    )
    # Synapse vertices: one in each of domains 0 and 3, one in no domain.
    in_domain = np.flatnonzero(vertices[:, 2] <= 0.4)
    v0 = in_domain[domain[in_domain] == 0][0]
    v3 = in_domain[domain[in_domain] == 3][0]
    v_none = np.flatnonzero(vertices[:, 2] > 0.4)[0]
    synapse_ids = [11, 12, 13, 14]
    synapse_mapping = pl.DataFrame(
        {"synapse_id": synapse_ids[:3], "mesh_index": np.array([v0, v3, v_none], dtype=np.int32)}
    )
    positions = np.vstack([vertices[[v0, v3, v_none]], [[5.0, 5.0, 5.0]]])
    synapses = pl.DataFrame(
        {"synapse_id": synapse_ids, "x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2]}
    )
    labels = pl.DataFrame({"synapse_id": [11, 12, 14], "label": ["shaft", "shaft", "spine"]})
    nuclei = pl.DataFrame(
        {"root_id": [root_id], "nucleus_id": [1], "nucleus_x": [0.0], "nucleus_y": [0.0], "nucleus_z": [0.0]}
    )
    tables = {
        "vertex_domains": vertex_domains,
        "domain_features": domain_features,
        "domain_predictions": domain_predictions,
        "synapse_mapping": synapse_mapping,
    }
    cell = build_root_cell(root_id, (vertices, faces), tables, synapses, nuclei, labels)
    return cell, dict(vertices=vertices, domain_id=domain_id.to_numpy(), v0=v0, v3=v3, v_none=v_none)


def test_label_codes_are_fixed():
    assert label_codes(["spine"]).tolist() == [CLASS_ORDER.index("spine")]
    assert label_codes(["soma", "spine"]).tolist() == label_codes(["soma", "shaft", "spine"])[[0, 2]].tolist()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        codes = label_codes([None, np.nan, "axon"])
    assert np.isnan(codes).all()
    assert len(caught) == 1 and "axon" in str(caught[0].message)


def test_categorical_kwargs():
    kwargs = categorical_kwargs()
    assert kwargs["interpolate_before_map"] is False
    assert kwargs["clim"] == (-0.5, len(CLASS_ORDER) - 0.5)
    assert kwargs["cmap"] == [LABEL_PALETTE[c] for c in CLASS_ORDER]


def test_posterior_rgb_one_hot_is_class_color():
    rgb = posterior_rgb(np.array([[0.0, 1.0, 0.0], [np.nan, np.nan, np.nan]]), CLASSES)
    assert np.allclose(rgb[0], pv.Color(LABEL_PALETTE["soma"]).float_rgb)
    assert np.isnan(rgb[1]).all()


def test_cell_layers():
    cell, _ = fake_cell()
    assert NO_DOMAIN in cell.annotations["domains"].vertex_index
    assert len(cell.annotations["synapses"].vertices) == 3
    assert len(cell.annotations["unmapped_synapses"].vertices) == 1
    frame = domain_frame(cell)
    assert frame["domain_id"].to_list() == [0, 1, 2, 3]
    # The nucleus is at the origin, so the distance is the norm of the centroid.
    assert np.allclose(frame["distance_to_nucleus"], np.sqrt(2))
    assert deployment_posteriors(cell).columns == [
        "domain_id", "posterior_shaft", "posterior_soma", "posterior_spine", "pred_label"
    ]


def test_cell_to_scene():
    cell, truth = fake_cell()
    scene = cell_to_scene(cell, {"deployment": deployment_posteriors(cell)}, ["hks_0"])

    pred = scene.values_on("mesh", "deployment/pred_label")
    expected = np.array([CLASS_ORDER.index(c) for c in ["shaft", "soma", "spine", "spine"]], dtype=float)
    has_domain = ~np.isnan(truth["domain_id"].astype(float))
    assert np.isnan(pred[~has_domain]).all()
    assert np.array_equal(pred[has_domain], expected[truth["domain_id"][has_domain].astype(int)])

    # Synapses read the domain of their vertex: domain 0 (shaft), domain 3
    # (spine), and no domain.
    on_synapses = scene.values_on("synapses", "deployment/pred_label")
    assert on_synapses[:2].tolist() == [CLASS_ORDER.index("shaft"), CLASS_ORDER.index("spine")]
    assert np.isnan(on_synapses[2])
    assert scene.values_on("synapses", "label")[:2].tolist() == [CLASS_ORDER.index("shaft")] * 2
    # Synapse 11 (shaft) is right, synapse 12 (shaft, in a spine domain) is not.
    assert scene.channels["deployment/pred_label"].note == "1/2 labels right"

    hks = scene.values_on("mesh", "feature/hks_0")
    assert np.array_equal(hks[has_domain], truth["domain_id"][has_domain].astype(float))
    assert scene.values_on("mesh", "deployment/blend").shape == (len(truth["vertices"]), 3)


def test_scene_pickles():
    cell, _ = fake_cell()
    scene = cell_to_scene(cell, {"deployment": deployment_posteriors(cell)})
    again = pickle.loads(pickle.dumps(scene))
    assert np.array_equal(again.vertex_domain, scene.vertex_domain)
    assert again.channels.keys() == scene.channels.keys()


def test_cell_file_round_trip(tmp_path):
    cell, _ = fake_cell()
    path = tmp_path / f"{ROOT_ID}.osy"
    save_cell(cell, str(path))
    loaded = load_cell(str(path))
    posteriors = {"deployment": deployment_posteriors(cell)}
    scene, again = cell_to_scene(cell, posteriors), cell_to_scene(loaded, posteriors)
    assert np.array_equal(scene.vertex_domain, again.vertex_domain)
    assert np.array_equal(scene.synapse_vertex, again.synapse_vertex)
    assert np.array_equal(
        scene.channels["label"].values, again.channels["label"].values, equal_nan=True
    )


def test_cache_keeps_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(root_cache, "root_pins", lambda datastack: {"datastack": datastack})
    monkeypatch.setattr(root_cache, "check_not_held_out", lambda *args: None)
    calls = []

    def load(root_id, datastack):
        calls.append(root_id)
        return fake_cell(root_id)[0]

    cached = root_cache.cached(load, max_roots=2, folder=tmp_path)
    for root_id in [1, 2, 3, 3]:
        cached(root_id, "fake")
    assert calls == [1, 2, 3]
    files = sorted(p.name for p in tmp_path.rglob("*.osy"))
    assert files == ["2.osy", "3.osy"]
    with pytest.raises(ValueError):
        root_cache.cached(load, max_roots=1)


def test_cell_order_visits_every_root_once():
    order = CellOrder([5, 3, 9, 1], start=2)
    seen = [order.current] + [order.step(1) for _ in range(3)]
    assert seen == [5, 9, 1, 3]
    assert order.position == "4 / 4"
    assert order.peek(1) == 5


def test_viewer_smoke(tmp_path):
    cells = {root_id: fake_cell(root_id)[0] for root_id in [1, 2]}

    def scene_for_root(root_id):
        cell = cells[root_id]
        return cell_to_scene(cell, {"deployment": deployment_posteriors(cell)}, ["hks_0"])

    def resolve(state):
        if state["color"] == "feature":
            return state["feature"], "deployment/pred_label"
        return f"deployment/{state['color']}", "deployment/pred_label"

    viewer = Viewer(
        scene_for_root,
        CellOrder([1, 2], start=0),
        selectors={"color": ["pred_label", "blend", "p(spine)", "feature"]},
        resolve=resolve,
        sliders={"feature": Slider(["feature/hks_0"], visible=lambda s: s["color"] == "feature")},
        prefetch=False,
        off_screen=True,
    )
    viewer.step(0)
    assert viewer.scene.name == "1"
    for color in ["blend", "p(spine)", "feature", "pred_label"]:
        viewer.select("color", color)
    assert "synapses" in viewer.plotter.actors
    viewer._setter("synapses")(False)
    assert "synapses" not in viewer.plotter.actors
    viewer._setter("synapses")(True)
    viewer.step(1)
    assert viewer.scene.name == "2"
    viewer.plotter.screenshot(str(tmp_path / "view.png"))
    viewer.plotter.close()
    viewer.close()
    assert (tmp_path / "view.png").exists()


def test_viewer_smoothing():
    cell = fake_cell(1)[0]

    def scene_for_root(root_id):
        return cell_to_scene(cell, {"deployment": deployment_posteriors(cell)}, ["hks_0"])

    viewer = Viewer(
        scene_for_root,
        CellOrder([1], start=0),
        selectors={"color": ["pred_label"]},
        resolve=lambda state: ("deployment/pred_label", "deployment/pred_label"),
        prefetch=False,
        off_screen=True,
        smoothing=True,
    )
    viewer.step(0)
    raw = np.asarray(viewer.surface.points).copy()
    assert not viewer.state["smooth"]
    viewer.set_smooth(True)
    smoothed = np.asarray(viewer.surface.points)
    assert smoothed.shape == raw.shape
    assert not np.allclose(smoothed, raw)
    assert "synapses" in viewer.plotter.actors
    # each mapped synapse moved with its vertex
    on = viewer.scene.synapse_vertex >= 0
    shift = smoothed[viewer.scene.synapse_vertex[on]] - raw[viewer.scene.synapse_vertex[on]]
    assert np.allclose(viewer.synapse_positions[on] - viewer.scene.synapse_positions[on], shift)
    viewer.set_smooth(False)
    assert np.allclose(viewer.surface.points, raw)
    viewer.plotter.close()
    viewer.close()
