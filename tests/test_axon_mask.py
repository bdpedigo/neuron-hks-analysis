import numpy as np
import pyvista as pv
from ossify import Cell, Link

from analysis.axon_mask import axon_vertex_mask, mask_axon

LEG_NM = 20_000.0
BEND_NM = 800.0  # radius of the bend, so the legs are 1600 nm apart at the center
RADIUS_NM = 300.0  # so the two surfaces are 1000 nm apart
CHUNK_GRID = (np.zeros(3), np.full(3, 2000.0))


def hairpin(step=500.0):
    """Points along a U: a dendrite leg on y = 0, a bend, and an axon leg back on
    y = 1600, with the label of each point."""
    leg = np.arange(0.0, LEG_NM, step)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 9)[1:-1]
    dendrite = np.stack([leg, np.zeros_like(leg), np.zeros_like(leg)], axis=1)
    bend = np.stack(
        [LEG_NM + BEND_NM * np.cos(theta), BEND_NM + BEND_NM * np.sin(theta), np.zeros_like(theta)], axis=1
    )
    axon = np.stack([leg[::-1], np.full_like(leg, 2 * BEND_NM), np.zeros_like(leg)], axis=1)
    points = np.concatenate([dendrite, bend, axon])
    is_axon = np.concatenate([np.zeros(len(dendrite) + len(bend), bool), np.ones(len(axon), bool)])
    return points, is_axon


def fake_inputs(no_axon=False):
    """A tube mesh around the U, and a cell whose level2 points are the skeleton
    vertices."""
    points, is_axon = hairpin()
    if no_axon:
        is_axon[:] = False
    tube = pv.Spline(points, 400).tube(radius=RADIUS_NM, n_sides=16).triangulate().clean()
    vertices = np.asarray(tube.points, dtype=float)
    faces = np.asarray(tube.regular_faces)
    edges = np.stack([np.arange(1, len(points)), np.arange(len(points) - 1)], axis=1)
    cell = Cell(name=1, meta={"datastack": "unused"})
    cell.add_skeleton(points, edges, features={"is_axon": is_axon}, root=0)
    cell.add_graph(
        points,
        linkage=Link(mapping=np.arange(len(points)), target="skeleton", map_value_is_index=False),
    )
    return (vertices, faces), cell


def test_mask_axon_removes_axon_leg_and_keeps_dendrite_leg():
    mesh, cell = fake_inputs()
    vertices = mesh[0]
    (kept_vertices, kept_faces), remove = mask_axon(mesh, cell, chunk_grid=CHUNK_GRID)
    on_dendrite_leg = (vertices[:, 1] < BEND_NM) & (vertices[:, 0] < LEG_NM)
    # Far from the bend, more than the margin plus the bend length along the surface.
    on_axon_leg = (vertices[:, 1] > BEND_NM) & (vertices[:, 0] < LEG_NM - 5000.0)
    assert not remove[on_dendrite_leg].any()
    assert remove[on_axon_leg].all()
    assert len(kept_vertices) == (~remove).sum()
    assert kept_faces.max() < len(kept_vertices)


def test_domain_vote_keeps_a_domain_with_few_axon_vertices():
    mesh, cell = fake_inputs()
    vertices = mesh[0]
    remove = axon_vertex_mask(mesh, cell, chunk_grid=CHUNK_GRID)
    # One domain per 2 um of x on each leg.
    domain = (vertices[:, 0] // 2000).astype(int) * 2 + (vertices[:, 1] > BEND_NM)
    voted = axon_vertex_mask(mesh, cell, chunk_grid=CHUNK_GRID, vertex_domain=domain, domain_frac=0.9)
    for d in np.unique(domain):
        sel = domain == d
        assert voted[sel].all() == (remove[sel].mean() >= 0.9)
        assert voted[sel].all() or not voted[sel].any()


def test_no_axon_removes_nothing():
    mesh, cell = fake_inputs(no_axon=True)
    assert not axon_vertex_mask(mesh, cell, chunk_grid=CHUNK_GRID).any()
