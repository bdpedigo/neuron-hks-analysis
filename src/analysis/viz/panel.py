"""Static image panels of a `Scene`: one column per channel, one row per view.

A view is the whole cell, or a close-up of a sphere around a point (for example,
the soma around the nucleus). Every panel uses the camera of the viewer (the xy
plane, y down), so a panel and the viewer show a cell the same way.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyvista as pv

from .scene import Scene
from .style import (
    FONT_SIZE,
    INK,
    categorical_kwargs,
    new_plotter,
    rgb_to_uint8,
    scalar_kwargs,
)

PANEL_SIZE = 600


@dataclass
class View:
    """The whole cell (`center` None), or the part of it within `radius` (nm) of
    `center`, cropped so that the far side does not hide the near side."""

    name: str
    center: Optional[np.ndarray] = None
    radius: Optional[float] = None


def _mesh_kwargs(scene: Scene, name: str, surface: pv.PolyData) -> dict:
    channel = scene.channels[name]
    values = scene.values_on("mesh", name)
    if channel.kind == "rgb":
        surface.point_data["rgb"] = rgb_to_uint8(values)
        return {"scalars": "rgb", "rgb": True}
    surface.point_data["values"] = values
    if channel.kind == "categorical":
        return {"scalars": "values", **categorical_kwargs(channel.categories)}
    return {"scalars": "values", **scalar_kwargs(values, symmetric=channel.symmetric), "show_scalar_bar": False}


def render_panel(
    scene: Scene,
    channels: Sequence[str],
    views: Sequence[View],
    path: str,
    titles: Optional[Sequence[str]] = None,
) -> None:
    """Write an image of `channels` (columns) in `views` (rows) to `path`."""
    titles = list(titles or channels)
    plotter = new_plotter(
        off_screen=True,
        shape=(len(views), len(channels)),
        window_size=(PANEL_SIZE * len(channels), PANEL_SIZE * len(views)),
        border=False,
    )
    surface = pv.PolyData.from_regular_faces(scene.vertices, scene.faces)
    for row, view in enumerate(views):
        for col, name in enumerate(channels):
            plotter.subplot(row, col)
            mesh = surface.copy(deep=False)
            kwargs = _mesh_kwargs(scene, name, mesh)
            if view.center is not None:
                keep = np.linalg.norm(scene.vertices - view.center, axis=1) <= view.radius
                mesh = mesh.extract_points(keep, adjacent_cells=False)
            plotter.add_mesh(mesh, **kwargs)
            note = scene.channels[name].note
            label = f"{titles[col]} ({view.name})" + (f"\n{note}" if note else "")
            plotter.add_text(label, position="upper_left", font_size=FONT_SIZE - 2, color=INK)
            plotter.view_xy()
            plotter.camera.up = (0, -1, 0)
            plotter.reset_camera()
        # NOTE: one camera per row, so the columns compare the same place.
        plotter.link_views([row * len(channels) + col for col in range(len(channels))])
    plotter.screenshot(path)
    plotter.close()
