"""Interactive pyvista views: the shared conventions (`style`), the data contract
(`scene`), widgets, and the viewer. See `scripts/h01/view_retrain_models.py`."""

from .scene import Channel, Scene, add_prediction, cell_to_scene
from .style import (
    CLASS_ORDER,
    LABEL_PALETTE,
    categorical_kwargs,
    label_codes,
    new_plotter,
    posterior_rgb,
    scalar_kwargs,
)
from .panel import View, render_panel
from .viewer import CellOrder, Slider, Viewer

__all__ = [
    "CLASS_ORDER",
    "CellOrder",
    "Channel",
    "LABEL_PALETTE",
    "Scene",
    "Slider",
    "View",
    "Viewer",
    "add_prediction",
    "categorical_kwargs",
    "cell_to_scene",
    "label_codes",
    "new_plotter",
    "posterior_rgb",
    "render_panel",
    "scalar_kwargs",
]
