"""An interactive view of one root at a time, with selectors that choose what
colors the mesh and the synapses, and keys that step through the roots.

The viewer knows only `Scene`s (`viz.scene`). It gets them from `scene_for_root`,
which by default runs in a worker process: while one root shows, the worker
builds the next one, so a key press finds it ready or nearly ready.
"""

import multiprocessing
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyvista as pv

from .scene import Scene
from .style import (
    FONT_SIZE,
    INK,
    MESH_COLOR,
    MESH_OPACITY,
    POINT_SIZE,
    WINDOW,
    categorical_kwargs,
    new_plotter,
    rgb_to_uint8,
    scalar_kwargs,
)
from .widgets import BUTTON_PITCH, ChannelSlider, RadioGroup, add_checkbox_row

LEFT = 12
GROUP_GAP = 12

# The "smooth mesh" option: pyvista's Taubin (VTK windowed sinc) smoothing, with the
# parameters of riff's `taubin` operation (from the postsyn-census-benchmarks
# Taubin study). Display only: the scene's vertices and channels do not change.
TAUBIN_KWARGS = dict(n_iter=20, pass_band=0.1, normalize_coordinates=True)

# resolve(state) -> (mesh channel, synapse channel); None draws the plain mesh or
# no synapses.
Resolve = Callable[[dict], tuple[Optional[str], Optional[str]]]


@dataclass
class Slider:
    """A slider over channel `names`. `visible(state)` says when it shows; by
    default always."""

    names: Sequence[str]
    visible: Optional[Callable[[dict], bool]] = None


class CellOrder:
    """The roots in sorted order, from a random start, with wraparound, so a
    session visits every root once before it repeats."""

    def __init__(self, roots: Sequence[int], start: Optional[int] = None, seed=None):
        self.roots = sorted({int(root) for root in roots})
        if not self.roots:
            raise ValueError("no roots to view")
        if start is None:
            start = int(np.random.default_rng(seed).integers(len(self.roots)))
        self._start = start % len(self.roots)
        self.index = self._start

    @property
    def current(self) -> int:
        return self.roots[self.index]

    def peek(self, step: int) -> int:
        return self.roots[(self.index + step) % len(self.roots)]

    def step(self, step: int) -> int:
        self.index = (self.index + step) % len(self.roots)
        return self.current

    @property
    def position(self) -> str:
        visited = (self.index - self._start) % len(self.roots) + 1
        return f"{visited} / {len(self.roots)}"


class _Ready(Future):
    """A future that runs its call on the first `result()`, in this process."""

    def __init__(self, fn, *args):
        super().__init__()
        self._call = (fn, args)

    def result(self, timeout=None):
        if not self.done():
            fn, args = self._call
            try:
                self.set_result(fn(*args))
            except Exception as error:
                self.set_exception(error)
        return super().result(timeout)


class Viewer:
    """Show the scene of each root in `order`, one at a time.

    - `selectors`: radio groups, name to options. Their values form the `state`.
    - `sliders`: name to a `Slider`. The picked channel name is `state[name]`.
    - `resolve(state)`: the mesh channel and the synapse channel to show. Every
      synapse shows, in the no-value color where the channel has no value.
    - A checkbox shows or hides the synapse points (`state["synapses"]`).
    - With `smoothing`, a checkbox (off by default) draws the mesh Taubin-smoothed
      (`TAUBIN_KWARGS`, `state["smooth"]`); each synapse point moves with its vertex.
    - Keys: `n` for the next root, `b` for the one before.

    With `prefetch`, `scene_for_root` runs in one worker process (spawned, so it
    must be importable: a module-level function). `worker_initializer` runs once
    in the worker, for example to load models.
    """

    def __init__(
        self,
        scene_for_root: Callable[[int], Scene],
        order: CellOrder,
        selectors: dict[str, Sequence],
        resolve: Resolve,
        *,
        sliders: Optional[dict[str, Slider]] = None,
        prefetch: bool = True,
        worker_initializer: Optional[Callable[[], None]] = None,
        keep_scenes: int = 2,
        off_screen: bool = False,
        smoothing: bool = False,
    ):
        self.scene_for_root = scene_for_root
        self.order = order
        self.resolve = resolve
        self.keep_scenes = keep_scenes
        self.scene: Optional[Scene] = None
        self._scenes: OrderedDict[int, Scene] = OrderedDict()
        self._futures: dict[int, Future] = {}
        self._executor = (
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=worker_initializer,
            )
            if prefetch
            else None
        )
        if not prefetch and worker_initializer is not None:
            worker_initializer()

        self.plotter = new_plotter(off_screen=off_screen)
        self.state: dict = {"synapses": True, "smooth": False}
        self._radios = {}
        top = WINDOW[1] - 40
        for name, options in selectors.items():
            group = RadioGroup(
                self.plotter, name, options, self._setter(name), x=LEFT, top=top
            )
            self._radios[name] = group
            self.state[name] = group.value
            top = group.bottom - GROUP_GAP - BUTTON_PITCH
        add_checkbox_row(
            self.plotter,
            self._setter("synapses"),
            value=True,
            x=LEFT,
            y=top,
            label="synapse points",
            name="checkbox:synapses",
        )
        if smoothing:
            top -= BUTTON_PITCH
            add_checkbox_row(
                self.plotter,
                self.set_smooth,
                value=False,
                x=LEFT,
                y=top,
                label="smooth mesh (Taubin)",
                name="checkbox:smooth",
            )
        self._sliders: dict[str, tuple[ChannelSlider, Slider]] = {}
        for i, (name, spec) in enumerate((sliders or {}).items()):
            y = 0.12 + 0.1 * i
            slider = ChannelSlider(
                self.plotter,
                spec.names,
                self._setter(name),
                pointa=(0.55, y),
                pointb=(0.95, y),
                label=lambda channel: channel.split("/", 1)[-1],
            )
            self._sliders[name] = (slider, spec)
            self.state[name] = slider.value
        self.plotter.add_key_event("n", lambda: self.step(1))
        self.plotter.add_key_event("b", lambda: self.step(-1))

    # -- state

    def _setter(self, name: str) -> Callable:
        def set_value(value) -> None:
            self.state[name] = value
            if self.scene is not None:
                self._paint()

        return set_value

    def set_smooth(self, value: bool) -> None:
        """Draw the mesh smoothed or not (as a click on the checkbox does)."""
        self.state["smooth"] = bool(value)
        if self.scene is not None:
            self._set_surface(self.scene)
            self._paint()

    def select(self, name: str, value) -> None:
        """Set a selector (as a click on it does)."""
        group = self._radios[name]
        group.value = value
        group._sync()
        self._setter(name)(value)

    # -- loading

    def _submit(self, root_id: int) -> Future:
        if root_id not in self._futures and root_id not in self._scenes:
            if self._executor is None:
                self._futures[root_id] = _Ready(self.scene_for_root, root_id)
            else:
                self._futures[root_id] = self._executor.submit(self.scene_for_root, root_id)
        return self._futures.get(root_id)

    def _get_scene(self, root_id: int) -> Scene:
        if root_id in self._scenes:
            self._scenes.move_to_end(root_id)
            return self._scenes[root_id]
        self._submit(root_id)
        future = self._futures.pop(root_id)
        scene = future.result()
        self._scenes[root_id] = scene
        while len(self._scenes) > self.keep_scenes:
            self._scenes.popitem(last=False)
        return scene

    def step(self, step: int) -> None:
        """Show the root `step` places on in the order, and start the one after."""
        root_id = self.order.step(step)
        self._title(f"loading {root_id} ({self.order.position})...")
        self.plotter.render()
        try:
            scene = self._get_scene(root_id)
        except Exception as error:
            print(f"root {root_id}: {error!r}")
            self._title(f"root {root_id} failed: {error}")
            self._submit(self.order.peek(step or 1))
            return
        self._show(scene)
        self._submit(self.order.peek(step or 1))

    # -- drawing

    def _title(self, text: str) -> None:
        self.plotter.add_text(
            text, position="upper_edge", font_size=FONT_SIZE + 2, color=INK, name="title"
        )

    def _smoothed_vertices(self, scene: Scene) -> np.ndarray:
        """The Taubin-smoothed vertices of `scene`, computed once per scene."""
        if "taubin_vertices" not in scene.meta:
            self._title(f"smoothing {scene.name}...")
            self.plotter.render()
            smoothed = pv.PolyData.from_regular_faces(
                np.asarray(scene.vertices, dtype=float), scene.faces
            ).smooth_taubin(**TAUBIN_KWARGS)
            if smoothed.n_points != len(scene.vertices):
                raise RuntimeError("Taubin smoothing changed the vertex count")
            scene.meta["taubin_vertices"] = np.asarray(smoothed.points)
        return scene.meta["taubin_vertices"]

    def _set_surface(self, scene: Scene) -> None:
        """The drawn surface and synapse points of `scene`, smoothed or not."""
        vertices = scene.vertices
        self.synapse_positions = scene.synapse_positions
        if self.state["smooth"]:
            vertices = self._smoothed_vertices(scene)
            # each synapse moves with its vertex; one with no vertex stays
            on = scene.synapse_vertex >= 0
            shift = np.zeros_like(scene.synapse_positions, dtype=float)
            shift[on] = vertices[scene.synapse_vertex[on]] - scene.vertices[scene.synapse_vertex[on]]
            self.synapse_positions = scene.synapse_positions + shift
        self.surface = pv.PolyData.from_regular_faces(vertices, scene.faces)
        self.base = self.plotter.add_mesh(
            self.surface, color=MESH_COLOR, opacity=MESH_OPACITY, name="base", reset_camera=False
        )

    def _show(self, scene: Scene) -> None:
        self.scene = scene
        self._set_surface(scene)
        self._paint()
        self.plotter.view_xy()
        self.plotter.camera.up = (0, -1, 0)
        self.plotter.reset_camera()

    def _paint(self) -> None:
        """Color the mesh and the synapses by the channels of the current state."""
        scene = self.scene
        mesh_name, synapse_name = self.resolve(self.state)
        for title in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title)
        for slider, spec in self._sliders.values():
            slider.set_visible(spec.visible is None or spec.visible(self.state))

        self.plotter.remove_actor("painted", render=False)
        if mesh_name is not None and mesh_name in scene.channels:
            surface = self.surface.copy(deep=False)
            kwargs = self._color_kwargs(scene, "mesh", mesh_name, surface, show_bar=True)
            self.plotter.add_mesh(
                surface, opacity=MESH_OPACITY, name="painted", reset_camera=False, **kwargs
            )
        self.base.SetVisibility(mesh_name is None or mesh_name not in scene.channels)

        self.plotter.remove_actor("synapses", render=False)
        if (
            self.state["synapses"]
            and synapse_name is not None
            and synapse_name in scene.channels
            and len(scene.synapse_positions)
        ):
            points = pv.PolyData(self.synapse_positions)
            kwargs = self._color_kwargs(scene, "synapses", synapse_name, points, show_bar=mesh_name is None)
            self.plotter.add_points(
                points,
                point_size=POINT_SIZE,
                render_points_as_spheres=True,
                name="synapses",
                reset_camera=False,
                **kwargs,
            )

        position = self.order.position
        lines = [f"{scene.name}   ({position})"]
        for target, name in (("mesh", mesh_name), ("synapses", synapse_name)):
            if name is not None and name in scene.channels:
                note = scene.channels[name].note
                lines.append(f"{target}: {name}" + (f"  ({note})" if note else ""))
        self._title("\n".join(lines))
        self.plotter.render()

    def _color_kwargs(self, scene, target, name, poly, show_bar) -> dict:
        channel = scene.channels[name]
        values = scene.values_on(target, name)
        if channel.kind == "rgb":
            poly.point_data["rgb"] = rgb_to_uint8(values)
            return {"scalars": "rgb", "rgb": True}
        poly.point_data["values"] = values
        if channel.kind == "categorical":
            return {"scalars": "values", **categorical_kwargs(channel.categories)}
        kwargs = scalar_kwargs(values, symmetric=channel.symmetric)
        title = name.split("/", 1)[-1]
        return {
            "scalars": "values",
            **kwargs,
            "show_scalar_bar": show_bar,
            "scalar_bar_args": {"title": title, "vertical": True, "position_x": 0.88, "position_y": 0.25},
        }

    # -- running

    def show(self, screenshot: Optional[str] = None) -> None:
        """Show the first root and start the interactive window, or with
        `screenshot`, write an image of the first root and close."""
        try:
            self.step(0)
            if screenshot is not None:
                self.plotter.screenshot(screenshot)
                self.plotter.close()
            else:
                self.plotter.show()
        finally:
            self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
