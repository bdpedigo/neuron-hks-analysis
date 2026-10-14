"""The contract between data and the viewer: a `Scene` of plain numpy arrays.

A viewer gets a `Scene` and does not know where its data comes from. A `Scene`
pickles, so a worker process can build the next one while a viewer shows the
current one. `cell_to_scene` is the only code that knows both an ossify cell from
`products.roots` and the viewer's types.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import polars as pl

from .style import CLASS_ORDER, label_codes, posterior_rgb

Basis = Literal["domains", "vertices", "synapses"]
Kind = Literal["categorical", "scalar", "rgb"]
Target = Literal["mesh", "synapses"]


@dataclass
class Channel:
    """Values for each element of `basis`.

    `categorical`: float codes into `categories` (see `style.label_codes`), NaN for
    none. `scalar`: floats. `rgb`: (n, 3) floats in [0, 1], NaN rows for none.
    `note` is a short text the viewer shows with the channel, for example a score.
    """

    name: str
    basis: Basis
    kind: Kind
    values: np.ndarray
    categories: Optional[tuple[str, ...]] = None
    symmetric: bool = False
    note: str = ""


@dataclass
class Scene:
    """A mesh, its synapses, and channels to color them by.

    `vertex_domain` and `synapse_domain` give each vertex's and synapse's row in
    the domain basis, and `synapse_vertex` each synapse's vertex; -1 for none.
    """

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    synapse_positions: np.ndarray
    vertex_domain: np.ndarray
    synapse_vertex: np.ndarray
    channels: dict[str, Channel] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def synapse_domain(self) -> np.ndarray:
        return np.where(self.synapse_vertex >= 0, self.vertex_domain[self.synapse_vertex], -1)

    def add(self, channel: Channel) -> None:
        self.channels[channel.name] = channel

    def values_on(self, target: Target, name: str) -> np.ndarray:
        """The values of channel `name` for each vertex (`mesh`) or synapse."""
        channel = self.channels[name]
        if channel.basis == {"mesh": "vertices", "synapses": "synapses"}[target]:
            return channel.values
        if (channel.basis, target) == ("domains", "mesh"):
            index = self.vertex_domain
        elif (channel.basis, target) == ("domains", "synapses"):
            index = self.synapse_domain
        elif (channel.basis, target) == ("vertices", "synapses"):
            index = self.synapse_vertex
        else:
            raise ValueError(f"channel {name!r} on {channel.basis} cannot show on the {target}")
        return _take(channel.values, index)


def _aligned(table: pl.DataFrame, domain_ids: np.ndarray) -> pl.DataFrame:
    """The rows of `table` in the order of `domain_ids`, null for a missing domain."""
    order = pl.DataFrame({"domain_id": domain_ids}).cast({"domain_id": table["domain_id"].dtype})
    return order.join(table, on="domain_id", how="left", maintain_order="left")


def _agreement(labels: np.ndarray, predicted: np.ndarray, categories) -> str:
    """How many labeled synapses a prediction gets right, as a note."""
    labels = np.asarray(labels, dtype=object)
    if "not_spine" in categories:
        # A binary model: soma and shaft are not_spine.
        labels = np.where(np.isin(labels, ["soma", "shaft"]), "not_spine", labels)
    true = label_codes(labels, CLASS_ORDER)
    scored = np.isfinite(true) & np.isfinite(predicted)
    if not scored.any():
        return ""
    return f"{int((true[scored] == predicted[scored]).sum())}/{int(scored.sum())} labels right"


def cell_to_scene(
    cell,
    posteriors: dict[str, pl.DataFrame],
    feature_columns: list[str] = (),
) -> Scene:
    """A scene of `cell` (from `products.roots.load_root_cell`).

    `posteriors` maps a model name to its `predict` output for the domains
    (`domain_id`, `posterior_<class>`, `pred_label`). For each model this makes the
    channels `{model}/pred_label`, `{model}/blend` and `{model}/p({class})`. It also
    makes `label` (the manual labels, on the synapses) and `feature/{column}` for
    each of `feature_columns`.
    """
    from ..products.roots import NO_DOMAIN

    domains = cell.annotations["domains"]
    domain_index = domains.vertex_index
    real = domain_index != NO_DOMAIN
    # The positions of the real domains, then -1 for the NO_DOMAIN row.
    position = np.full(len(domain_index), -1, dtype=np.int64)
    position[real] = np.arange(real.sum())
    vertex_domain = position[
        cell.mesh.map_index_to_layer("domains", as_positional=True, missing=-1)
    ]
    synapses = cell.annotations["synapses"]
    synapse_vertex = synapses.map_index_to_layer("mesh", as_positional=True, missing=-1)
    scene = Scene(
        name=str(cell.name),
        vertices=np.asarray(cell.mesh.vertices, dtype=np.float32),
        faces=np.asarray(cell.mesh.faces_positional, dtype=np.int64),
        synapse_positions=np.asarray(synapses.vertices, dtype=np.float32),
        vertex_domain=vertex_domain.astype(np.int64),
        synapse_vertex=np.asarray(synapse_vertex, dtype=np.int64),
        meta=dict(cell.meta),
    )

    labels = synapses.features["label"].to_numpy(dtype=object)
    scene.add(
        Channel("label", "synapses", "categorical", label_codes(labels), CLASS_ORDER)
    )

    domain_ids = domain_index[real]
    features = domains.features[real]
    for column in feature_columns:
        scene.add(
            Channel(
                f"feature/{column}",
                "domains",
                "scalar",
                features[column].to_numpy(dtype=np.float32),
                # NOTE: the curvature and normal columns center on zero.
                symmetric=column.startswith(("curvature_", "normal_")),
            )
        )

    for model, table in posteriors.items():
        table = _aligned(table, domain_ids)
        classes = [c.removeprefix("posterior_") for c in table.columns if c.startswith("posterior_")]
        classes = [c for c in classes if c in CLASS_ORDER]
        probabilities = table.select(f"posterior_{c}" for c in classes).to_numpy().astype(np.float32)
        predicted = label_codes(table["pred_label"].to_numpy())
        scene.add(
            Channel(
                f"{model}/pred_label",
                "domains",
                "categorical",
                predicted,
                CLASS_ORDER,
                note=_agreement(labels, _on_synapses(scene, predicted), classes),
            )
        )
        scene.add(Channel(f"{model}/blend", "domains", "rgb", posterior_rgb(probabilities, classes)))
        for i, name in enumerate(classes):
            scene.add(Channel(f"{model}/p({name})", "domains", "scalar", probabilities[:, i]))
    return scene


def _take(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """`values[index]`, NaN where `index` is -1."""
    # NOTE: one NaN row at the end, so that the -1 of "none" reads it.
    values = np.asarray(values, dtype=np.float32)
    padded = np.concatenate([values, np.full((1, *values.shape[1:]), np.nan, values.dtype)])
    return padded[index]


def _on_synapses(scene: Scene, domain_values: np.ndarray) -> np.ndarray:
    return _take(domain_values, scene.synapse_domain)
