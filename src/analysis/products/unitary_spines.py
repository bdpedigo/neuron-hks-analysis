"""P13 inputs: the unitary-spine labels (S11) and the cells they are on.

The labels are points in 7 neuroglancer states, exported once to
`data/tables/unitary_spines/states/` by
`scripts/unitary_spines/export_unitary_states.py`. Each state holds the putative
multi-input spines of 10 cells of one excitatory type, one point per spine,
tagged by hand (`true_multi`, `y`, `merge`, ...).

A point's description is `{root}_{component_id}`: the v1412 root and the spine
id of the old pipeline. Neither carries over to the new pipeline, so the cells
map forward by nucleus ID, and the spines map forward through their synapse IDs.
"""

import json
from datetime import datetime

import pandas as pd
import polars as pl
from caveclient import CAVEclient
from nglui.parser import StateParser

from ..io import TABLE_PATH
from ..sources import DATASTACK, VERSION, cutoff, pin_params
from .catalog import Tables, write_parquet_atomic
from .labels import fetch_nucleus_roots
from .structures import _count_inputs, _root_synapses, read_structures, synapse_structure_join

UNITARY_TABLE_PATH = TABLE_PATH / "unitary_spines"
LABEL_VERSION = 1412
# The old pipeline's synapse -> spine map (meshrep/tools/pull_morphometry.py), a
# frozen local copy. `index` is the synapse ID.
OLD_COMPONENT_PATH = (
    TABLE_PATH.parents[2] / "meshrep" / "data" / "synapse_component_deltalake"
)

TAG_CATEGORIES = ["true_multi", "y", "not_spine", "merge", "tip", "overseg", "other", "review"]


def read_unitary_annotations() -> pl.DataFrame:
    """One row per labeled spine: the state, the old spine key, and its tags."""
    states = pd.read_csv(UNITARY_TABLE_PATH / "states.csv")
    frames = []
    for state in states.itertuples():
        state_json = json.loads(
            (UNITARY_TABLE_PATH / "states" / f"{state.state_id}.json").read_text()
        )
        df = StateParser(state_json).annotation_dataframe(expand_tags=True)
        df = df.query("layer == 'synapse_points'").copy()
        df["state_id"] = state.state_id
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    key = df["description"].str.split("_", expand=True).astype("int64")
    return pl.DataFrame(
        {
            "state_id": df["state_id"].astype("int64"),
            "anno_id": df["anno_id"],
            "label_root_id": key[0],
            "component_id": key[1],
            **{tag: df[tag].astype(bool) for tag in TAG_CATEGORIES},
        }
    )


Tables.register(
    "unitary_annotations",
    description=(
        "One row per hand-tagged putative multi-input spine in the exported "
        "neuroglancer states (S11): its state, the v1412 root and old `component_id`, "
        "and its tags."
    ),
    build=lambda: read_unitary_annotations().lazy(),
    frozen=UNITARY_TABLE_PATH,
)

Tables.register(
    "old_synapse_components",
    description=(
        "The old pipeline's map from synapse (`index`) to spine (`component_id`) "
        "on each v1412 `post_pt_root_id`. The old pipeline is gone, so this copy is "
        "the only link from the old spine IDs to synapse IDs."
    ),
    build=lambda: pl.scan_delta(str(OLD_COMPONENT_PATH)),
    frozen=OLD_COMPONENT_PATH,
)


def fetch_unitary_label_cells(client, version=VERSION) -> pl.DataFrame:
    """The label cells: v1412 root -> nucleus ID -> root ID at `version`."""
    label_roots = Tables["unitary_annotations"].collect()["label_root_id"].unique()
    cells = (
        Tables["v1412_cell_info"]
        .lazy()
        .select(
            pl.col("id").alias("nucleus_id"),
            pl.col("pt_root_id").alias("label_root_id"),
            "cell_type",
        )
        .filter(pl.col("label_root_id").is_in(label_roots.implode()))
        .collect()
    )
    assert cells["label_root_id"].is_unique().all(), "a label root has more than one nucleus"
    assert len(cells) == len(label_roots), "a label root has no nucleus at v1412"
    roots = fetch_nucleus_roots(client, cells["nucleus_id"].to_list(), version=version)
    return cells.join(roots, on="nucleus_id", how="left").sort("label_root_id")


Tables.register(
    "unitary_label_cells",
    description=(
        "One row per cell with single-spine labels: its v1412 root, its nucleus ID, "
        "its cell type and its root ID at the current version."
    ),
    materialize=lambda path: write_parquet_atomic(
        fetch_unitary_label_cells(CAVEclient(DATASTACK), version=VERSION), path
    ),
    cache_params=lambda: {"version": VERSION, "label_version": LABEL_VERSION},
)


def build_unitary_label_synapses() -> pl.DataFrame:
    """The synapse IDs on each labeled spine, from the old synapse -> spine map."""
    annotations = Tables["unitary_annotations"].collect()
    keys = annotations.select("label_root_id", "component_id").unique()
    return (
        Tables["old_synapse_components"].lazy()
        .filter(pl.col("post_pt_root_id").is_in(keys["label_root_id"].implode()))
        .select(
            pl.col("index").alias("synapse_id"),
            pl.col("post_pt_root_id").alias("label_root_id"),
            "component_id",
        )
        .collect()
        .join(keys, on=["label_root_id", "component_id"], how="semi")
        .sort("label_root_id", "component_id", "synapse_id")
    )


Tables.register(
    "unitary_label_synapses",
    description=(
        "One row per synapse on a labeled spine, from the synapse-to-spine map of the "
        "old pipeline. The key is the v1412 root and the old `component_id`."
    ),
    materialize=lambda path: write_parquet_atomic(build_unitary_label_synapses(), path),
    cache_params=lambda: {"label_version": LABEL_VERSION},
)


# How a label's spine carries over to the new structures:
# - kept: its mapped synapses are on one structure, which holds no other label.
# - split: its mapped synapses are on more than one structure.
# - merged: its mapped synapses are on one structure, which another label shares.
# - lost: none of its synapses is on a structure.
# - root_not_done: the deployment has no structures for its cell yet.
OUTCOMES = ["kept", "split", "merged", "lost", "root_not_done"]
STRUCTURE_PRODUCTS = (
    "synapse_mapping",
    "vertex_structures",
    "structure_morphometry",
    "structure_features",
)
UNITARY_CLASSES = ("not_unitary", "unitary")


def build_unitary_label_outcomes(as_of: datetime) -> pl.DataFrame:
    """One row per labeled spine: its outcome, its counts, and the structure it
    maps to when it is kept or merged."""
    label_key = ["label_root_id", "component_id"]
    cells = Tables["unitary_label_cells"].collect().select("label_root_id", "root_id")
    synapses = (
        Tables["unitary_label_synapses"].collect().join(cells, on="label_root_id")
    )
    done_roots = (
        read_structures(cells["root_id"].to_list(), as_of=as_of)["root_id"].unique()
    )
    mapped = synapse_structure_join(synapses, as_of=as_of)
    per_label = (
        mapped.group_by(label_key)
        .agg(
            pl.col("root_id").first(),
            pl.len().alias("n_synapses"),
            pl.col("structure_id").is_not_null().sum().alias("n_on_structure"),
            pl.col("structure_id").drop_nulls().n_unique().alias("n_structures"),
            pl.col("structure_id").drop_nulls().first().alias("structure_id"),
        )
    )
    n_labels_on_structure = (
        mapped.drop_nulls("structure_id")
        .group_by("root_id", "structure_id")
        .agg(pl.struct(label_key).n_unique().alias("n_labels_on_structure"))
    )
    outcomes = (
        Tables["unitary_annotations"].collect()
        .join(per_label, on=label_key, how="left")
        .join(cells, on="label_root_id", how="left", suffix="_cell")
        .with_columns(
            pl.col("root_id").fill_null(pl.col("root_id_cell")),
            pl.col("n_synapses", "n_on_structure", "n_structures").fill_null(0),
        )
        .drop("root_id_cell")
        .join(n_labels_on_structure, on=["root_id", "structure_id"], how="left")
        .with_columns(
            outcome=pl.when(~pl.col("root_id").is_in(done_roots.implode()))
            .then(pl.lit("root_not_done"))
            .when(pl.col("n_structures") == 0)
            .then(pl.lit("lost"))
            .when(pl.col("n_structures") > 1)
            .then(pl.lit("split"))
            .when(pl.col("n_labels_on_structure") > 1)
            .then(pl.lit("merged"))
            .otherwise(pl.lit("kept"))
            .cast(pl.Enum(OUTCOMES))
        )
        .with_columns(
            # A split or lost label has no one structure.
            pl.when(pl.col("outcome").is_in(["kept", "merged"]))
            .then(pl.col("structure_id"))
            .alias("structure_id")
        )
    )
    assert len(outcomes) == len(Tables["unitary_annotations"].collect())
    return outcomes.sort(label_key)


Tables.register(
    "unitary_label_outcomes",
    description=(
        "One row per labeled spine from the neuroglancer states, with its hand tags "
        "and how it carries over to the new spine structures: `kept`, `split`, "
        "`merged`, `lost` or `root_not_done`. A kept or merged spine has the "
        "`structure_id` of its new structure."
    ),
    materialize=lambda path: write_parquet_atomic(
        build_unitary_label_outcomes(cutoff("unitary_training")), path
    ),
    cache_params=lambda: {
        "label_version": LABEL_VERSION,
        "unitary_label_cells": Tables["unitary_label_cells"].cache_key,
        **pin_params(*STRUCTURE_PRODUCTS, as_of=cutoff("unitary_training")),
    },
)


def build_unitary_training_spines(as_of: datetime) -> pl.DataFrame:
    """One row per new structure that carries a usable label, with its morphometry
    and spectral means.

    A kept label gives its structure that label. The labels of a merged structure
    give it their label when every label on it is merged (none is also split) and
    they agree. Only structures with at least two presynaptic roots are kept, as
    the labeled spines were sampled from those.
    """
    outcomes = Tables["unitary_label_outcomes"].collect()
    structure_key = ["root_id", "structure_id"]
    on_structure = outcomes.filter(pl.col("structure_id").is_not_null())
    # NOTE: a structure that a split label also reaches is left out, because
    # that label's spine is only partly on it.
    split_reached = (
        Tables["unitary_label_synapses"]
        .collect()
        .join(
            outcomes.filter(pl.col("outcome") == "split").select(
                "label_root_id", "component_id", "root_id"
            ),
            on=["label_root_id", "component_id"],
        )
    )
    split_structures = (
        synapse_structure_join(split_reached, as_of=as_of)
        .drop_nulls("structure_id")
        .select(structure_key)
        .unique()
    )
    labeled = (
        on_structure.join(split_structures, on=structure_key, how="anti")
        .group_by(structure_key)
        .agg(
            pl.len().alias("n_labels"),
            pl.col("true_multi").all().alias("all_true_multi"),
            pl.col("true_multi").any().alias("any_true_multi"),
            pl.col("component_id").alias("component_ids"),
            pl.col("label_root_id").first(),
        )
        .filter(pl.col("all_true_multi") == pl.col("any_true_multi"))
        .with_columns(
            pl.when(pl.col("all_true_multi"))
            .then(pl.lit(UNITARY_CLASSES[1]))
            .otherwise(pl.lit(UNITARY_CLASSES[0]))
            .alias("label")
        )
        .drop("all_true_multi", "any_true_multi")
    )
    root_ids = labeled["root_id"].unique().to_list()
    cells = Tables["unitary_label_cells"].collect().select("root_id", "cell_type")
    # NOTE: this cutoff is before the unitary predictions, so the counts come
    # without `spine_structures`, which reads them.
    pre_roots = _count_inputs(_root_synapses(root_ids, as_of=as_of)).select(
        *structure_key, "n_pre_root_ids"
    )
    return (
        labeled.join(pre_roots, on=structure_key, how="left")
        .filter(pl.col("n_pre_root_ids") >= 2)
        .join(read_structures(root_ids, as_of=as_of), on=structure_key, how="inner")
        .join(cells, on="root_id", how="left")
        .sort(structure_key)
    )


Tables.register(
    "unitary_training_spines",
    description=(
        "One row per new spine structure with a usable single-spine label, with its "
        "morphometry and spectral means. Only structures with at least two "
        "presynaptic roots are kept. These are the training rows of the single-spine "
        "model."
    ),
    materialize=lambda path: write_parquet_atomic(
        build_unitary_training_spines(cutoff("unitary_training")), path
    ),
    cache_params=lambda: {"unitary_label_outcomes": Tables["unitary_label_outcomes"].cache_key},
)
