"""P5: the H01 labels (S12) and their cells, read at the labeling time (v1054).

The labels are points in 16 neuroglancer states, exported once to
`data/tables/h01/states/` by `scripts/h01/export_h01_states.py`. Two cohorts:
`validation` (the Fig 6 states) and `retrain`. The model never saw H01 data, so
both cohorts are held out.
"""

import json

import pandas as pd
import polars as pl
from nglui.parser import StateParser

from ..io import TABLE_PATH
from .catalog import Tables, write_parquet_atomic
from .sources import H01_DATASTACK, H01_INPUTS_PATH, H01_SYNAPSES_PATH, H01_VERSION

H01_TABLE_PATH = TABLE_PATH / "h01"
H01_FOLDER = f"{H01_DATASTACK}/v{H01_VERSION}"

TAG_CATEGORIES = ["spine", "shaft", "soma", "orphan", "error", "other", "odd_segmentation"]
LABEL_CATEGORIES = ["spine", "shaft", "soma"]


def read_h01_annotations() -> pl.DataFrame:
    """One row per labeled point: the state root, the point in nm, and its tags."""
    states = pd.read_csv(H01_TABLE_PATH / "states.csv")
    frames = []
    for state in states.itertuples():
        state_json = json.loads((H01_TABLE_PATH / "states" / f"{state.state_id}.json").read_text())
        df = StateParser(state_json).annotation_dataframe(expand_tags=True)
        df["state_id"] = state.state_id
        df["cohort"] = state.cohort
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    assert df["linked_segmentation"].map(len).eq(1).all(), "a point links to more than one root"
    # The annotation layer is in nm, as the synapse positions are.
    points = pd.DataFrame(df["point"].tolist(), columns=["x", "y", "z"]).astype("int64")
    return pl.DataFrame(
        {
            "state_id": df["state_id"].astype("int64"),
            "cohort": df["cohort"],
            "anno_id": df["anno_id"],
            "state_root_id": df["linked_segmentation"].map(lambda x: int(x[0])).astype("int64"),
            "x": points["x"],
            "y": points["y"],
            "z": points["z"],
            **{tag: df[tag].astype(bool) for tag in TAG_CATEGORIES},
        }
    )


def fetch_h01_candidate_roots() -> pl.DataFrame:
    """The roots to snapshot and run: at v1054, the state roots themselves."""
    roots = (
        read_h01_annotations()
        .group_by("state_root_id", "cohort", "state_id")
        .agg(n_points=pl.len())
        .rename({"state_root_id": "root_id"})
        .sort("root_id")
    )
    assert roots["root_id"].is_unique().all(), "a root is in more than one state"
    return roots


Tables.register(
    "h01_candidate_roots",
    materialize=lambda path: write_parquet_atomic(fetch_h01_candidate_roots(), path),
    cache_params={"datastack": H01_DATASTACK, "version": H01_VERSION},
    folder=H01_FOLDER,
)


def fetch_h01_labels(annotations: pl.DataFrame, root_ids: list[int]) -> pl.DataFrame:
    """The labeled points matched to synapses by position, with one tag each.

    The synapses come from the snapshot that the deployment job reads, so the
    labels and the predictions use the same rows.
    """
    synapses = (
        pl.scan_delta(str(H01_SYNAPSES_PATH))
        .filter(pl.col("post_pt_root_id").is_in(root_ids))
        .select(
            pl.col("id").alias("synapse_id"),
            pl.col("post_pt_root_id").alias("root_id"),
            *[pl.col(f"ctr_pt_position_{c}").round().cast(pl.Int64).alias(c) for c in "xyz"],
        )
        .collect()
    )
    # NOTE: a few points sit on two synapses of the same root at the same
    # position (2 of 4,000 at v1054), a duplicate detection. Keep the lowest
    # synapse id, so each labeled point counts once.
    matched = (
        annotations.join(
            synapses,
            left_on=["x", "y", "z", "state_root_id"],
            right_on=["x", "y", "z", "root_id"],
            how="left",
            coalesce=False,
        )
        .sort("synapse_id", nulls_last=True)
        .unique("anno_id", keep="first", maintain_order=True)
    )
    assert len(matched) == len(annotations)

    summary = matched.group_by("cohort").agg(
        n_points=pl.len(),
        n_matched=pl.col("synapse_id").is_not_null().sum(),
        n_on_state_root=(pl.col("root_id") == pl.col("state_root_id")).sum(),
    )
    print(summary.sort("cohort"))

    # NOTE: the same synapse can be labeled twice; combine its tags, as the old
    # script did, then keep only synapses with exactly one tag.
    labels = (
        matched.drop_nulls("synapse_id")
        .group_by("synapse_id", "root_id", "state_root_id", "cohort", "state_id", "x", "y", "z")
        .agg(pl.col(TAG_CATEGORIES).any())
        .filter(pl.sum_horizontal(TAG_CATEGORIES) == 1)
        .with_columns(
            label=pl.coalesce(
                [pl.when(pl.col(tag)).then(pl.lit(tag)) for tag in TAG_CATEGORIES]
            )
        )
        .filter(pl.col("label").is_in(LABEL_CATEGORIES))
        .rename({"x": "ctr_pt_position_x", "y": "ctr_pt_position_y", "z": "ctr_pt_position_z"})
        .drop(TAG_CATEGORIES)
        .sort("root_id", "synapse_id")
    )
    assert labels["synapse_id"].is_unique().all()
    return labels


Tables.register(
    "h01_labels",
    materialize=lambda path: write_parquet_atomic(
        fetch_h01_labels(
            read_h01_annotations(),
            Tables["h01_candidate_roots"].collect()["root_id"].to_list(),
        ),
        path,
    ),
    cache_params={
        "inputs": str(H01_INPUTS_PATH),
        "h01_candidate_roots": Tables["h01_candidate_roots"].cache_key,
    },
    folder=H01_FOLDER,
)

Tables.register(
    "h01_cells",
    build=lambda: Tables["h01_labels"]
    .lazy()
    .group_by("root_id", "cohort")
    .agg(n_labels=pl.len())
    .sort("root_id"),
    cache_params={"h01_labels": Tables["h01_labels"].cache_key},
    folder=H01_FOLDER,
)
