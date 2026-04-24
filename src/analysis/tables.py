from pathlib import Path

import numpy as np
import polars as pl
from standard_transform.datasets import minnie_transform_vx

from .io import (
    BROAD_TYPE_CATEGORIES,
    CELL_TYPE_CATEGORIES,
    COMPARTMENT_CATEGORIES,
    DATA_PATH,
    LABEL_CATEGORIES,
    LABEL_DETAILED_CATEGORIES,
    MTYPE_CATEGORIES,
    load_neuron_info,
)

BROAD_TYPE_ENUM = pl.Enum(categories=BROAD_TYPE_CATEGORIES)
CELL_TYPE_ENUM = pl.Enum(categories=CELL_TYPE_CATEGORIES)
MTYPE_ENUM = pl.Enum(categories=MTYPE_CATEGORIES)
LABEL_ENUM = pl.Enum(categories=LABEL_CATEGORIES)
COMPARTMENT_CATEGORIES_ENUM = pl.Enum(categories=COMPARTMENT_CATEGORIES)


synapse_path = DATA_PATH / "synapses_pni_2_v1412_deltalake"
pre_synapse_feature_path = DATA_PATH / "pre_synapse_skeleton_features"
post_synapse_feature_path = DATA_PATH / "post_synapse_skeleton_features"
pre_synapse_skeleton_path = DATA_PATH / "pre_synapse_skeleton_features"
post_synapse_skeleton_path = DATA_PATH / "post_synapse_skeleton_features"
prediction_path = DATA_PATH / "auburn_elk_detour_predictions_deltalake"
spine_morphometry_column_path = DATA_PATH / "spine_morphometry_features"
spine_component_path = DATA_PATH / "synapse_component_deltalake"
spine_morphometry = "gs://bdp-ssa/minnie65_phase3_v1/absolute-solo-yak/auburn-elk-detour-synapse_hks_model/spine_morphometry_deltalake"

post_synapse_skeleton_extras_path = DATA_PATH / "post_synapse_skeleton_features_more_dists"

unitary_spine_predictions_path = DATA_PATH / "true_single_spine_model_predictions_deltalake"

TABLE_PATHS = {
    "synapses": synapse_path,
    "pre_hks": pre_synapse_feature_path,
    "post_hks": post_synapse_feature_path,
    "prediction": prediction_path,
    "pre_skeleton": pre_synapse_skeleton_path,
    "post_skeleton": post_synapse_skeleton_path,
    "spine_morphometry_column": spine_morphometry_column_path,
    "spine_morphometry": spine_morphometry,
    "spine_component": spine_component_path,
    "post_skeleton_extras": post_synapse_skeleton_extras_path,
    "unitary_spine_predictions": unitary_spine_predictions_path,
}


DEDUPLICATE = True

VERSION = 1412

simple_spine_morphometry = False  # just load spine size and area if True


# %%

cell_info = load_neuron_info(version=VERSION)

cell_info = (
    cell_info.reset_index()
    .drop_duplicates("pt_root_id", keep=False)
    .set_index("pt_root_id")
)

proofread_info = cell_info.query(
    "(cell_type_source == 'allen_v1_column_types_slanted_ref' or cell_type == 'TH') and axon_cleaned"
)
proofread_roots = proofread_info.index.unique()

column_roots = proofread_info.query(
    "cell_type_source == 'allen_v1_column_types_slanted_ref'"
).index.unique()


cell_info_pl = pl.LazyFrame(cell_info.reset_index())

# cast broad_type, cell_type, mtype to enum
cell_info_pl = cell_info_pl.with_columns(
    pl.col("broad_type").cast(BROAD_TYPE_ENUM),
    pl.col("cell_type").cast(CELL_TYPE_ENUM),
    pl.col("mtype").cast(MTYPE_ENUM),
)

# %%


def add_predictions(synapses: pl.LazyFrame) -> pl.LazyFrame:
    posteriors = (
        pl.scan_delta(TABLE_PATHS["prediction"])
        .select(["synapse_id", "label", "p_spine", "p_shaft", "p_soma"])
        .rename({"label": "tag"})
        .with_columns(pl.col("tag").cast(LABEL_ENUM))
    )
    synapses = synapses.join(
        posteriors,
        on="synapse_id",
        how="left",
    )
    return synapses


def add_cell_info(synapses: pl.LazyFrame) -> pl.LazyFrame:
    add_cell_info_pl = cell_info_pl.select(
        [
            "pt_root_id",
            "broad_type",
            "cell_type",
            "mtype",
            "axon_cleaned",
        ]
    )
    synapses = synapses.join(
        add_cell_info_pl.rename(lambda x: f"pre_{x}"),
        on="pre_pt_root_id",
        how="left",
    ).with_columns(pl.col("pre_axon_cleaned").fill_null(False))
    synapses = synapses.join(
        add_cell_info_pl.drop("axon_cleaned", strict=False).rename(
            lambda x: f"post_{x}"
        ),
        on="post_pt_root_id",
        how="left",
    )
    synapses = synapses.with_columns(
        pl.col("pre_pt_root_id").is_in(proofread_roots).alias("pre_in_selection"),
        pl.col("post_pt_root_id").is_in(column_roots).alias("post_in_selection"),
    )
    return synapses


def add_skeleton_info(synapses: pl.LazyFrame) -> pl.LazyFrame:
    pre_skeleton_info = (
        pl.scan_delta(TABLE_PATHS["pre_skeleton"])
        .select(["synapse_id", "distance_to_root", "compartment"])
        .with_columns(pl.col("compartment").cast(COMPARTMENT_CATEGORIES_ENUM))
        .rename(
            {
                "compartment": "pre_compartment",
                "distance_to_root": "pre_path_distance_to_nuc",
            }
        )
    )
    post_skeleton_info = (
        pl.scan_delta(TABLE_PATHS["post_skeleton"])
        .select(["synapse_id", "distance_to_root", "compartment"])
        .with_columns(pl.col("compartment").cast(COMPARTMENT_CATEGORIES_ENUM))
        .rename(
            {
                "compartment": "post_compartment",
                "distance_to_root": "post_path_distance_to_nuc",
            }
        )
    )

    synapses = synapses.join(
        pre_skeleton_info,
        on="synapse_id",
        how="left",
    ).join(
        post_skeleton_info,
        on="synapse_id",
        how="left",
    )
    return synapses


def add_extended_skeleton_info(synapses: pl.LazyFrame) -> pl.LazyFrame:
    post_skeleton_info = pl.scan_delta(
        str(DATA_PATH / "post_synapse_skeleton_features_more_dists")
    ).drop("post_pt_root_id_partition", "distance_to_root", "compartment")
    synapses = synapses.join(
        post_skeleton_info,
        on="synapse_id",
        how="left",
    )
    return synapses


def add_spine_components(synapses: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    # get info about spine components
    # there is some extra work that needs to happen here because we want to know when we
    # are looking at a multi-input spine and we don't have the other partner in the current
    # table

    synapse_spine_info = (
        pl.scan_delta(TABLE_PATHS["spine_component"])
        .drop("post_pt_root_id_partition")
        .filter(pl.col("component_id") != -1)
        .rename({"index": "synapse_id"})
        .unique(subset="synapse_id", keep="any")
    )

    synapses = synapses.join(
        synapse_spine_info,
        how="left",
        on=["synapse_id", "post_pt_root_id"],
    )
    relevant_spine_components = (
        synapses.filter(
            pl.col("tag") == "spine",
            pl.col("component_id").is_not_null(),
            pl.col("component_id") != -1,
        )
        .select(["post_pt_root_id", "component_id"])
        .unique()
    )
    synapse_spine_info = synapse_spine_info.join(
        relevant_spine_components,
        on=["post_pt_root_id", "component_id"],
        how="inner",
    )

    # start a spine info table, collate info about upstream partners of spines
    spine_info = synapse_spine_info.group_by(["post_pt_root_id", "component_id"]).agg(
        pl.col("synapse_id").n_unique().alias("n_pre_synapses")
    )

    # go back to the full synapse table for things that we might be missing on the
    # multi-spines
    spine_pre_root_info = (
        pl.scan_delta(TABLE_PATHS["synapses"])
        .select(["synapse_id", "pre_pt_root_id"])
        .join(
            synapse_spine_info,
            on="synapse_id",
            how="inner",
        )
        .group_by(["post_pt_root_id", "component_id"])
        .agg(
            pl.col("pre_pt_root_id").unique().alias("pre_pt_root_ids"),
            pl.col("pre_pt_root_id").n_unique().alias("n_pre_pt_root_ids"),
        )
    )

    unitary_spine_info = (
        (
            pl.scan_delta(TABLE_PATHS["unitary_spine_predictions"])
            .drop("post_pt_root_id_partition")
            .rename({"posterior": "is_unitary_spine_posterior"})
        )
        .select(
            "post_pt_root_id",
            "component_id",
            "is_unitary_spine",
            "is_unitary_spine_posterior",
        )
        .unique(subset=["post_pt_root_id", "component_id"], keep="any")
    )
    spine_info = (
        spine_info.join(
            spine_pre_root_info,
            on=["post_pt_root_id", "component_id"],
            how="left",
        )
        .join(
            unitary_spine_info,
            on=["post_pt_root_id", "component_id"],
            how="left",
        )
        .with_columns(
            ((pl.col("n_pre_pt_root_ids") > 1) & pl.col("is_unitary_spine")).alias(
                "is_multi"
            )
        )
    )
    spine_info = spine_info.unique(
        subset=["post_pt_root_id", "component_id"], keep="any"
    ).sort(["post_pt_root_id", "component_id"])
    # add a index column that counts up from 0 to simplify mappings
    spine_info = spine_info.with_columns(pl.arange(0, pl.len()).alias("group_id"))
    return synapses, spine_info


def add_spine_morphometry(spine_info: pl.LazyFrame) -> pl.LazyFrame:
    spine_morphometry = pl.scan_delta(TABLE_PATHS["spine_morphometry"]).drop(
        pl.selectors.contains("min_"), pl.selectors.contains("max_")
    )

    if simple_spine_morphometry:
        spine_morphometry = spine_morphometry.select(
            [
                "post_pt_root_id",
                "component_id",
                "size_nm3",
                "area_nm2",
            ]
        )

    spine_info = spine_info.join(
        spine_morphometry,
        on=["post_pt_root_id", "component_id"],
        how="left",
    )
    return spine_info


def join_spine_to_synapses(
    synapses: pl.LazyFrame, spine_info: pl.LazyFrame
) -> pl.LazyFrame:
    if DEDUPLICATE:
        spine_info = spine_info.unique(["post_pt_root_id", "component_id"], keep="any")
    select_cols = [
        "post_pt_root_id",
        "component_id",
        "n_pre_synapses",
        "pre_pt_root_ids",
        "n_pre_pt_root_ids",
        "is_multi",
        "group_id",
    ]
    if "size_nm3" in spine_info.collect_schema():
        select_cols += ["size_nm3", "area_nm2"]

    synapses = synapses.join(
        spine_info.select(*select_cols).rename(
            {
                "n_pre_synapses": "spine_n_pre_synapses",
                "n_pre_pt_root_ids": "spine_n_pre_pt_root_ids",
                "pre_pt_root_ids": "spine_pre_pt_root_ids",
                "group_id": "spine_group_id",
                "is_multi": "spine_is_multi",
                "size_nm3": "spine_size_nm3",
                "area_nm2": "spine_area_nm2",
            },
            strict=False,
        ),
        on=["post_pt_root_id", "component_id"],
        how="left",
    )
    synapses = synapses.with_columns(
        pl.col("spine_is_multi").fill_null(False),
    )

    # create a new column, 'tag_detailed' that splits 'spine' into "single_spine" and
    # "multi_spine" based on "is_multi"
    synapses = (
        synapses.with_columns(pl.col("tag").cast(pl.String).alias("_tag_str"))
        .with_columns(
            pl.when((pl.col("_tag_str") == "spine") & pl.col("spine_is_multi"))
            .then(pl.lit("multi_spine"))
            .when((pl.col("_tag_str") == "spine") & ~pl.col("spine_is_multi"))
            .then(pl.lit("single_spine"))
            .otherwise(pl.col("_tag_str"))
            .cast(pl.Enum(LABEL_DETAILED_CATEGORIES))
            .alias("tag_detailed")
        )
        .drop("_tag_str")
    )
    return synapses


def add_spatial_mappings(synapses: pl.LazyFrame) -> pl.LazyFrame:
    tform = minnie_transform_vx()

    pt_resolution = np.array([4.0, 4.0, 40.0])  # nm

    pre_transform_schema = synapses.collect_schema()

    synapses = synapses.select(pre_transform_schema.keys())

    def apply_transformation(batch: pl.DataFrame) -> pl.DataFrame:
        transformed_pts = tform.apply(
            batch.select(
                "ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"
            ).to_numpy(),
            as_int=False,
        )

        nm_pts = (
            batch.select(
                "ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"
            ).to_numpy()
            * pt_resolution
        )

        pre_nuc_positions_nm = (
            cell_info[
                ["pt_position_x", "pt_position_y", "pt_position_z"]
            ]  # already in nm
            .reindex(index=batch.select("pre_pt_root_id").to_numpy().squeeze())
            .values
        )
        pre_distances = np.linalg.norm(pre_nuc_positions_nm - nm_pts, axis=1)

        post_nuc_positions_nm = (
            cell_info[
                ["pt_position_x", "pt_position_y", "pt_position_z"]
            ]  # already in nm
            .reindex(index=batch.select("post_pt_root_id").to_numpy().squeeze())
            .values
        )
        post_distances = np.linalg.norm(post_nuc_positions_nm - nm_pts, axis=1)

        pre_nuc_positions_um_transformed = (
            cell_info[["pt_position_um_x", "pt_position_um_y", "pt_position_um_z"]]
            .reindex(index=batch.select("pre_pt_root_id").to_numpy().squeeze())
            .values
        )
        pre_delta_depth_um = (
            pre_nuc_positions_um_transformed[:, 1] - transformed_pts[:, 1]
        )
        pre_delta_depth_nm = pre_delta_depth_um * 1000.0

        post_nuc_positions_um_transformed = (
            cell_info[["pt_position_um_x", "pt_position_um_y", "pt_position_um_z"]]
            .reindex(index=batch.select("post_pt_root_id").to_numpy().squeeze())
            .values
        )
        post_delta_depth_um = (
            post_nuc_positions_um_transformed[:, 1] - transformed_pts[:, 1]
        )
        post_delta_depth_nm = post_delta_depth_um * 1000.0

        # TODO add pre and post depth difference

        # create a small DataFrame for just the new columns
        new_cols = pl.DataFrame(
            {
                "transformed_um_x": transformed_pts[:, 0].astype(float),
                "transformed_um_y": transformed_pts[:, 1].astype(float),
                "transformed_um_z": transformed_pts[:, 2].astype(float),
                "ctr_pt_position_nm_x": nm_pts[:, 0],
                "ctr_pt_position_nm_y": nm_pts[:, 1],
                "ctr_pt_position_nm_z": nm_pts[:, 2],
                "pre_euc_distance_to_nuc": pre_distances,
                "post_euc_distance_to_nuc": post_distances,
                "pre_delta_depth_nm": pre_delta_depth_nm,
                "post_delta_depth_nm": post_delta_depth_nm,
            }
        )

        # merge new columns back into the current batch
        return batch.hstack(new_cols)

    # build explicit output schema for lazy safety
    schema_out = {
        **pre_transform_schema,
        "transformed_um_x": pl.Float64,
        "transformed_um_y": pl.Float64,
        "transformed_um_z": pl.Float64,
        "ctr_pt_position_nm_x": pl.Float64,
        "ctr_pt_position_nm_y": pl.Float64,
        "ctr_pt_position_nm_z": pl.Float64,
        "pre_euc_distance_to_nuc": pl.Float64,
        "post_euc_distance_to_nuc": pl.Float64,
        "pre_delta_depth_nm": pl.Float64,
        "post_delta_depth_nm": pl.Float64,
    }

    # NOTE: was helpful to put map_batches last to avoid the optimizer removing relevant
    # columns too early. other solution was 'projection_pushdown': False in map_batches
    synapses = synapses.map_batches(apply_transformation, schema=schema_out)
    return synapses
