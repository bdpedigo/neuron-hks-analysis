"""The census tables on the column cells."""

from datetime import datetime

import polars as pl
from standard_transform.datasets import minnie_transform_vx

from ..io import LABEL_CATEGORIES, LABEL_DETAILED_CATEGORIES
from ..sources import MICRONS_SYNAPSE_RESOLUTION, MICRONS_SYNAPSES_PATH, VERSION, cutoff, pin_params
from . import cell_info, nuclei, skeleton  # noqa: F401 -- register the tables that column_post_synapses reads
from .catalog import Tables
from .deployment import delta_table
from .features import add_distance_to_nucleus
from .predictions import synapse_posteriors
from .structures import read_spines, synapse_partitions, synapse_structure

SPINE_PRODUCTS = (
    "synapse_mapping",
    "vertex_structures",
    "structure_morphometry",
    "structure_features",
    "structure_predictions",
)
DOMAIN_PRODUCTS = ("vertex_domains", "domain_predictions")
SIDES = ("pre", "post")


def column_root_ids() -> list[int]:
    """The roots of the column cells in `cell_selection`."""
    return (
        Tables["cell_selection"]
        .lazy()
        .filter(pl.col("is_column"))
        .select("pt_root_id")
        .collect()["pt_root_id"]
        .to_list()
    )


# NOTE: the old `filtered_spine_info` also held the spines on other roots that a
# proofread axon reaches, but 5C left and 5E left kept only the column cells
# (`scripts/stat_summaries.py`, the map of `post_broad_type`). Revisit this scope
# with the full-run tables: a wider table is another call of `read_spines`.
Tables.register(
    "column_spines",
    description=(
        "One row per spine structure on a column cell, keyed by (`root_id`, "
        "`structure_id`). It holds the morphometry and spectral features, the "
        "synapse and presynaptic root counts, the unitary-spine posteriors and "
        "`is_multi`. Read at the `test_census` cutoff. It replaces the old "
        "`filtered_spine_info`."
    ),
    build=lambda: read_spines(column_root_ids(), as_of=cutoff("test_census")).lazy(),
    cache_params=lambda: {
        "version": VERSION,
        "cell_selection": Tables["cell_selection"].cache_key,
        **pin_params(*SPINE_PRODUCTS, as_of=cutoff("test_census")),
    },
)


def _read_post_synapse_rows(root_ids: list[int]) -> pl.DataFrame:
    """Every column of the synapse table (v1822) for the synapses onto the post
    roots `root_ids`, with `id` as `synapse_id`."""
    return (
        pl.scan_delta(delta_table(str(MICRONS_SYNAPSES_PATH)))
        .filter(
            pl.col("post_pt_root_id_partition").is_in(synapse_partitions(root_ids)),
            pl.col("post_pt_root_id").is_in(root_ids),
        )
        .drop("post_pt_root_id_partition")
        .rename({"id": "synapse_id"})
        .collect()
    )


def _drop_autapses(synapses: pl.DataFrame) -> pl.DataFrame:
    return synapses.filter(pl.col("pre_pt_root_id") != pl.col("post_pt_root_id"))


def _spine_columns(structure: pl.DataFrame) -> pl.DataFrame:
    """`synapse_structure` keyed by `synapse_id`, with a `spine_` prefix on each
    structure column. `structure_id` keeps its name."""
    structure = structure.drop("root_id", "pre_pt_root_id")
    return structure.rename(
        {
            name: f"spine_{name}"
            for name in structure.columns
            if name not in ("synapse_id", "structure_id")
        }
    )


def _add_skeleton_sides(synapses: pl.DataFrame, features: pl.LazyFrame) -> pl.DataFrame:
    """Add the compartment and path distance of each side of each synapse, from
    `skeleton_synapse_features`, and the distance to the farthest tip of the post
    side.

    A side whose root has no skeleton has null values. Only proofread roots have
    skeletons, so most pre sides are null.
    """
    columns = {"pre": ["compartment", "path_distance_to_nuc"]}
    columns["post"] = columns["pre"] + ["max_dist_to_tip"]
    synapse_ids = synapses["synapse_id"].to_list()
    for side in SIDES:
        side_features = (
            features.filter(pl.col("side") == side, pl.col("synapse_id").is_in(synapse_ids))
            .select(
                "synapse_id",
                pl.col("root_id").alias(f"{side}_pt_root_id"),
                *[pl.col(name).alias(f"{side}_{name}") for name in columns[side]],
            )
            .collect()
        )
        synapses = synapses.join(
            side_features, on=["synapse_id", f"{side}_pt_root_id"], how="left"
        )
    return synapses


def _add_cell_info(synapses: pl.DataFrame, cells: pl.DataFrame) -> pl.DataFrame:
    """Add the broad type, cell type, mtype and `axon_cleaned` of each side from
    `cell_info`. They are null for a root that `cell_info` does not hold."""
    columns = ["broad_type", "cell_type", "mtype", "axon_cleaned"]
    for side in SIDES:
        synapses = synapses.join(
            cells.select(
                pl.col("pt_root_id").alias(f"{side}_pt_root_id"),
                *[pl.col(name).alias(f"{side}_{name}") for name in columns],
            ),
            on=f"{side}_pt_root_id",
            how="left",
        )
    return synapses


def _add_positions(synapses: pl.DataFrame, nucleus_points: pl.LazyFrame) -> pl.DataFrame:
    """Add `{pre,post}_euc_distance_to_nuc`, the distance (nm) from the synapse to
    the nearest nucleus of each side's root, and `ctr_pt_position_um_{x,y,z}`, the
    synapse position in the transformed space (µm).

    The distance is null for a root with no nucleus.
    """
    nm = [f"_ctr_nm_{c}" for c in "xyz"]
    lf = synapses.lazy().with_columns(
        (pl.col(f"ctr_pt_position_{c}") * resolution).alias(name)
        for c, resolution, name in zip("xyz", MICRONS_SYNAPSE_RESOLUTION, nm)
    )
    for side in SIDES:
        lf = add_distance_to_nucleus(
            lf.with_columns(root_id=pl.col(f"{side}_pt_root_id")),
            nucleus_points,
            position=tuple(nm),
            name=f"{side}_euc_distance_to_nuc",
        ).drop("root_id")
    synapses = lf.drop(nm).collect()
    transformed = minnie_transform_vx().apply(
        synapses.select(f"ctr_pt_position_{c}" for c in "xyz").to_numpy()
    )
    return synapses.with_columns(
        pl.Series(f"ctr_pt_position_um_{c}", transformed[:, i]) for i, c in enumerate("xyz")
    )


def _add_tag_detailed(synapses: pl.DataFrame) -> pl.DataFrame:
    """Add `tag_detailed`: a spine-tagged synapse is `multi_spine` if its spine is
    multi-input, and `single_spine` otherwise. Every other synapse keeps its `tag`.

    A spine-tagged synapse with a null `spine_is_multi` (no structure, or a
    structure with no prediction) is `single_spine`.
    """
    is_multi = pl.col("spine_is_multi").fill_null(False)
    return synapses.with_columns(
        tag_detailed=pl.when(pl.col("tag") == "spine")
        .then(pl.when(is_multi).then(pl.lit("multi_spine")).otherwise(pl.lit("single_spine")))
        .otherwise(pl.col("tag").cast(pl.String))
        .cast(pl.Enum(LABEL_DETAILED_CATEGORIES))
    )


def read_post_synapses(root_ids: list[int], *, as_of: datetime) -> pl.DataFrame:
    """One row per synapse in the synapse table (v1822) onto the post roots
    `root_ids`, with no autapses, and every column of the synapse table.

    - `synapse_structure`: `structure_id`, and its other columns with a `spine_`
      prefix (`spine_n_pre_root_ids`, `spine_is_multi` and so on).
    - `synapse_posteriors`: the domain posteriors and `posterior_entropy`, and the
      domain label as `tag`.
    - `spine_size_nm3` and `spine_area_nm2` from `column_spines`.
    - The skeleton compartments and distances, the cell info and the nucleus
      distances of each side, with a `pre_` or `post_` prefix. See
      `_add_skeleton_sides`, `_add_cell_info` and `_add_positions`.
    - `tag_detailed`: see `_add_tag_detailed`.

    It reads every deployment table at `as_of`.
    """
    # NOTE: `synapse_structure` reads the synapse table a second time, but only
    # three columns, and it counts the presynaptic roots of a structure from every
    # synapse on the root, autapses too.
    synapses = _drop_autapses(_read_post_synapse_rows(root_ids))
    posteriors = synapse_posteriors(
        synapses.select("synapse_id", pl.col("post_pt_root_id").alias("root_id")), as_of=as_of
    )
    spines = Tables["column_spines"].lazy().select(
        pl.col("root_id").alias("post_pt_root_id"),
        "structure_id",
        pl.col("size_nm3").alias("spine_size_nm3"),
        pl.col("area_nm2").alias("spine_area_nm2"),
    )
    return (
        synapses.join(
            _spine_columns(synapse_structure(root_ids, as_of=as_of)), on="synapse_id", how="left"
        )
        .join(
            posteriors.drop("root_id").rename({"pred_label": "tag"}), on="synapse_id", how="left"
        )
        .with_columns(pl.col("tag").cast(pl.Enum(LABEL_CATEGORIES)))
        .join(spines.collect(), on=["post_pt_root_id", "structure_id"], how="left")
        .pipe(_add_skeleton_sides, Tables["skeleton_synapse_features"].lazy())
        .pipe(_add_cell_info, Tables["cell_info"].collect())
        .pipe(_add_positions, Tables["root_nuclei"].lazy())
        .pipe(_add_tag_detailed)
        .sort("post_pt_root_id", "synapse_id")
    )


# NOTE: 864691136674080135 is a column cell with no skeleton file. Its synapses
# stay, with null post compartments and path distances (user decision,
# 2026-10-23).
Tables.register(
    "column_post_synapses",
    description=(
        "One row per synapse onto a column cell, with no autapses, at the "
        "`test_census` cutoff. The synapses from the column cells onto other roots "
        "are not here: see `proofread_pre_synapses`. It holds every column of the "
        "synapse table, the domain posteriors and `tag`, the spine structure "
        "(`structure_id` and `spine_*`, with the volume from `column_spines`), "
        "`tag_detailed`, and per side (`pre_`, `post_`) the skeleton compartment and "
        "distances, the cell info and the distance to the nucleus. The column cell "
        "864691136674080135 has no skeleton, so its post compartments are null. It "
        "replaces the column half of the old `filtered_synapses`."
    ),
    build=lambda: read_post_synapses(column_root_ids(), as_of=cutoff("test_census")).lazy(),
    cache_params=lambda: {
        "version": VERSION,
        **{
            name: Tables[name].cache_key
            for name in (
                "cell_selection",
                "cell_info",
                "column_spines",
                "skeleton_synapse_features",
                "root_nuclei",
            )
        },
        **pin_params(*SPINE_PRODUCTS, *DOMAIN_PRODUCTS, as_of=cutoff("test_census")),
    },
)
