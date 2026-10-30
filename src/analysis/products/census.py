"""The census tables on the column cells and the proofread cells, and the per-root
aggregations of their synapses."""

from datetime import datetime

import polars as pl
from standard_transform.datasets import minnie_transform_vx

from ..io import LABEL_CATEGORIES, LABEL_DETAILED_CATEGORIES
from ..sources import MICRONS_SYNAPSE_RESOLUTION, VERSION, cutoff, pin_params
from . import cell_info, nuclei, skeleton  # noqa: F401 -- register the tables that column_post_synapses reads
from .catalog import Tables
from .features import add_distance_to_nucleus
from .predictions import synapse_posteriors
from .structures import read_spines, synapse_structure
from .synapses import scan_post_synapses

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
# NOTE: the old table held only the spines with a spine-tagged synapse. This one
# holds every structure, so 1,848,798 of its 4,315,702 rows have no synapse (user
# decision, 2026-10-30). 5C left and 5E left inner-join it to the spine synapses,
# so they read the same spines as before.
Tables.register(
    "column_spines",
    description=(
        "One row per spine structure on a column cell, keyed by (`root_id`, "
        "`structure_id`), with or without a synapse. It holds the morphometry and spectral features, the "
        "synapse and presynaptic root counts, the unitary-spine posteriors and "
        "`is_multi`. Read at the `test_census` cutoff. It replaces the old "
        "`filtered_spine_info`."
    ),
    build=lambda: read_spines(column_root_ids(), as_of=cutoff("test_census")).lazy(),
    cache_params=lambda: {
        "version": VERSION,
        "cell_selection": Tables["cell_selection"].cache_key,
        "microns_synapses": Tables["microns_synapses"].cache_key,
        **pin_params(*SPINE_PRODUCTS, as_of=cutoff("test_census")),
    },
)


def _read_post_synapse_rows(root_ids: list[int]) -> pl.DataFrame:
    """Every column of `microns_synapses` (v1822) for the synapses onto the post
    roots `root_ids`, with `id` as `synapse_id`."""
    return scan_post_synapses(root_ids).rename({"id": "synapse_id"}).collect()


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
    # NOTE: the old tables (meshrep `tools/make_mega_tables.py`) measured to the
    # soma point in `cell_info`, so a root that `cell_info` drops, such as a root
    # listed twice, had NaN. Here a root with several nuclei, such as a merge, gets
    # the distance to the nearest one. This is the rule of `distance_to_nucleus`
    # in the domain model (user decision, 2026-10-30).
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
    # three columns.
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


# The spine columns that each synapse row carries. The spine table keeps the rest.
SPINE_SYNAPSE_COLUMNS = [
    "n_synapses",
    "n_pre_root_ids",
    "unitary_pred_label",
    "is_multi",
    "size_nm3",
    "area_nm2",
]


def join_census_features(
    synapses: pl.LazyFrame, features: pl.LazyFrame, spines: pl.LazyFrame
) -> pl.LazyFrame:
    """`synapses` (rows of `microns_synapses`, with `id`) with no autapses, and with
    the run features of each synapse:

    - `tag` and `structure_id` from `features` (`synapse_census_features`). A
      synapse with no row there (unmapped) has null values.
    - The `SPINE_SYNAPSE_COLUMNS` of its spine from `spines`, keyed by (`root_id`,
      `structure_id`), with a `spine_` prefix.
    - `tag_detailed`: see `_add_tag_detailed`.
    """
    spines = spines.select(
        pl.col("root_id").alias("post_pt_root_id"),
        "structure_id",
        *[pl.col(name).alias(f"spine_{name}") for name in SPINE_SYNAPSE_COLUMNS],
    )
    return (
        synapses.rename({"id": "synapse_id"})
        .pipe(_drop_autapses)
        .join(features.select("synapse_id", "tag", "structure_id"), on="synapse_id", how="left")
        .with_columns(pl.col("tag").cast(pl.Enum(LABEL_CATEGORIES)))
        .join(spines, on=["post_pt_root_id", "structure_id"], how="left")
        .pipe(_add_tag_detailed)
    )


def annotate_synapses(synapses: pl.DataFrame) -> pl.DataFrame:
    """`synapses` (from `join_census_features`) with the skeleton compartments and
    distances, the cell info and the nucleus distances of each side. See
    `_add_skeleton_sides`, `_add_cell_info` and `_add_positions`."""
    return (
        synapses.pipe(_add_skeleton_sides, Tables["skeleton_synapse_features"].lazy())
        .pipe(_add_cell_info, Tables["cell_info"].collect())
        .pipe(_add_positions, Tables["root_nuclei"].lazy())
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
        "are not here: see `proofread_pre_synapses`. It holds every column of "
        "`microns_synapses`, the domain posteriors and `tag`, the spine structure "
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
                "microns_synapses",
                "skeleton_synapse_features",
                "root_nuclei",
            )
        },
        **pin_params(*SPINE_PRODUCTS, *DOMAIN_PRODUCTS, as_of=cutoff("test_census")),
    },
)


KNOWN_TAGS = ["spine", "shaft", "soma"]
DETAILED_SPINE_TAGS = ["single_spine", "multi_spine"]


def _aggregation_exprs() -> list[pl.Expr]:
    """The counts and mean sizes per group, from the old `agg_exprs` in
    `meshrep/tools/make_mega_tables.py`.

    A known synapse has a `tag` of spine, shaft or soma. A spine site is a distinct
    (`post_pt_root_id`, `structure_id`) pair with at least one spine-tagged
    synapse. The old pre side counted distinct structure IDs, so the spines on two
    post roots with the same ID counted as one site.
    """
    tag, detailed = pl.col("tag"), pl.col("tag_detailed")
    site = pl.struct("post_pt_root_id", "structure_id")
    on_site = (tag == "spine") & pl.col("structure_id").is_not_null()
    return [
        pl.len().alias("total_synapses"),
        tag.is_in(KNOWN_TAGS).sum().alias("total_known_synapses"),
        *[(tag == t).sum().alias(f"{t}_synapses") for t in KNOWN_TAGS],
        *[(detailed == t).sum().alias(f"{t}_synapses") for t in DETAILED_SPINE_TAGS],
        site.filter(on_site).n_unique().alias("spine_sites"),
        site.filter(on_site & (detailed == "multi_spine")).n_unique().alias("multi_spine_sites"),
        pl.col("size").mean().alias("mean_synapse_size"),
        *[pl.col("size").filter(tag == t).mean().alias(f"mean_{t}_size") for t in KNOWN_TAGS],
        *[
            pl.col("size").filter(detailed == t).mean().alias(f"mean_{t}_size")
            for t in DETAILED_SPINE_TAGS
        ],
    ]


def _ratio(numerator: str, denominator: str) -> pl.Expr:
    return pl.col(numerator) / pl.col(denominator)


def _aggregate(synapses: pl.LazyFrame, by: str, prefix: str) -> pl.LazyFrame:
    """One row per value of `by`, with the counts, proportions and mean sizes of
    its synapses. Each column but `by` takes `prefix`."""
    return (
        synapses.group_by(by)
        .agg(_aggregation_exprs())
        .with_columns(
            unknown_synapses=pl.col("total_synapses") - pl.col("total_known_synapses"),
        )
        .with_columns(
            spine_shaft_ratio=_ratio("spine_synapses", "shaft_synapses"),
            p_spine_synapse=_ratio("spine_synapses", "total_known_synapses"),
            p_shaft_synapse=_ratio("shaft_synapses", "total_known_synapses"),
            p_soma_synapse=_ratio("soma_synapses", "total_known_synapses"),
            p_unknown_synapse=_ratio("unknown_synapses", "total_synapses"),
            p_single_spine_synapse=_ratio("single_spine_synapses", "total_known_synapses"),
            p_multi_spine_synapse=_ratio("multi_spine_synapses", "total_known_synapses"),
            p_spine_synapse_is_multi=_ratio("multi_spine_synapses", "spine_synapses"),
            p_spine_site_is_multi=_ratio("multi_spine_sites", "spine_sites"),
            mean_multi_spine_inputs=_ratio("multi_spine_synapses", "multi_spine_sites"),
            mean_spine_inputs=_ratio("spine_synapses", "spine_sites"),
        )
        .rename(lambda name: name if name == by else f"{prefix}{name}")
        .sort(by)
    )


def aggregate_post(synapses: pl.LazyFrame) -> pl.LazyFrame:
    """One row per post root of `synapses`, with `post_*` counts, proportions and
    mean sizes. See `_aggregation_exprs`. The caller picks the synapses."""
    return _aggregate(synapses, "post_pt_root_id", "post_")


def aggregate_pre(synapses: pl.LazyFrame) -> pl.LazyFrame:
    """One row per pre root of `synapses`, with the `post_*` columns of
    `aggregate_post` as `pre_*` for all targets, and as `pre_to_exc_*` for the
    excitatory targets (`post_broad_type`). The `pre_to_exc_*` columns are null for
    a pre root with no excitatory target."""
    to_exc = _aggregate(
        synapses.filter(pl.col("post_broad_type") == "excitatory"), "pre_pt_root_id", "pre_to_exc_"
    )
    return (
        _aggregate(synapses, "pre_pt_root_id", "pre_")
        .join(to_exc, on="pre_pt_root_id", how="left")
        .sort("pre_pt_root_id")
    )


def build_proofread_cell_info() -> pl.LazyFrame:
    proofread = Tables["cell_selection"].lazy().filter(pl.col("is_proofread"))
    post = aggregate_post(Tables["column_post_synapses"].lazy()).rename(
        {"post_pt_root_id": "pt_root_id"}
    )
    return (
        Tables["cell_info"]
        .lazy()
        .join(proofread, on="pt_root_id", how="inner")
        .join(post, on="pt_root_id", how="left")
        .sort("pt_root_id")
    )


# NOTE: a test-cohort version. Only the column cells have `post_*` data, because
# `column_post_synapses` holds only the synapses onto the column cells. TASK-19
# rebuilds this table at the full-run pins, and adds the `pre_*` columns from
# `aggregate_pre` on `proofread_pre_synapses` (TASK-18).
Tables.register(
    "proofread_cell_info",
    description=(
        "Test-cohort version. One row per proofread cell (`is_proofread` in "
        "`cell_selection`, the thalamic cells too): the columns of `cell_info`, the "
        "`cell_selection` flags, and the `post_*` counts and proportions of "
        "`aggregate_post` on `column_post_synapses`. Only the column cells have "
        "`post_*` data, and the other rows have nulls. It has no `pre_*` columns yet. "
        "TASK-19 rebuilds it at the full-run pins and adds them. It replaces the old "
        "`proofread_info` of `stat_summaries.py`."
    ),
    build=build_proofread_cell_info,
    cache_params=lambda: {
        "version": VERSION,
        **{
            name: Tables[name].cache_key
            for name in ("cell_info", "cell_selection", "column_post_synapses")
        },
    },
)
