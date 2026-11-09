"""The census tables, and the per-root aggregations of their synapses.

Every table here derives locally from `microns_synapses` and the two scan tables of
`census_scan.py` (`synapse_census_features`, `spine_census_features`), with the
fixed lookups. All three share `bucket[256]` of the post root, so the join of a
synapse to its run features never leaves one bucket (see `census_synapses`).
"""

import polars as pl
from standard_transform.datasets import minnie_transform_vx

from ..io import LABEL_CATEGORIES, LABEL_DETAILED_CATEGORIES
from ..sources import MICRONS_SYNAPSE_RESOLUTION, VERSION
from . import cell_info, census_scan, nuclei, skeleton  # noqa: F401 -- register the tables that the census tables read
from .catalog import Tables
from .deployment import bucket_filter
from .features import add_distance_to_nucleus
from .synapses import BUCKET_COLUMN, N_BUCKETS

SIDES = ("pre", "post")
# The tables that `census_synapses` joins.
GRAIN_TABLES = ("microns_synapses", "synapse_census_features", "spine_census_features")
# The lookups that `annotate_synapses` reads.
ANNOTATION_TABLES = ("skeleton_synapse_features", "cell_info", "root_nuclei")
# The buckets that `census_synapses` collects at once.
BUCKETS_PER_BATCH = 8


def _root_ids(flag: str) -> list[int]:
    """The roots of `cell_selection` with `flag`."""
    return (
        Tables["cell_selection"]
        .lazy()
        .filter(pl.col(flag))
        .select("pt_root_id")
        .collect()["pt_root_id"]
        .to_list()
    )


def column_root_ids() -> list[int]:
    """The roots of the column cells in `cell_selection`."""
    return _root_ids("is_column")


def _cache_keys(*names: str) -> dict:
    return {"version": VERSION, **{name: Tables[name].cache_key for name in names}}


def _drop_autapses(synapses: pl.DataFrame) -> pl.DataFrame:
    return synapses.filter(pl.col("pre_pt_root_id") != pl.col("post_pt_root_id"))


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

    - `tag`, `tag_posterior` and `structure_id` from `features`
      (`synapse_census_features`). A synapse with no row there has null values: it
      is unmapped, or its root has no run data. It counts as unknown.
    - The `SPINE_SYNAPSE_COLUMNS` of its spine from `spines`
      (`spine_census_features`), keyed by (`post_pt_root_id`, `structure_id`), with
      a `spine_` prefix.
    - `tag_detailed`: see `_add_tag_detailed`.
    """
    spines = spines.select(
        "post_pt_root_id",
        "structure_id",
        *[pl.col(name).alias(f"spine_{name}") for name in SPINE_SYNAPSE_COLUMNS],
    )
    return (
        synapses.rename({"id": "synapse_id"})
        .pipe(_drop_autapses)
        .join(
            features.select("synapse_id", "tag", "tag_posterior", "structure_id"),
            on="synapse_id",
            how="left",
        )
        .with_columns(pl.col("tag").cast(pl.Enum(LABEL_CATEGORIES)))
        .join(spines, on=["post_pt_root_id", "structure_id"], how="left")
        .pipe(_add_tag_detailed)
    )


def _scan_bucket(name: str, chunk: int) -> pl.LazyFrame:
    """The rows of the bucketed table `name` in bucket `chunk` of `bucket[256]`."""
    table = Tables[name]
    return (
        table.lazy()
        .filter(bucket_filter(table.cache_path, chunk, N_BUCKETS))
        .drop(BUCKET_COLUMN)
    )


def census_synapses(where: pl.Expr, then=None) -> pl.DataFrame:
    """Every synapse of `microns_synapses` that `where` keeps, with its run features
    (see `join_census_features`), and `then` applied per bucket.

    The three tables share the buckets of the post root, so the join runs bucket by
    bucket, `BUCKETS_PER_BATCH` buckets at a time. `then` takes the lazy synapses of
    one bucket. It can aggregate them by post root, because a post root has all its
    synapses in one bucket.
    """
    parts = [
        join_census_features(
            _scan_bucket("microns_synapses", chunk).filter(where),
            _scan_bucket("synapse_census_features", chunk),
            _scan_bucket("spine_census_features", chunk),
        )
        for chunk in range(N_BUCKETS)
    ]
    if then is not None:
        parts = [then(part) for part in parts]
    batches = [parts[i : i + BUCKETS_PER_BATCH] for i in range(0, len(parts), BUCKETS_PER_BATCH)]
    return pl.concat(
        [frame for batch in batches for frame in pl.collect_all(batch, engine="streaming")]
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


def build_column_spines() -> pl.LazyFrame:
    return (
        Tables["spine_census_features"]
        .lazy()
        .filter(pl.col("post_pt_root_id").is_in(column_root_ids()))
        .drop(BUCKET_COLUMN)
        .sort("post_pt_root_id", "structure_id")
    )


# NOTE: the old `filtered_spine_info` held only the spines with a spine-tagged
# synapse, and also the spines on other roots that a proofread axon reaches. This
# table holds every structure on the column cells, with or without a synapse (user
# decision, 2026-10-30): 5C left and 5E left inner-join it to the spine synapses.
Tables.register(
    "column_spines",
    description=(
        "The rows of `spine_census_features` on the column cells: one row per spine "
        "structure, with or without a synapse, keyed by (`post_pt_root_id`, "
        "`structure_id`). It holds `size_nm3`, `area_nm2`, the representative "
        "point `rep_x/y/z` in nm, the synapse and "
        "presynaptic root counts (no autapses), `posterior_unitary`, "
        "`unitary_pred_label` and `is_multi`, at the `census` cutoff. It replaces "
        "the old `filtered_spine_info`."
    ),
    build=build_column_spines,
    cache_params=lambda: _cache_keys("cell_selection", "spine_census_features"),
)


def build_column_post_synapses() -> pl.LazyFrame:
    synapses = census_synapses(pl.col("post_pt_root_id").is_in(column_root_ids()))
    return annotate_synapses(synapses).lazy()


# NOTE: 864691136674080135 is a column cell with no skeleton file. Its synapses
# stay, with null post compartments and path distances (user decision,
# 2026-10-23).
Tables.register(
    "column_post_synapses",
    description=(
        "One row per synapse onto a column cell, with no autapses, at the `census` "
        "cutoff. The synapses from the column cells onto other roots are not here: "
        "see `proofread_pre_synapses`. It holds every column of `microns_synapses`, "
        "`tag`, `tag_posterior` and `structure_id` from `synapse_census_features`, "
        "the `spine_*` columns of `spine_census_features`, `tag_detailed`, and per "
        "side (`pre_`, `post_`) the skeleton compartment and distances, the cell "
        "info and the distance to the nucleus. The column cell 864691136674080135 "
        "has no skeleton, so its post compartments are null. It replaces the column "
        "half of the old `filtered_synapses`."
    ),
    build=build_column_post_synapses,
    cache_params=lambda: _cache_keys("cell_selection", *GRAIN_TABLES, *ANNOTATION_TABLES),
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


# The target broad types of the pre-side rollups, and their column prefixes.
PRE_TARGETS = {"excitatory": "pre_to_exc_", "inhibitory": "pre_to_inh_"}


def aggregate_pre(synapses: pl.LazyFrame) -> pl.LazyFrame:
    """One row per pre root of `synapses`, with the `post_*` columns of
    `aggregate_post` as `pre_*` for all targets, and as `pre_to_exc_*` and
    `pre_to_inh_*` for the excitatory and inhibitory targets (`post_broad_type`). A
    pre root with no target of a type has nulls in its columns."""
    rollup = _aggregate(synapses, "pre_pt_root_id", "pre_")
    for broad_type, prefix in PRE_TARGETS.items():
        to_type = _aggregate(
            synapses.filter(pl.col("post_broad_type") == broad_type), "pre_pt_root_id", prefix
        )
        rollup = rollup.join(to_type, on="pre_pt_root_id", how="left")
    return rollup.sort("pre_pt_root_id")


def build_post_aggregation() -> pl.LazyFrame:
    roots = Tables["cell_info"].lazy().select("pt_root_id").collect()["pt_root_id"]
    post = census_synapses(pl.col("post_pt_root_id").is_in(roots.implode()), then=aggregate_post)
    return post.sort("post_pt_root_id").lazy()


Tables.register(
    "post_aggregation",
    description=(
        "One row per root of `cell_info` with a synapse in `microns_synapses`, as in "
        "the old `post_synapse_aggregation`: the `post_*` columns of "
        "`aggregate_post` on every synapse onto the root, with no autapses, at the "
        "`census` cutoff. These are the counts and proportions of spine, shaft, soma "
        "and unknown synapses and of single- and multi-input spines, the spine "
        "sites, the mean sizes and the mean inputs per spine. A synapse with no run "
        "features is unknown, so a root with no run data has only unknown synapses."
    ),
    build=build_post_aggregation,
    cache_params=lambda: _cache_keys("cell_info", *GRAIN_TABLES),
)


def build_proofread_pre_synapses() -> pl.LazyFrame:
    synapses = census_synapses(pl.col("pre_pt_root_id").is_in(_root_ids("is_proofread")))
    return annotate_synapses(synapses).lazy()


Tables.register(
    "proofread_pre_synapses",
    description=(
        "One row per synapse from a proofread cell onto any root, with no autapses, "
        "at the `census` cutoff, and with the columns of `column_post_synapses`. As "
        "in the old `filtered_synapses`, the targets are not filtered. A target "
        "outside the full run has null run features (`tag`, `structure_id`, "
        "`spine_*`), and a target outside `cell_info` has null cell info. The "
        "distance to the post nucleus comes from the nucleus position, because most "
        "targets have no skeleton. It replaces the proofread half of the old "
        "`filtered_synapses`."
    ),
    build=build_proofread_pre_synapses,
    cache_params=lambda: _cache_keys("cell_selection", *GRAIN_TABLES, *ANNOTATION_TABLES),
)


def build_proofread_cell_info() -> pl.LazyFrame:
    proofread = Tables["cell_selection"].lazy().filter(pl.col("is_proofread"))
    post = Tables["post_aggregation"].lazy().rename({"post_pt_root_id": "pt_root_id"})
    pre = aggregate_pre(Tables["proofread_pre_synapses"].lazy()).rename(
        {"pre_pt_root_id": "pt_root_id"}
    )
    return (
        Tables["cell_info"]
        .lazy()
        .join(proofread, on="pt_root_id", how="inner")
        .join(post, on="pt_root_id", how="left")
        .join(pre, on="pt_root_id", how="left")
        .sort("pt_root_id")
    )


Tables.register(
    "proofread_cell_info",
    description=(
        "One row per proofread cell (`is_proofread` in `cell_selection`, the "
        "thalamic cells too): the columns of `cell_info`, the `cell_selection` "
        "flags, the `post_*` columns of `post_aggregation`, and the `pre_*`, "
        "`pre_to_exc_*` and `pre_to_inh_*` columns of `aggregate_pre` on "
        "`proofread_pre_synapses`. It replaces the old `proofread_info` of "
        "`stat_summaries.py`."
    ),
    build=build_proofread_cell_info,
    cache_params=lambda: _cache_keys(
        "cell_info", "cell_selection", "post_aggregation", "proofread_pre_synapses"
    ),
)
