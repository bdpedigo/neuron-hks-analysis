"""The step times and mesh sizes of the full-run cluster session, from the
meshmash-deployment log tables.

The log tables have one row per root and step of each run, with `elapsed_s` and a
map of `metrics`. Each table here reads one snapshot of the logs, and its cache key
holds the Delta version of each, so a new log write gives a new cache.
"""

from functools import lru_cache

import polars as pl

from ..sources import BASE_CLOUD_PATH, DATASTACK, H01_DATASTACK
from .catalog import Tables
from .deployment import delta_table

# The full-run cluster session, 2026-10-28 10:00Z to the cluster delete (ssa-backlog
# TASK-49): Spot workers $329 and the default pool $1, closed at each pool delete.
SESSION_DOLLARS = 331.0
# Every run of that session, by datastack. It also ran the H01 run and the wide-root
# runs, so the dollars per step-hour hold some work outside the census.
SESSION_RUNS = {
    DATASTACK: [
        "run-20261028-095858-243c",
        "run-20261028-140637-c62c",
        "run-20261028-143147-ddc2",
        "run-20261028-190710-318d",
        "run-20261028-193551-476c",
    ],
    H01_DATASTACK: ["run-20261028-164729-6ba4"],
}
# The runs of the session on a wide root set (roots with no neuron nucleus too).
WIDE_RUNS = ["run-20261028-095858-243c", "run-20261028-140637-c62c"]
# The last full run. Its step hashes are the ones the census tables hold.
FINAL_RUN = "run-20261028-193551-476c"
LOG_COLUMNS = ["run_id", "root_id", "step", "step_hash", "outcome", "elapsed_s"]


def log_path(datastack: str) -> str:
    return str(BASE_CLOUD_PATH / "logs" / "v3" / datastack)


@lru_cache
def log_versions() -> dict[str, int]:
    # NOTE: one snapshot per process, the same one that `read_log` reads.
    return {ds: delta_table(log_path(ds)).version() for ds in SESSION_RUNS}


def read_log(datastack: str, columns: list[str] = LOG_COLUMNS) -> pl.DataFrame:
    table = delta_table(log_path(datastack)).to_pyarrow_table(columns=columns)
    return pl.from_arrow(table)


def _cache_params() -> dict:
    return dict(
        session_runs=SESSION_RUNS, final_run=FINAL_RUN, log_versions=log_versions()
    )


def build_session_root_hours() -> pl.LazyFrame:
    return pl.concat(
        read_log(datastack)
        .filter(pl.col("run_id").is_in(runs))
        .group_by("run_id", "root_id")
        .agg(pl.col("elapsed_s").sum())
        .with_columns(datastack=pl.lit(datastack))
        for datastack, runs in SESSION_RUNS.items()
    ).lazy()


Tables.register(
    "session_root_hours",
    description=(
        "One row per run and root of the full-run cluster session (`SESSION_RUNS`, "
        "the MICrONS and H01 runs), with the summed `elapsed_s` of its log rows and "
        "its `datastack`. Its sum is the step time that the session dollars bought."
    ),
    build=build_session_root_hours,
    cache_params=_cache_params,
)


def build_census_step_hours() -> pl.LazyFrame:
    rows = read_log(DATASTACK)
    final_hashes = (
        rows.filter(pl.col("run_id") == FINAL_RUN)
        .select("step", "step_hash")
        .unique()
        .drop_nulls()
    )
    written = rows.join(final_hashes, on=["step", "step_hash"], how="semi").filter(
        pl.col("outcome") == "ok"
    )
    loaders = rows.filter(pl.col("step_hash").is_null()).join(
        written.select("run_id", "root_id").unique(),
        on=["run_id", "root_id"],
        how="semi",
    )
    return (
        pl.concat([written, loaders])
        .group_by("root_id", "step")
        .agg(pl.col("elapsed_s").sum())
        .lazy()
    )


Tables.register(
    "census_step_hours",
    description=(
        "One row per MICrONS root and step, with the summed `elapsed_s` of the "
        "attempts that wrote the census tables: the ok rows at the step hashes of "
        "`FINAL_RUN`, in any run, and the loader rows (no hash) of the same run and "
        "root. It covers the test-cohort runs before the session too."
    ),
    build=build_census_step_hours,
    cache_params=_cache_params,
)


def build_root_mesh_vertices() -> pl.LazyFrame:
    rows = read_log(DATASTACK, columns=["root_id", "step", "outcome", "metrics"])
    n_vertices = (
        pl.col("metrics")
        .list.eval(
            pl.element()
            .filter(pl.element().struct.field("key") == "n_vertices")
            .struct.field("value")
        )
        .list.first()
    )
    return (
        rows.filter(pl.col("step") == "mesh", pl.col("outcome") == "ok")
        .group_by("root_id")
        .agg(n_vertices.max().alias("n_vertices"))
        .drop_nulls()
        .lazy()
    )


Tables.register(
    "root_mesh_vertices",
    description=(
        "One row per MICrONS root with an ok mesh load in the log, with "
        "`n_vertices`, the vertex count of the mesh that the pipeline read. Every "
        "run reads the same mesh at the pinned version, so any run gives the count."
    ),
    build=build_root_mesh_vertices,
    cache_params=_cache_params,
)


def dollars_per_step_hour() -> float:
    """The session dollars over the step-hours of `session_root_hours`. Idle slots
    and the time outside `elapsed_s` are in the dollars, so they raise the rate."""
    seconds = Tables["session_root_hours"].lazy().select(pl.col("elapsed_s").sum())
    return SESSION_DOLLARS / (seconds.collect().item() / 3600)
