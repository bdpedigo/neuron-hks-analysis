"""The ablation runs of condensed_spectral (ssa-backlog TASK-77): one run per arm,
each on the same 40 cells, with the tables and the run log under
`ABLATION_CLOUD_PATH`.

`ablation_runs` reads the time, memory and storage of each root from the log and
the Delta logs. `ablation_labeled_features` pulls the domain features of the
labeled synapses from each arm's tables, so the scoring reads no cloud tables.
"""

import shutil
from pathlib import Path

import polars as pl

from ..sources import (
    ABLATION_CLOUD_PATH,
    ABLATION_PINS,
    DATASTACK,
    ablation_path,
    cutoff,
    pin_params,
)
from . import labels  # noqa: F401 -- registers "labels" and "synapse_labels"
from .catalog import Tables
from .deployment import delta_table
from .features import write_synapse_features

# The run of each arm (meshmash-deployment jobs/strawman, the light phase).
ABLATION_RUNS = {
    "proposed": "run-20261103-182509-c242",
    "no_cluster": "run-20261103-182528-f043",
    "no_chunk": "run-20261103-182453-2e32",
    "no_simplify": "run-20261103-182543-cae7",
    "full_res_vertex": "run-20261103-182555-93b1",
}
ABLATION_TABLES = ["vertex_domains", "domain_features"]
# Arm tables read at once in the pull. Each reads one chunk of roots at a time.
MAX_WORKERS = 4


def ablation_log_path(datastack: str = DATASTACK) -> str:
    return str(ABLATION_CLOUD_PATH / "logs" / "v3" / datastack)


def _add_actions(arm: str, product: str) -> pl.DataFrame:
    return pl.from_arrow(
        delta_table(str(ablation_path(arm, product))).get_add_actions(flatten=True)
    )


def _runs_cache_params() -> dict:
    return dict(
        runs=ABLATION_RUNS,
        pins=ABLATION_PINS,
        log_version=delta_table(ablation_log_path()).version(),
        table_versions={
            f"{arm}/{product}": delta_table(str(ablation_path(arm, product))).version()
            for arm in ABLATION_PINS
            for product in ABLATION_TABLES
        },
    )


def root_bytes(arm: str, product: str) -> pl.DataFrame:
    """`root_id` and the summed file size of its latest attempt in an arm's table."""
    actions = _add_actions(arm, product)
    # NOTE: the size of a file counts for one root only if the file holds one
    # root and one attempt. The ablation tables have one file per root.
    mixed = actions.filter(
        (pl.col("min.root_id") != pl.col("max.root_id"))
        | (pl.col("min.attempt_id") != pl.col("max.attempt_id"))
    )
    if mixed.height:
        raise ValueError(
            f"{arm}/{product}: {mixed.height} files hold more than one attempt"
        )
    return (
        actions.filter(
            pl.col("min.attempt_id")
            == pl.col("min.attempt_id").max().over("min.root_id")
        )
        .group_by(root_id="min.root_id")
        .agg(pl.col("size_bytes").sum().alias(f"{product}_bytes"))
    )


def build_ablation_runs() -> pl.LazyFrame:
    arm_of = {run_id: arm for arm, run_id in ABLATION_RUNS.items()}
    log = pl.from_arrow(
        delta_table(ablation_log_path()).to_pyarrow_table(
            columns=["run_id", "root_id", "step", "outcome", "elapsed_s", "metrics"]
        )
    ).filter(
        pl.col("step") == "condensed_spectral", pl.col("run_id").is_in(list(arm_of))
    )
    if log.select("run_id", "root_id").is_duplicated().any():
        raise ValueError("a root has more than one condensed_spectral row in its run")
    metrics = (
        log.select("run_id", "root_id", "metrics")
        .explode("metrics")
        .unnest("metrics")
        .pivot(on="key", values="value", index=["run_id", "root_id"])
    )
    # NOTE: a metric name such as `rows.domain_features` becomes
    # `domain_features_rows`.
    metrics = metrics.rename(
        {
            column: f"{column.removeprefix('rows.')}_rows"
            for column in metrics.columns
            if column.startswith("rows.")
        }
    )
    runs = (
        log.select("run_id", "root_id", "outcome", wall_s="elapsed_s")
        .join(metrics, on=["run_id", "root_id"], how="left")
        .with_columns(arm=pl.col("run_id").replace_strict(arm_of))
    )
    frames = []
    for arm in ABLATION_RUNS:
        sizes = runs.filter(pl.col("arm") == arm)
        for product in ABLATION_TABLES:
            sizes = sizes.join(root_bytes(arm, product), on="root_id", how="left")
        frames.append(sizes)
    return pl.concat(frames).select("arm", pl.exclude("arm")).lazy()


Tables.register(
    "ablation_runs",
    description=(
        "One row per ablation arm and root, from the condensed_spectral row of the "
        "arm's run in the ablation log: `outcome`, `wall_s` (its `elapsed_s`), and "
        "the logged metrics (for example `pipeline_s`, `rss_peak_bytes`, "
        "`domain_features_rows`). `vertex_domains_bytes` and `domain_features_bytes` "
        "are the file sizes of the root's attempt in the arm's tables."
    ),
    build=build_ablation_runs,
    cache_params=_runs_cache_params,
)


def ablation_synapses() -> pl.DataFrame:
    """The synapse labels on the roots that every arm finished."""
    runs = Tables["ablation_runs"].collect()
    done = (
        runs.filter(pl.col("outcome") == "ok")
        .group_by("root_id")
        .len()
        .filter(pl.col("len") == len(ABLATION_RUNS))["root_id"]
    )
    return (
        Tables["synapse_labels"]
        .collect()
        .filter(pl.col("root_id").is_in(done.to_list()))
    )


def write_ablation_features(path: Path) -> None:
    """Each arm's synapse features into a part, then all parts into `path`, with
    `arm`. An interrupted pull resumes from the arms (and the chunks) it finished."""
    synapses = ablation_synapses()
    partial = path.with_name(path.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    for arm in ABLATION_PINS:
        part = partial / f"{arm}.parquet"
        if part.exists():
            continue
        print(
            f"{arm}: {synapses.height} synapses on {synapses['root_id'].n_unique()} roots"
        )
        write_synapse_features(
            synapses,
            part,
            as_of=cutoff("ablation"),
            max_workers=MAX_WORKERS,
            paths={product: ablation_path(arm, product) for product in ABLATION_TABLES},
        )
    tmp = path.with_name(path.name + ".tmp")
    pl.concat(
        [
            pl.scan_parquet(partial / f"{arm}.parquet").with_columns(arm=pl.lit(arm))
            for arm in ABLATION_PINS
        ],
        how="diagonal_relaxed",
    ).sink_parquet(tmp, compression="snappy")
    tmp.rename(path)
    shutil.rmtree(partial)


def estimate_ablation_reads() -> str:
    lines = [
        "The build reads these ablation tables (every column of the labeled roots):"
    ]
    for arm in ABLATION_PINS:
        for product in ABLATION_TABLES:
            actions = _add_actions(arm, product)
            lines.append(
                f"  {arm:>16} {product:>16}: {actions['num_records'].sum() / 1e6:.1f}M rows, "
                f"{actions['size_bytes'].sum() / 1e9:.1f} GB"
            )
    return "\n".join(lines)


Tables.register(
    "ablation_labeled_features",
    description=(
        "One row per ablation arm and labeled synapse (from `synapse_labels`) on the "
        "roots that every arm finished, with the spectral features of the mesh "
        "domain that it maps to in that arm's tables. The synapse mapping is the "
        "pinned deployment table. Read at the `ablation` cutoff. The rows of "
        "`labeled_synapse_features`, once per arm."
    ),
    materialize=write_ablation_features,
    cache_params=lambda: {
        "labels": Tables["labels"].cache_key,
        "runs": ABLATION_RUNS,
        "pins": ABLATION_PINS,
        **pin_params("synapse_mapping", as_of=cutoff("ablation")),
    },
    estimate=estimate_ablation_reads,
)
