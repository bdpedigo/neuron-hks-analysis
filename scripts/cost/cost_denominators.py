"""The cost of the pipeline per meter of dendrite and per synapse, and its
extrapolation to 1 mm³, the mouse cortex and the mouse brain (reviewer R2.4).

Our side:
- Dendrite cable: `dendrite_cable_length_um` of `cellpax_features` per neuron. A
  neuron with no cable there gets its input synapses times the cable per synapse of
  its cell type (about 1% of the neurons).
- Cost: the dollars per step-hour of the full-run cluster session, times the
  step-hours that wrote the census tables. Both come from the deployment log table.
- The MICrONS densities: every synapse with a post root, counted in cubes of
  `BIN_UM`. The tissue is the cubes that hold a synapse, with the enclosed empty
  cubes filled in (`FILL_HOLES`). The inner cubes are the tissue cubes whose six
  face neighbors are tissue too, or the tissue cubes further than `INNER_MARGIN_UM`
  from an edge cube (`INNER_BY_DISTANCE`). Both leave out the partly filled cubes
  at the edge. A density is the synapses or the neuron nuclei in the inner cubes
  over their volume.

The extrapolation is a table with one row per estimate: a quantity (synapses,
dendrite cable or neurons), times our cost per unit of it, times an assumed density
of it (from MICrONS for synapses and neurons, or from the literature). The columns
are the target volumes. A target with no source is a placeholder: its column says
so, and it is not glued.

Scope: full run. Run it with `--scratch` until `post_aggregation` exists (see
`analysis.census_inputs`). The synapse binning reads all 337M synapses in chunks of
post-root buckets.

    uv run python scripts/cost/cost_denominators.py [--scratch]
"""

# %%
import sys

import numpy as np
import polars as pl
import pyvista as pv
from scipy import ndimage

from analysis import FIG_PATH, Tables
from analysis.census_inputs import glue, post_cells
from analysis.products.deployment_logs import WIDE_RUNS, dollars_per_step_hour
from analysis.sources import MICRONS_SYNAPSE_RESOLUTION
from analysis.stash import Stash

SCRATCH = "--scratch" in sys.argv
OUT_PATH = FIG_PATH / "cost" / ("_scratch" if SCRATCH else "")
CABLE = "dendrite_cable_length_um"

# The side of the cubes that the synapses are counted in, in µm.
BIN_UM = 10
# The synapse table is bucketed on the post root. Bin this many buckets at a time.
BUCKETS_PER_CHUNK = 32
# Fill the empty cubes that the tissue encloses (somata, vessels) before the inner
# cubes are found. With False, the tissue is only the cubes that hold a synapse.
FILL_HOLES = True
# With True, the inner cubes are the tissue cubes further than `INNER_MARGIN_UM` from
# an edge cube (a tissue cube with a face neighbor that is not tissue). With False,
# they are the tissue cubes whose six face neighbors are tissue too.
INNER_BY_DISTANCE = False
INNER_MARGIN_UM = 30
# Show the tissue cubes in an interactive 3D view, inner or not.
SHOW_CUBES = False

BOOK = "Braitenberg & Schüz 1998"
# The quantities that the cost is extrapolated by, and their units.
QUANTITIES = {"synapses": "synapse", "dendrite cable": "m", "neurons": "neuron"}
# Densities in mouse cortex, as (quantity, value per mm³).
BOOK_DENSITIES = [
    # Ch. 5 "Density of Synapses": the grand average of the sample.
    ("synapses", 7.2e8),
    # Ch. 11 "Density of Dendrites": 276 to 460 m from Golgi stains (3 to 5 mm per
    # neuron), 456 m from electron micrographs, 0.4 km in the Fig. 80 summary. The
    # value here is the middle of the Golgi range.
    ("dendrite cable", 368.0),
    # Ch. 4 "Density of Neurons": the average of the sample.
    ("neurons", 9.2e4),
]
# Target volumes in mm³, as (name, mm³, source).
TARGETS = [
    ("1 mm³", 1.0, "definition"),
    # Ch. 4 and Fig. 80: 87 mm³ of iso- and allocortex in one hemisphere.
    ("mouse cortex", 2 * 87.0, f"{BOOK}, ch. 4, Fig. 80"),
    # An approximate whole mouse brain, chosen by the user with no citation yet
    # (ssa-backlog TASK-78). The cortical densities stand in for the whole brain.
    ("mouse brain", 500.0, "approximate, no citation"),
]


# %% MICRONS SYNAPSE CUBES
def bin_synapses() -> pl.DataFrame:
    """The number of synapses with a post root in each cube, keyed by the cube's
    integer position (`x`, `y`, `z`)."""
    synapses = Tables["microns_synapses"].lazy().filter(pl.col("post_pt_root_id") != 0)
    buckets = (
        synapses.select(pl.col("post_pt_root_id_bucket").unique())
        .collect()
        .to_series()
        .sort()
        .to_list()
    )
    cube = [
        (pl.col(f"ctr_pt_position_{axis}").cast(pl.Float64) * nm / 1000 // BIN_UM).cast(
            pl.Int32
        )
        for axis, nm in zip("xyz", MICRONS_SYNAPSE_RESOLUTION)
    ]
    chunks = []
    for start in range(0, len(buckets), BUCKETS_PER_CHUNK):
        chunk = buckets[start : start + BUCKETS_PER_CHUNK]
        chunks.append(
            synapses.filter(pl.col("post_pt_root_id_bucket").is_in(chunk))
            .select(*(c.alias(a) for c, a in zip(cube, "xyz")))
            .group_by("x", "y", "z")
            .len("n_synapses")
            .collect(engine="streaming")
        )
    return pl.concat(chunks).group_by("x", "y", "z").agg(pl.col("n_synapses").sum())


cubes_stash = Stash(
    "cost/synapse_cubes", inputs=[Tables["microns_synapses"]], bin_um=BIN_UM
)
if not cubes_stash.exists():
    cubes_stash.save(bin_synapses())
cubes = cubes_stash.load()

# %% INNER CUBES
# A grid over the bounding box of the cubes, with the synapse count of each cube.
origin = np.array([cubes[axis].min() for axis in "xyz"])
shape = tuple(cubes[axis].max() - o + 1 for axis, o in zip("xyz", origin))
counts = np.zeros(shape, dtype=np.int64)
counts[tuple(cubes[axis].to_numpy() - o for axis, o in zip("xyz", origin))] = cubes[
    "n_synapses"
].to_numpy()
tissue = counts > 0
n_occupied = tissue.sum()
if FILL_HOLES:
    tissue = ndimage.binary_fill_holes(tissue)
# The inner cubes: tissue cubes whose six face neighbors are tissue too.
inner = ndimage.binary_erosion(tissue, ndimage.generate_binary_structure(3, 1))
if INNER_BY_DISTANCE:
    # The distance in µm from each cube center to the nearest edge cube center.
    edge = tissue & ~inner
    inner = tissue & (
        ndimage.distance_transform_edt(~edge, sampling=BIN_UM) > INNER_MARGIN_UM
    )
cube_mm3 = (BIN_UM / 1000) ** 3
inner_mm3 = inner.sum() * cube_mm3
volume_synapses = cubes["n_synapses"].sum()
print(f"{volume_synapses / 1e6:.1f} million synapses with a post root")
print(
    f"{n_occupied:,} occupied cubes, {tissue.sum() - n_occupied:,} filled, "
    f"{inner.sum():,} inner ({inner_mm3:.3f} mm³)"
)

# %% CUBE VIEW
# The tissue cubes in µm, colored by inner or not. Drag the plane to see inside.
if SHOW_CUBES:
    grid = pv.ImageData(
        dimensions=np.array(shape) + 1, spacing=(BIN_UM,) * 3, origin=origin * BIN_UM
    )
    # VTK orders the cells with x fastest.
    grid.cell_data["tissue"] = tissue.ravel(order="F").astype(np.uint8)
    grid.cell_data["inner"] = inner.ravel(order="F").astype(np.uint8)
    plotter = pv.Plotter()
    plotter.add_mesh_clip_plane(
        grid.threshold(0.5, scalars="tissue"),
        scalars="inner",
        crinkle=True,
        normal="-z",
        cmap=["tomato", "lightgray"],
        show_scalar_bar=False,
    )
    plotter.add_legend([("edge", "tomato"), ("inner", "lightgray")])
    plotter.show_axes()
    plotter.show()

# The neuron nuclei, in the cube that holds each one. The positions are in nm.
nuclei = (
    Tables["cell_info"]
    .lazy()
    .filter(pl.col("broad_type").is_in(["excitatory", "inhibitory"]))
    .select(
        (pl.col(f"pt_position_{axis}") / 1000 // BIN_UM).cast(pl.Int64)
        for axis in "xyz"
    )
    .collect()
    .to_numpy()
    - origin
)
nuclei = nuclei[((nuclei >= 0) & (nuclei < shape)).all(axis=1)]
inner_nuclei = inner[tuple(nuclei.T)].sum()

# The local densities, away from the partly filled edge cubes.
synapses_per_mm3 = counts[inner].sum() / inner_mm3
neurons_per_mm3 = inner_nuclei / inner_mm3
print(f"{synapses_per_mm3:.3g} synapses per mm³ in the inner cubes")
print(
    f"{inner_nuclei:,} neuron nuclei in the inner cubes, {neurons_per_mm3:.3g} per mm³"
)

# %% DENDRITE CABLE
cells = post_cells(SCRATCH)
cells = cells.join(
    Tables["cellpax_features"]
    .lazy()
    .select(pl.col("root_id").alias("pt_root_id"), CABLE)
    .collect(),
    on="pt_root_id",
    how="left",
)
covered = cells.filter(pl.col(CABLE).is_not_null())
pooled_rate = covered[CABLE].sum() / covered["post_total_synapses"].sum()
type_rates = covered.group_by("cell_type").agg(
    (pl.col(CABLE).sum() / pl.col("post_total_synapses").sum()).alias("rate")
)
cells = cells.join(type_rates, on="cell_type", how="left").with_columns(
    imputed=pl.col(CABLE).is_null(),
    cable_um=pl.coalesce(
        CABLE, pl.col("post_total_synapses") * pl.col("rate").fill_null(pooled_rate)
    ),
)
total_m = cells["cable_um"].sum() / 1e6
imputed_m = cells.filter("imputed")["cable_um"].sum() / 1e6
print(f"{len(cells):,} neurons, {cells['imputed'].sum():,} with an imputed cable")
print(f"dendrite cable {total_m:.1f} m, imputed {imputed_m:.2f} m")
print(f"pooled cable per synapse {pooled_rate:.3f} um")


# %% STEP-HOURS FROM THE LOG TABLE
runs = Tables["session_root_hours"].collect()
written = (
    Tables["census_step_hours"]
    .collect()
    .join(cells.select(pl.col("pt_root_id").alias("root_id")), on="root_id", how="semi")
)

session_step_hours = runs["elapsed_s"].sum() / 3600
step_hour_dollars = dollars_per_step_hour()
census_step_hours = written["elapsed_s"].sum() / 3600
census_dollars = census_step_hours * step_hour_dollars
print(
    f"session {session_step_hours:,.0f} step-hours, ${step_hour_dollars:.4f} per step-hour"
)
print(
    written.group_by("step")
    .agg((pl.col("elapsed_s").sum() / 3600).round(0).alias("hours"))
    .sort("hours", descending=True)
)
print(f"census {census_step_hours:,.0f} step-hours, ${census_dollars:,.0f}")

# A check on scaling by synapses: the smaller pieces cost more per synapse, because
# each root has a fixed cost (the mesh fetch and the writes).
neuron_roots = Tables["cell_info"].lazy().select("pt_root_id").collect()["pt_root_id"]
fragments = runs.filter(
    pl.col("run_id").is_in(WIDE_RUNS) & ~pl.col("root_id").is_in(neuron_roots.implode())
)
print(
    f"wide-root runs, roots with no neuron nucleus: {fragments.height:,} root runs, "
    f"{fragments['elapsed_s'].mean():.0f} s each"
)

# %% PER-UNIT COSTS
# NOTE: the known synapses, the same total that whole_dataset.py glues.
n_synapses = cells["post_total_known_synapses"].sum()
# The units of each quantity that the census ran.
units_run = {"synapses": n_synapses, "dendrite cable": total_m, "neurons": len(cells)}
dollars_per_unit = {q: census_dollars / n for q, n in units_run.items()}
print(f"${dollars_per_unit['dendrite cable']:.2f} per m of dendrite")
print(f"${dollars_per_unit['neurons'] * 1000:.2f} per thousand neurons")
print(f"${dollars_per_unit['synapses'] * 1e6:.2f} per million synapses")
print(
    f"{volume_synapses / 1e6:.1f} million synapses with a post root, "
    f"{n_synapses / volume_synapses:.1%} in the census, "
    f"${volume_synapses * dollars_per_unit['synapses']:,.0f} for all of them"
)

# %% EXTRAPOLATION TABLE
# One row per estimate: a quantity, times our cost per unit of it, times an assumed
# density of it.
estimates = pl.DataFrame(
    [("MICrONS", "synapses", synapses_per_mm3), ("MICrONS", "neurons", neurons_per_mm3)]
    + [(BOOK, q, value) for q, value in BOOK_DENSITIES],
    schema=["density_source", "quantity", "density_per_mm3"],
    orient="row",
).select(
    pl.col("quantity"),
    pl.col("quantity").replace_strict(QUANTITIES).alias("unit"),
    pl.col("quantity")
    .replace_strict({q: float(n) for q, n in units_run.items()})
    .alias("units_run"),
    pl.col("quantity").replace_strict(dollars_per_unit).alias("unit_cost"),
    "density_per_mm3",
    "density_source",
)
# A target with no source is a placeholder, and its column says so.
target_columns = {
    name if source else f"{name} (placeholder)": mm3 for name, mm3, source in TARGETS
}
table = estimates.with_columns(
    (pl.col("unit_cost") * pl.col("density_per_mm3") * mm3).alias(name)
    for name, mm3 in target_columns.items()
).sort(pl.col("quantity").cast(pl.Enum(list(QUANTITIES))), "density_source")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=50):
    print(table.drop("unit"))
OUT_PATH.mkdir(parents=True, exist_ok=True)
table.write_csv(OUT_PATH / "cost_extrapolation.csv")

# %%
glue(
    SCRATCH,
    cost_dendrite_cable_m=total_m,
    cost_census_dollars=census_dollars,
    cost_volume_dollars=volume_synapses * dollars_per_unit["synapses"],
    format="{:,.0f}",
)
glue(
    SCRATCH,
    cost_dollars_per_m_dendrite=dollars_per_unit["dendrite cable"],
    cost_dollars_per_million_synapses=dollars_per_unit["synapses"] * 1e6,
    format="{:.2f}",
)
glue(
    SCRATCH,
    cost_imputed_cable_percent=100 * imputed_m / total_m,
    cost_volume_synapses_millions=volume_synapses / 1e6,
    format="{:.1f}",
)
# The best estimate: our cost per synapse times the MICrONS synapse density.
best = table.row(
    by_predicate=(pl.col("quantity") == "synapses")
    & (pl.col("density_source") == "MICrONS"),
    named=True,
)
# The best estimate of each target with a source, and the range over every estimate.
for name, _, source in TARGETS:
    if source is None:
        continue
    key = name.replace(" ", "_").replace("³", "3")
    glue(
        SCRATCH,
        **{
            f"cost_{key}_dollars_best": best[name],
            f"cost_{key}_dollars_low": table[name].min(),
            f"cost_{key}_dollars_high": table[name].max(),
        },
        format="{:,.0f}",
    )
