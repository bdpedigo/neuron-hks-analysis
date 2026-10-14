"""View the predictions of every model variant, with and without the H01 retrain
rows, on whole MICrONS cells. See `analysis.model_viewer` for the controls.

Pools:
- `validation` (default): the 132 validation cells. No model trained on them.
- `labeled`: the 6,009 training roots with domain features. Every model trained
  on their labeled synapses, so the predictions on those synapses are not held out.

    uv run python scripts/model/view_microns_models.py [--pool labeled] [--start N]
"""

import polars as pl

from analysis import Tables
from analysis.model_viewer import main
from analysis.products.sources import DATASTACK

POOLS = {
    "validation": lambda: Tables["validation_cells"].collect()["root_id"].to_list(),
    "labeled": lambda: Tables["labeled_synapse_features"]
    .lazy()
    .drop_nulls("domain_n_vertices")
    .select(pl.col("root_id").unique())
    .collect()["root_id"]
    .to_list(),
}

if __name__ == "__main__":
    main(DATASTACK, POOLS, __doc__.splitlines()[0])
