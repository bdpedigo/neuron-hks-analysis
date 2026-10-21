"""View the predictions of every model variant, with and without the H01 retrain
rows, on whole H01 retrain cells. See `analysis.model_viewer` for the controls.

The `retrain` models trained on the labeled synapses of these cells, so their
predictions on those synapses are not held out. The H01 validation cohort is not
in the pool, and `load_root_cell` refuses its roots.

    uv run python scripts/h01/view_retrain_models.py [--start N] [--screenshot out.png]
"""

from analysis import Tables
from analysis.model_viewer import main
from analysis.sources import H01_DATASTACK

POOLS = {"retrain": lambda: Tables["h01_retrain_cells"].collect()["root_id"].to_list()}

if __name__ == "__main__":
    main(H01_DATASTACK, POOLS, __doc__.splitlines()[0])
