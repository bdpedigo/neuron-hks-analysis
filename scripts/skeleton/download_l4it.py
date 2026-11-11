"""Download the skeleton files of the L4 IT cells of `scratch/r2_11_4P_bimodality.py`
to `L4IT_SKELETON_CACHE`, a folder of its own, so that it can be deleted in one step.

The cells are the L4 IT (`meso_type`) cells with `MIN_SYNAPSES` or more synapses and
a soma depth in the 1st to 99th percentile, with no cluster filter. Cells with a file
in `SKELETON_CACHE` already are skipped. The order is random, so the files that are
local at any time are a random sample. A rerun resumes.

    mkdir -p data/skeletons_l4it
    nohup uv run python scripts/skeleton/download_l4it.py > data/skeletons_l4it/download.log 2>&1 &
"""

# %%
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from analysis.census_inputs import post_cells
from analysis.products.skeleton import L4IT_SKELETON_CACHE, SKELETON_CACHE
from analysis.sources import SKELETON_PATH

MIN_SYNAPSES = 1000
MAX_WORKERS = 8

cells = post_cells(False).to_pandas().set_index("pt_root_id")
cells = cells[
    (cells["meso_type"].astype(str) == "L4IT") & (cells["post_total_synapses"] >= MIN_SYNAPSES)
]
depth = cells["pt_position_um_y"]
cells = cells[depth.between(depth.quantile(0.01), depth.quantile(0.99))]
local = {int(path.stem) for path in SKELETON_CACHE.glob("*.osy")}
local |= {int(path.stem) for path in L4IT_SKELETON_CACHE.glob("*.osy")}
todo = [root_id for root_id in cells.index if root_id not in local]
todo = list(np.random.default_rng(0).permutation(todo))
print(f"{len(cells)} L4 IT cells, {len(cells) - len(todo)} local, {len(todo)} to download")


def download(root_id: int) -> str | None:
    """Copy the file of `root_id`, and return the error if it fails."""
    path = L4IT_SKELETON_CACHE / f"{root_id}.osy"
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes((SKELETON_PATH / f"{root_id}.osy").read_bytes())
    except Exception as error:
        return f"{root_id}: {error!r}"
    tmp.rename(path)
    return None


L4IT_SKELETON_CACHE.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    errors = [
        error
        for error in tqdm(executor.map(download, todo), total=len(todo), mininterval=10)
        if error is not None
    ]
print(f"{len(errors)} failed")
for error in errors:
    print(error)
