# %%
"""Compare the paper's H01 run (`absolute-solo-yak`) with the deployment run, on
the H01 retrain cohort only, so the validation cohort stays held out.

Checks:

- 0a. Same mesh: the paper's per-vertex label array and the deployment's
  `vertex_domains` must have one entry per vertex of the same mesh. The paper
  dropped components under 200 vertices and the deployment drops components under
  10, so a vertex with no domain in the deployment must also have no domain in the
  paper.
- 0b. Same nucleus point: the paper's v1054 `nucleus` rows against
  `h01_root_nuclei`, the snapshot that the deployment read.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import polars as pl

from analysis import Tables
from analysis.products.deployment import latest_attempts, scan_roots
from analysis.products.paper_h01 import PAPER_H01_PATH, load_paper_root, paper_nuclei
from analysis.sources import H01_DATASTACK, cutoff, deployment_path

retrain_roots = (
    Tables["h01_cells"].lazy().filter(pl.col("cohort") == "retrain").collect()["root_id"].to_list()
)
print(f"{len(retrain_roots)} retrain roots")

# %%
# 0a. Same mesh.


def vertex_domains(root_id: int) -> pl.DataFrame:
    (lf,) = latest_attempts(
        [
            scan_roots(
                deployment_path("vertex_domains", H01_DATASTACK),
                [root_id],
                as_of=cutoff("features"),
            )
        ]
    )
    return lf.select("mesh_index", "domain_id").collect()


def mesh_check(root_id: int) -> dict:
    paper = load_paper_root(root_id)
    new = vertex_domains(root_id).sort("mesh_index")
    n_paper, n_new = len(paper.labels), len(new)
    row = dict(root_id=root_id, n_vertices_paper=n_paper, n_vertices_new=n_new)
    if n_paper != n_new or not np.array_equal(new["mesh_index"].to_numpy(), np.arange(n_new)):
        return row | dict(same_count=False)
    paper_none = paper.labels == -1
    new_none = new["domain_id"].is_null().to_numpy()
    return row | dict(
        same_count=True,
        frac_none_paper=paper_none.mean(),
        frac_none_new=new_none.mean(),
        # Expected 0: the deployment's cutoff (10) is below the paper's (200).
        n_none_new_only=int((new_none & ~paper_none).sum()),
    )


with ThreadPoolExecutor(8) as pool:
    mesh = pd.DataFrame(pool.map(mesh_check, retrain_roots))
mesh.to_parquet(PAPER_H01_PATH / "mesh_check.parquet")
print(mesh.describe().T)
print(f"same vertex count: {mesh['same_count'].sum()} of {len(mesh)}")
print(f"roots with a no-domain vertex only in the deployment: {(mesh['n_none_new_only'] > 0).sum()}")

# %%
# 0b. Same nucleus point.

paper_nuc = paper_nuclei().query("root_id in @retrain_roots")
paper_counts = paper_nuc.groupby("root_id").size().reindex(retrain_roots, fill_value=0)
print("paper nucleus rows per root:", paper_counts.value_counts().to_dict())

new_nuc = Tables["h01_root_nuclei"].lazy().filter(pl.col("root_id").is_in(retrain_roots)).collect().to_pandas()
print("snapshot nucleus rows per root:", new_nuc.groupby("root_id").size().value_counts().to_dict())

nuc = new_nuc.merge(paper_nuc, on="root_id", suffixes=("_new", "_paper"))
nuc["same_id"] = nuc["nucleus_id_new"] == nuc["nucleus_id_paper"]
nuc["shift_nm"] = np.linalg.norm(
    nuc[[f"nucleus_{c}_new" for c in "xyz"]].to_numpy()
    - nuc[[f"nucleus_{c}_paper" for c in "xyz"]].to_numpy(),
    axis=1,
)
# The closest paper row per root, so a root with several paper rows is matched
# to the one the snapshot kept.
best = nuc.sort_values("shift_nm").groupby("root_id").head(1)
print(f"roots matched: {len(best)} of {len(retrain_roots)}")
print(f"same nucleus id: {best['same_id'].sum()}")
print(best["shift_nm"].describe())
print("roots with a shift over 1 um:")
print(best.query("shift_nm > 1000")[["root_id", "nucleus_id_new", "nucleus_id_paper", "shift_nm"]])

# %%
