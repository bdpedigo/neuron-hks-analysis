import pandas as pd

from analysis.products.cell_info import _as_bool
from analysis.products.cell_selection import (
    COLUMN_SOURCE,
    column_roots,
    proofread_roots,
    select_cells,
    thalamic_roots,
)


def _cell_info():
    return pd.DataFrame(
        {
            "cell_type_source": [COLUMN_SOURCE, None, None, COLUMN_SOURCE, COLUMN_SOURCE, None],
            "cell_type": ["L4a", "TH", "TH", "L5a", "L5a", "TH"],
            "broad_type": ["excitatory", "thalamic", "thalamic", "excitatory", "excitatory", "thalamic"],
            "axon_cleaned": [True, True, False, False, True, True],
            "axon_strategy": [
                "partially_extended",
                "fully_extended",
                "fully_extended",
                "fully_extended",
                "partially_extended",
                "partially_extended",
            ],
        },
        index=pd.Index([1, 2, 3, 4, 5, 6], name="pt_root_id"),
    )


def test_selection_rules():
    # Root 6 is a thalamic cell with a partially extended axon, so it is out. The
    # column cells keep any strategy.
    sel = select_cells(_cell_info())
    assert list(proofread_roots(sel)) == [1, 2, 5]
    assert list(column_roots(sel)) == [1, 5]
    assert list(thalamic_roots(sel)) == [2]


def test_duplicate_roots_dropped():
    info = _cell_info()
    info.index = pd.Index([1, 2, 3, 4, 1, 6], name="pt_root_id")
    assert list(proofread_roots(select_cells(info))) == [2]


def test_as_bool_reads_both_encodings():
    values = pd.Series(["t", "f", True, False, None])
    assert list(_as_bool(values)) == [True, False, True, False, False]
