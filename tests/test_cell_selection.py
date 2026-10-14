import pandas as pd

from analysis.io import _as_bool
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
            "cell_type_source": [COLUMN_SOURCE, None, None, COLUMN_SOURCE, COLUMN_SOURCE],
            "cell_type": ["L4a", "TH", "TH", "L5a", "L5a"],
            "broad_type": ["excitatory", "thalamic", "thalamic", "excitatory", "excitatory"],
            "axon_cleaned": [True, True, False, False, True],
        },
        index=pd.Index([1, 2, 3, 4, 5, ], name="pt_root_id"),
    )


def test_selection_rules():
    sel = select_cells(_cell_info())
    assert list(proofread_roots(sel)) == [1, 2, 5]
    assert list(column_roots(sel)) == [1, 5]
    assert list(thalamic_roots(sel)) == [2]


def test_duplicate_roots_dropped():
    info = _cell_info()
    info.index = pd.Index([1, 2, 3, 4, 1], name="pt_root_id")
    assert list(proofread_roots(select_cells(info))) == [2]


def test_as_bool_reads_both_encodings():
    values = pd.Series(["t", "f", True, False, None])
    assert list(_as_bool(values)) == [True, False, True, False, False]
