"""An optional, small disk cache for `load_root_cell`, for development.

    load = cached(load_root_cell, max_roots=5)
    cell = load(root_id, datastack)

Each root is one ossify `.osy` file at
`data/root_cache/{datastack}/{pin hash}/{root_id}.osy`, where the pin hash is that
of `root_pins(datastack)`. A folder keeps at most `max_roots` files, and the file
read longest ago goes first. With `max_roots >= 2`, the root on view and the root
that loads next are the two newest files, so neither is removed.

Nothing else imports this module's internals. To drop the cache, use
`load_root_cell` directly.
"""

import os
from pathlib import Path
from typing import Callable

from ossify import Cell, load_cell, save_cell

from ..io import DATA_PATH, param_hash
from .roots import check_not_held_out, root_pins

ROOT_CACHE_PATH = DATA_PATH / "root_cache"


def cached(
    load: Callable[..., Cell], max_roots: int = 5, folder: Path = ROOT_CACHE_PATH
) -> Callable[..., Cell]:
    """`load(root_id, datastack, **kwargs)`, read from and written to the cache."""
    if max_roots < 2:
        raise ValueError("max_roots must be at least 2: the root on view and the next one")

    def cached_load(root_id: int, datastack: str, **kwargs) -> Cell:
        # NOTE: the held-out check runs on a cache hit too, so a file written with
        # `allow_held_out=True` is not read without it.
        check_not_held_out(root_id, datastack, kwargs.get("allow_held_out", False))
        path = folder / datastack / param_hash(**root_pins(datastack)) / f"{root_id}.osy"
        if path.exists():
            os.utime(path)
            return load_cell(str(path))
        cell = load(root_id, datastack, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.tmp.osy")
        save_cell(cell, str(tmp), allow_overwrite=True)
        tmp.rename(path)
        _evict(path.parent, max_roots)
        return cell

    return cached_load


def _evict(folder: Path, max_roots: int) -> None:
    files = sorted(
        (p for p in folder.glob("*.osy") if ".tmp" not in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in files[max_roots:]:
        stale.unlink(missing_ok=True)
