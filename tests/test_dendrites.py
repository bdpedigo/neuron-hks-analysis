import numpy as np
import pytest

from analysis.dendrites import FRAMES, frame_points
from analysis.products.skeleton import dendrite_mask

# A soma in the column (nm) and points within 300 µm of it.
SOMA = np.array([700_000.0, 600_000.0, 800_000.0])
POINTS = SOMA + np.random.default_rng(0).uniform(-300_000, 300_000, size=(50, 3))


def test_affine_frame_is_rigid_in_um():
    xyz = frame_points(POINTS, SOMA, "affine")
    native = np.linalg.norm(POINTS[:, None] - POINTS[None], axis=-1) / 1000
    framed = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    assert np.allclose(native, framed)


@pytest.mark.parametrize("frame", FRAMES)
def test_soma_is_on_its_own_axis(frame):
    xyz = frame_points(np.vstack([SOMA, POINTS]), SOMA, frame)
    affine = frame_points(SOMA[None], SOMA, "affine")[0]
    # The soma keeps its depth (within 1 µm of the affine depth).
    assert abs(xyz[0, 1] - affine[1]) < 1
    if frame == "streamline":
        # x and z are radial around the streamline through the soma.
        assert np.allclose(xyz[0, [0, 2]], 0, atol=1e-6)


def test_unknown_frame_raises():
    with pytest.raises(ValueError):
        frame_points(POINTS, SOMA, "pia")


class _Tree:
    """The parts of an ossify skeleton that `dendrite_mask` reads."""

    def __init__(self, parent, is_axon):
        self.parent_node_array = np.asarray(parent)
        self._is_axon = np.asarray(is_axon)
        self.root_positional = int(np.flatnonzero(self.parent_node_array < 0)[0])

    def get_feature(self, name):
        assert name == "is_axon"
        return self._is_axon


def test_dendrite_mask_keeps_axon_on_root_paths():
    # 0 is the root. 1-2 is a stretch of axon with dendrite 3 below it. 4-5 is an axon
    # with no dendrite below it.
    tree = _Tree(parent=[-1, 0, 1, 2, 0, 4], is_axon=[0, 1, 1, 0, 1, 1])
    assert dendrite_mask(tree).tolist() == [True, True, True, True, False, False]
