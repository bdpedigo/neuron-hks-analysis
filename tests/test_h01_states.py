import json

import pandas as pd

from analysis.products.h01 import TAG_CATEGORIES, read_h01_annotations

TRANSFORM = {
    "outputDimensions": {"x": [8e-09, "m"], "y": [8e-09, "m"], "z": [3.3e-08, "m"]},
    "inputDimensions": {"0": [1e-09, "m"], "1": [1e-09, "m"], "2": [1e-09, "m"]},
}
ROOT_ID = 864691132164989148
# Each point and the index of its tag in TAG_CATEGORIES (spine, shaft, soma).
POINTS = [([1930120, 1512136, 15807], 2), ([1919536, 1524576, 26037], 1), ([1925176, 1529584, 27852], 0)]


def annotation_layer(properties, props_of):
    """An annotation layer in the shape of the exported H01 states."""
    return {
        "type": "annotation",
        "source": {"url": "local://annotations", "transform": TRANSFORM},
        "annotationProperties": properties,
        "linkedSegmentationLayer": {"segments": "segmentation"},
        "name": "annotation",
        "annotations": [
            {
                "point": point,
                "type": "point",
                "id": f"a{i}",
                "segments": [[str(ROOT_ID)]],
                "props": props_of(tag),
            }
            for i, (point, tag) in enumerate(POINTS)
        ],
    }


def legacy_state():
    # The first round: uint8 properties that carry a `tag` key.
    properties = [
        {"id": f"tag{i}", "type": "uint8", "tag": tag} for i, tag in enumerate(TAG_CATEGORIES)
    ]
    return annotation_layer(
        properties, lambda tag: [int(i == tag) for i in range(len(TAG_CATEGORIES))]
    )


def bool_state():
    # Current Spelunker: one bool property per tag.
    properties = [{"id": tag, "type": "bool"} for tag in TAG_CATEGORIES]
    return annotation_layer(properties, lambda tag: [i == tag for i in range(len(TAG_CATEGORIES))])


def write_states(path, layers):
    (path / "states").mkdir()
    rows = []
    for state_id, layer in layers.items():
        state = {
            "dimensions": TRANSFORM["outputDimensions"],
            "layers": [{"type": "segmentation", "source": "graphene://x", "name": "segmentation"}, layer],
        }
        (path / "states" / f"{state_id}.json").write_text(json.dumps(state))
        rows.append({"state_id": state_id, "cohort": "retrain"})
    pd.DataFrame(rows).to_csv(path / "states.csv", index=False)


def test_both_tag_schemes_read_the_same(tmp_path):
    write_states(tmp_path, {1: legacy_state(), 2: bool_state()})
    df = read_h01_annotations(tmp_path).to_pandas()
    assert len(df) == 2 * len(POINTS)
    assert (df["state_root_id"] == ROOT_ID).all()
    for state_id in [1, 2]:
        state = df[df["state_id"] == state_id].reset_index(drop=True)
        assert state[["x", "y", "z"]].values.tolist() == [point for point, _ in POINTS]
        tags = state[TAG_CATEGORIES].values
        assert tags.sum(axis=1).tolist() == [1, 1, 1]
        assert [TAG_CATEGORIES[i] for i in tags.argmax(axis=1)] == ["soma", "shaft", "spine"]
