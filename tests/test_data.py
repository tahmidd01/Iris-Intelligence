import numpy as np
import pytest
from src.config import FEATURES, LABELS, DATA_PATH
from src.data_loader import load_data, validate_data, extract_features, inspection


def test_dataset_loading(iris_frame, tmp_path):
    csv = tmp_path / "iris.csv"
    iris_frame.to_csv(csv, index=False)
    loaded = load_data(csv)
    assert len(loaded) == 150
    assert set(loaded.Species) == set(LABELS)
    X, y = extract_features(loaded)
    assert list(X.columns) == FEATURES
    assert "Id" not in X
    assert y.name == "Species"


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.download"):
        load_data(tmp_path / "absent.csv")


@pytest.mark.parametrize("mutation", ["missing", "extra", "unknown_label", "missing_label", "duplicate_id", "conflicting", "empty"])
def test_schema_failures(iris_frame, mutation):
    frame = iris_frame.copy()
    if mutation == "missing":
        frame = frame.drop(columns=FEATURES[0])
    elif mutation == "extra":
        frame["unexpected"] = 1
    elif mutation == "unknown_label":
        frame.loc[0, "Species"] = "rose"
    elif mutation == "missing_label":
        frame = frame[frame.Species != LABELS[0]]
    elif mutation == "duplicate_id":
        frame.loc[1, "Id"] = 1
    elif mutation == "conflicting":
        frame.loc[1, FEATURES] = frame.loc[0, FEATURES]
        frame.loc[1, "Species"] = LABELS[1]
    else:
        frame = frame.iloc[0:0]
    with pytest.raises(ValueError):
        validate_data(frame)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1, 0, "not numeric"])
def test_bad_measurements(iris_frame, bad):
    iris_frame[FEATURES[0]] = iris_frame[FEATURES[0]].astype(object)
    iris_frame.loc[0, FEATURES[0]] = bad
    with pytest.raises(ValueError):
        validate_data(iris_frame)


def test_duplicate_observations_retained(iris_frame):
    before = len(iris_frame)
    clean = validate_data(iris_frame)
    assert len(clean) == before
    assert inspection(clean)["duplicate_measurement_rows"] >= 1


def test_actual_kaggle_csv():
    if not DATA_PATH.exists():
        pytest.skip("Kaggle CSV not downloaded")
    frame = load_data()
    assert frame.shape == (150, 6)
    assert inspection(frame)["class_counts"] == {label: 50 for label in LABELS}
