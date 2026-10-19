import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from src.config import FEATURES, LABELS
from src.predict import load_bundle, predict, input_frame
from src.preprocessing import split_data, measurement_groups, make_pipeline
from src.utils import read_json


def test_grouped_boundaries(iris_frame):
    train, test = split_data(iris_frame)
    groups = measurement_groups(iris_frame)
    assert set(groups.loc[train]).isdisjoint(groups.loc[test])
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(iris_frame.index)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    sub = iris_frame.loc[train]
    for a, b in cv.split(sub[FEATURES], sub.Species, groups.loc[train]):
        assert set(groups.loc[train].iloc[a]).isdisjoint(groups.loc[train].iloc[b])


def test_scaler_fit_on_training_only(iris_frame):
    train, test = split_data(iris_frame)
    model = make_pipeline(LogisticRegression(max_iter=1000), scaled=True)
    model.fit(iris_frame.loc[train, FEATURES], iris_frame.loc[train, "Species"])
    np.testing.assert_allclose(model.named_steps["scale"].mean_, iris_frame.loc[train, FEATURES].mean())
    original = model.named_steps["scale"].mean_.copy()
    model.predict(iris_frame.loc[test, FEATURES] * 10)
    np.testing.assert_array_equal(original, model.named_steps["scale"].mean_)


@pytest.mark.parametrize("bad", [None, 3, [], [1, 2, 3], [1, 2, 3, 4, 5], "1234", [0, 1, 1, 1], [-1, 2, 3, 4], [np.inf, 2, 3, 4], [np.nan, 2, 3, 4], ["x", 2, 3, 4], [True, 2, 3, 4], {"wrong": 3}])
def test_invalid_prediction_input(bad):
    with pytest.raises(ValueError):
        input_frame(bad)


def test_feature_ordering():
    values = dict(zip(FEATURES, [5.1, 3.5, 1.4, .2]))
    reversed_mapping = dict(reversed(list(values.items())))
    assert input_frame(values).equals(input_frame(reversed_mapping))
    assert list(input_frame(reversed_mapping)) == FEATURES


@pytest.mark.integration
def test_training_saving_loading_inference(trained_bundle):
    directory, result = trained_bundle
    model, metadata = load_bundle(directory / "models")
    prediction = predict(model, metadata, [5.1, 3.5, 1.4, .2])
    assert prediction["species"] == LABELS[0]
    assert set(prediction["probabilities"]) == set(LABELS)
    assert sum(prediction["probabilities"].values()) == pytest.approx(1)
    assert all(0 <= value <= 1 for value in prediction["probabilities"].values())
    assert metadata["features"] == FEATURES
    assert len(result["comparison"]) == 5
    assert result["metrics"]["test_rows"] == metadata["test_rows"]
    assert metadata["provenance"]["method"] == "local CSV; origin unverified"
    assert all("best_params" in row for row in result["comparison"])
    records = pd.read_csv(directory / "reports" / "metrics" / "test_predictions.csv")
    assert result["metrics"]["accuracy"] == accuracy_score(records.actual, records.predicted)
    assert result["metrics"]["macro_f1"] == pytest.approx(f1_score(records.actual, records.predicted, average="macro"))
    assert np.sum(result["metrics"]["confusion_matrix"]) == len(records)
    assert result["metrics"]["correct"] + result["metrics"]["incorrect"] == len(records)
    assert predict(model, metadata, [99, 3, 1.4, .2])["outside_training_range"] == [FEATURES[0]]
    # The final estimator's root sees training rows only (for trees).
    if hasattr(model.named_steps["model"], "tree_"):
        assert model.named_steps["model"].tree_.n_node_samples[0] == metadata["train_rows"]


def test_checksum_rejects_modified_model(trained_bundle, tmp_path):
    import shutil
    directory, _ = trained_bundle
    shutil.copytree(directory / "models", tmp_path / "models")
    with (tmp_path / "models" / "iris_pipeline.joblib").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path / "models")
