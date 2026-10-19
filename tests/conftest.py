import pandas as pd
import pytest
from sklearn.datasets import load_iris
from src.config import FEATURES, LABELS
from src.train import train


@pytest.fixture
def iris_frame():
    """Explicit sklearn fixture, never presented as Kaggle data."""
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=FEATURES)
    frame.insert(0, "Id", range(1, len(frame) + 1))
    frame["Species"] = [LABELS[i] for i in iris.target]
    return frame


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory):
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=FEATURES)
    frame["Species"] = [LABELS[i] for i in iris.target]
    directory = tmp_path_factory.mktemp("experiment")
    csv = directory / "fixture.csv"
    frame.to_csv(csv, index=False)
    results = train(csv, output=directory, quick=True, folds=3)
    return directory, results
