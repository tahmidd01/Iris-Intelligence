import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import FEATURES, SEED


def measurement_groups(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.factorize(pd.MultiIndex.from_frame(frame[FEATURES]))[0], index=frame.index)


def split_data(frame: pd.DataFrame, seed: int = SEED, test_size: float = .2):
    """Stratify unique measurement groups, retaining every original observation."""
    groups = measurement_groups(frame)
    representatives = frame.assign(group=groups).drop_duplicates("group")
    train_groups, test_groups = train_test_split(
        representatives.group, random_state=seed, test_size=test_size, stratify=representatives.Species
    )
    return frame.index[groups.isin(train_groups)], frame.index[groups.isin(test_groups)]


def make_pipeline(estimator, scaled: bool) -> Pipeline:
    return Pipeline([("scale", StandardScaler() if scaled else "passthrough"), ("model", estimator)])
