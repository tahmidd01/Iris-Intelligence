from pathlib import Path
from collections.abc import Mapping, Sequence
import joblib
import numpy as np
import pandas as pd
from .config import FEATURES, ROOT
from .utils import read_json, sha256


def load_bundle(directory: Path = ROOT / "models"):
    """Load only a trusted locally generated artifact; joblib can execute code."""
    metadata = read_json(directory / "metadata.json")
    path = directory / "iris_pipeline.joblib"
    if sha256(path) != metadata["model_sha256"]:
        raise ValueError("Model checksum mismatch. Retrain with python -m src.train.")
    if metadata["features"] != FEATURES:
        raise ValueError("Artifact feature schema does not match this application.")
    model = joblib.load(path)
    first_fitted_step = next(step for _, step in model.steps if step != "passthrough")
    if list(first_fitted_step.feature_names_in_) != FEATURES or list(model.classes_) != metadata["labels"]:
        raise ValueError("Model and metadata schemas disagree.")
    return model, metadata


def input_frame(values: Mapping | Sequence) -> pd.DataFrame:
    if isinstance(values, Mapping):
        if set(values) != set(FEATURES):
            raise ValueError(f"Supply exactly these four features: {FEATURES}")
        values = [values[f] for f in FEATURES]
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__") or len(values) != 4 or any(isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError("Supply four numeric measurements in cm.")
    try:
        numbers = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Measurements must be numeric.") from exc
    if numbers.shape != (4,) or not np.isfinite(numbers).all() or (numbers <= 0).any():
        raise ValueError("Measurements must be finite and greater than zero.")
    return pd.DataFrame([numbers], columns=FEATURES)


def predict(model, metadata: dict, values: Mapping | Sequence) -> dict:
    frame = input_frame(values)
    species = str(model.predict(frame)[0])
    outside = [f for f in FEATURES if not metadata["training_ranges"][f]["min"] <= frame.iloc[0][f] <= metadata["training_ranges"][f]["max"]]
    probability = dict(zip(model.classes_, model.predict_proba(frame)[0].astype(float))) if hasattr(model, "predict_proba") else {}
    return {"species": species, "probabilities": probability, "measurements": frame.iloc[0].to_dict(), "outside_training_range": outside}
