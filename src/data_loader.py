"""Strict CSV contract; no imputation, row removal, or silent fallback."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATA_PATH, FEATURES, LABELS


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURES + ["Species"])
    missing = required - set(frame.columns)
    extra = set(frame.columns) - required - {"Id"}
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {sorted(missing)}; unexpected: {sorted(extra)}")
    if frame.empty:
        raise ValueError("The dataset contains no observations.")
    result = frame.copy()
    for feature in FEATURES:
        try:
            result[feature] = pd.to_numeric(result[feature], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{feature} must contain numeric measurements in cm.") from exc
    values = result[FEATURES].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Measurements must be finite, non-missing, and greater than zero.")
    if result.Species.isna().any() or set(result.Species.unique()) != set(LABELS):
        raise ValueError(f"Species must contain exactly these three labels: {LABELS}")
    if "Id" in result:
        ids = pd.to_numeric(result.Id, errors="coerce")
        if ids.isna().any() or not np.isfinite(ids).all() or ids.duplicated().any() or (ids % 1 != 0).any() or (ids <= 0).any():
            raise ValueError("Id values must be unique positive integers.")
        result["Id"] = ids.astype(int)
    # Identical measurements with conflicting targets need explicit adjudication.
    if result.groupby(FEATURES).Species.nunique().gt(1).any():
        raise ValueError("Conflicting species labels for identical measurements.")
    return result.reset_index(drop=True)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path}. Run python -m src.download or import the Kaggle CSV with --import-csv.")
    frame = pd.read_csv(path)
    return validate_data(frame)


def inspection(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame), "columns": list(frame.columns),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing": frame.isna().sum().astype(int).to_dict(),
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "duplicate_measurement_rows": int(frame.duplicated(FEATURES + ["Species"]).sum()),
        "class_counts": frame.Species.value_counts().astype(int).to_dict(),
        "summary": frame[FEATURES].describe().to_dict(),
        "cleaning": "Numeric types validated; Id excluded from features. All observations retained. Identical measurements grouped for splitting and CV.",
    }


def extract_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean = validate_data(frame)
    return clean.loc[:, FEATURES], clean.Species
