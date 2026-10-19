import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .config import LABELS, FEATURES, SEED


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    predicted = model.predict(X)
    report = classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"], "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"], "weighted_f1": report["weighted avg"]["f1-score"],
        "correct": int(np.sum(predicted == y)), "incorrect": int(np.sum(predicted != y)), "test_rows": len(y),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, predicted, labels=LABELS).tolist(),
        "normalized_confusion_matrix": confusion_matrix(y, predicted, labels=LABELS, normalize="true").tolist(),
    }
    records = X.copy()
    records.insert(0, "row_index", X.index)
    records["actual"] = y.to_numpy()
    records["predicted"] = predicted
    records["correct"] = predicted == y.to_numpy()
    importance = permutation_importance(model, X, y, scoring="f1_macro", n_repeats=30, random_state=SEED, n_jobs=1)
    table = pd.DataFrame({"feature": FEATURES, "mean": importance.importances_mean, "std": importance.importances_std})
    return metrics, records, table
