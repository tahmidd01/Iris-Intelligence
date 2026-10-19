"""Train five classifiers using grouped stratified CV, then evaluate one holdout."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from .config import ROOT, DATA_PATH, FEATURES, LABELS, SEED
from .data_loader import load_data, inspection
from .evaluate import evaluate
from .preprocessing import make_pipeline, measurement_groups, split_data
from .utils import read_json, write_json, sha256


def candidates(seed: int = SEED) -> dict:
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=seed), True, {"model__C": [.1, 1, 10]}),
        "K-Nearest Neighbors": (KNeighborsClassifier(), True, {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"], "model__p": [1, 2]}),
        "Support Vector Machine": (SVC(probability=True, random_state=seed), True, [
            {"model__kernel": ["linear"], "model__C": [.1, 1, 10]},
            {"model__kernel": ["rbf"], "model__C": [.1, 1, 10], "model__gamma": ["scale", .1]}]),
        "Decision Tree": (DecisionTreeClassifier(random_state=seed), False, {"model__max_depth": [2, 3, None], "model__min_samples_leaf": [1, 3]}),
        "Random Forest": (RandomForestClassifier(random_state=seed, n_jobs=1), False, {"model__n_estimators": [100, 200], "model__max_depth": [3, None], "model__min_samples_leaf": [1, 2]}),
    }


def train(data_path: Path = DATA_PATH, output: Path = ROOT, seed: int = SEED, folds: int = 5, quick: bool = False) -> dict:
    frame = load_data(data_path)
    training, testing = split_data(frame, seed)
    X, y = frame[FEATURES], frame.Species
    X_train, y_train = X.loc[training], y.loc[training]
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    groups = measurement_groups(X_train)
    if any(groups[y_train == label].nunique() < folds for label in LABELS):
        raise ValueError(f"Each training species needs at least {folds} distinct measurement groups for CV.")
    rows, fitted, grids = [], {}, {}
    for rank, (name, (estimator, scaled, grid)) in enumerate(candidates(seed).items()):
        if quick:
            first = grid[0] if isinstance(grid, list) else grid
            grid = {key: [values[0]] for key, values in first.items()}
        grids[name] = grid
        search = GridSearchCV(make_pipeline(estimator, scaled), grid, cv=cv,
                              scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"}, refit="macro_f1", n_jobs=1, error_score="raise")
        search.fit(X_train, y_train, groups=groups)
        index = search.best_index_
        rows.append({"model": name, "cv_macro_f1": float(search.best_score_),
                     "cv_std": float(search.cv_results_["std_test_macro_f1"][index]),
                     "cv_accuracy": float(search.cv_results_["mean_test_accuracy"][index]),
                     "best_params": search.best_params_, "complexity_rank": rank,
                     "fold_scores": [float(search.cv_results_[f"split{i}_test_macro_f1"][index]) for i in range(folds)],
                     "candidate_count": len(search.cv_results_["params"])})
        fitted[name] = search.best_estimator_
        print(f"{name}: CV macro F1={search.best_score_:.5f}", flush=True)
    # Test measurements and targets are used only after all selection is complete.
    best_score = max(r["cv_macro_f1"] for r in rows)
    tied = [r for r in rows if np.isclose(r["cv_macro_f1"], best_score, rtol=0, atol=1e-12)]
    winner = sorted(tied, key=lambda r: (-r["cv_accuracy"], r["complexity_rank"]))[0]
    model = fitted[winner["model"]]
    metrics, predictions, importance = evaluate(model, X.loc[testing], y.loc[testing])
    model_dir, metrics_dir = output / "models", output / "reports" / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "iris_pipeline.joblib")
    provenance_path = data_path.with_suffix(".provenance.json")
    provenance = read_json(provenance_path) if provenance_path.exists() else {"method": "local CSV; origin unverified"}
    if provenance.get("sha256") and provenance["sha256"] != sha256(data_path):
        provenance = {"method": "local CSV modified after acquisition; origin unverified"}
    metadata = {
        "selected_model": winner["model"], "estimator_class": type(model.named_steps["model"]).__name__,
        "features": FEATURES, "labels": list(model.classes_), "units": "cm", "random_seed": seed,
        "trained_utc": datetime.now(timezone.utc).isoformat(), "dataset_sha256": sha256(data_path),
        "model_sha256": sha256(model_dir / "iris_pipeline.joblib"), "provenance": provenance,
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "streamlit", "plotly"]},
        "python": platform.python_version(), "train_rows": len(training), "test_rows": len(testing),
        "train_indices": training.tolist(), "test_indices": testing.tolist(),
        "training_ranges": {f: {"min": float(X_train[f].min()), "max": float(X_train[f].max())} for f in FEATURES},
        "selection": "Highest training CV macro F1; numerical ties resolved by CV accuracy then fixed simplicity order (linear, neighbors, SVM, tree, forest). Holdout never used for selection.",
        "cv": {"folds": folds, "shuffle": True, "grouping": "identical four-feature measurement tuples", "test_fraction_of_unique_groups": .2},
        "winning_cv": winner, "search_spaces": grids, "quick_mode": quick,
    }
    write_json(model_dir / "metadata.json", metadata)
    write_json(metrics_dir / "evaluation.json", metrics)
    write_json(metrics_dir / "model_comparison.json", rows)
    write_json(metrics_dir / "data_validation.json", inspection(frame))
    write_json(metrics_dir / "experiment.json", metadata)
    write_json(metrics_dir / "classification_report.json", metrics["classification_report"])
    pd.DataFrame(rows).drop(columns=["best_params", "fold_scores"]).to_csv(metrics_dir / "model_comparison.csv", index=False)
    predictions.to_csv(metrics_dir / "test_predictions.csv", index=False)
    predictions[~predictions.correct].to_csv(metrics_dir / "errors.csv", index=False)
    importance.to_csv(metrics_dir / "permutation_importance.csv", index=False)
    processed = output / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed / "validated_iris.csv", index=False)
    frame.groupby("Species")[FEATURES].agg(["mean", "std", "min", "max"]).to_csv(metrics_dir / "species_summary.csv")
    return {"metadata": metadata, "metrics": metrics, "comparison": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--quick", action="store_true", help="One candidate per model; development only.")
    args = parser.parse_args()
    try:
        results = train(args.data, quick=args.quick)
        from .eda import export_figures
        from .report import write_report
        export_figures(load_data(args.data), ROOT)
        write_report(results, ROOT)
        print(f"Selected {results['metadata']['selected_model']}; holdout accuracy {results['metrics']['accuracy']:.4f}")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
