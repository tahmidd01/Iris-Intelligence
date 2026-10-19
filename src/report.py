from pathlib import Path


def write_report(results: dict, output: Path) -> None:
    m, e = results["metadata"], results["metrics"]
    rows = "\n".join(f"| {r['model']} | {r['cv_macro_f1']:.4f} | {r['cv_std']:.4f} | {r['cv_accuracy']:.4f} |" for r in results["comparison"])
    content = f"""# Iris Intelligence — experimental report

## Abstract
Five classical classifiers were tuned on four numerical Iris flower measurements. The selected {m['selected_model']} achieved {e['accuracy']:.2%} accuracy and {e['macro_f1']:.4f} macro F1 on {e['test_rows']} held-out observations. These results describe one small benchmark split, not general botanical reliability.

## Introduction and objective
The supervised task maps sepal length, sepal width, petal length and petal width (cm) to setosa, versicolor or virginica. The objective is a transparent, reproducible classification workflow with an interactive measurement laboratory.

## Dataset and integrity
Source: {m['provenance'].get('source', 'local CSV, unverified origin')}. Acquisition: {m['provenance']['method']}.
CSV SHA-256: `{m['dataset_sha256']}`. Validation rejects missing columns, unexpected fields, non-finite or non-positive values, invalid IDs and conflicting labels. No observations are removed. Id is excluded from model inputs. Repeated measurements remain together across all train/validation/test boundaries.

## Methodology and experimental setup
Random seed {m['random_seed']}; an 80/20 stratified split of unique measurement groups yields {m['train_rows']} training rows and {m['test_rows']} test rows. Grouping can shift the exact row ratio. Five-fold shuffled stratified group CV is used inside GridSearchCV. Logistic regression, KNN and SVM use a StandardScaler inside their pipelines; trees use passthrough. String labels preserve the original species names. All tuning and scaler fitting use training partitions only. The final saved pipeline is fitted on the training partition, not refitted on the full dataset.

## Algorithms and selection
Logistic regression is a regularized linear classifier; KNN uses local distances; SVM uses linear or radial boundaries; a decision tree recursively partitions features; a random forest averages randomized trees. Search spaces and selected parameters are in `metrics/experiment.json`.
{m['selection']}

## Actual model comparison
| Algorithm | CV macro F1 | Fold SD | CV accuracy |
|---|---:|---:|---:|
{rows}

Selected model: **{m['selected_model']}**. CV macro F1: **{m['winning_cv']['cv_macro_f1']:.4f} ± {m['winning_cv']['cv_std']:.4f}**. The SD measures fold variation, not a confidence interval. Tuned CV estimates are optimistic selection estimates; no nested-CV superiority claim is made.

## Final evaluation
Only the selected model is evaluated on the untouched holdout. Accuracy: **{e['accuracy']:.4f}**; macro precision **{e['macro_precision']:.4f}**; macro recall **{e['macro_recall']:.4f}**; macro F1 **{e['macro_f1']:.4f}**. Weighted precision/recall/F1: **{e['weighted_precision']:.4f} / {e['weighted_recall']:.4f} / {e['weighted_f1']:.4f}**. Correct: **{e['correct']}**; incorrect: **{e['incorrect']}**.
Accuracy is the fraction correct; precision concerns predicted positives; recall concerns actual positives; F1 balances precision and recall. Macro averaging weights each class equally. Weighted averaging weights class support.

![Confusion matrix](figures/confusion_matrix.png)
![Model comparison](figures/model_comparison.png)
![Permutation importance](figures/permutation_importance.png)

## Error analysis and discussion
Every test example is in `metrics/test_predictions.csv`; errors are in `metrics/errors.csv`. {'There were no errors on this split; none are invented.' if e['incorrect'] == 0 else 'Misclassified examples are listed in the errors CSV and dashboard; overlapping versicolor and virginica measurements are a plausible explanation, not a proven cause.'}
Descriptive plots show strong petal-feature separation for setosa and more overlap between versicolor and virginica. Correlated measurements can share predictive information, so permutation importance can understate individual feature relevance. Importance is a global holdout diagnostic; it does not establish a causal explanation of a specific prediction. PCA is a descriptive full-dataset visualization and is never fed into the predictive pipeline.

## Limitations
The dataset is small, balanced and restricted to three species. Scores vary with the split. Probabilities are model estimates, not guarantees; they have not undergone an independent calibration study. SVM probability estimates use internal probability fitting. Predictions outside training feature ranges extrapolate. Nearest training examples provide similarity context, not proof of a decision mechanism. No photographs are analyzed.

## Conclusion
The implementation meets the classification task with validated measurements, training-only selection, an independent evaluation, persisted inference and a reusable dashboard. Future work includes externally collected observations, nested CV and independent probability calibration.

## References
- [Kaggle Iris.csv dataset](https://www.kaggle.com/datasets/saurabh00007/iriscsv)
- [Scikit-learn: common pitfalls and leakage prevention](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn: permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Streamlit documentation](https://docs.streamlit.io/)
"""
    (output / "reports" / "results.md").write_text(content, encoding="utf-8")
