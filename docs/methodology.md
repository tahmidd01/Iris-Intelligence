# Methodology

## Collection and validation
The primary source is [saurabh00007/iriscsv on Kaggle](https://www.kaggle.com/datasets/saurabh00007/iriscsv). `python -m src.download` retrieves the official public archive, validates its CSV before replacement and records acquisition UTC time, method, filename and SHA-256. `--import-csv` validates a manual copy and labels the provenance as user supplied. Local CSVs without matching acquisition records are explicitly unverified. There is no automatic fallback.

Required fields are SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm and Species. Optional Id must contain unique positive integer values. Unknown/missing columns, missing classes, non-numeric or missing measurements, infinity, non-positive values and conflicting targets for identical measurements raise helpful errors. Species must match the three original Iris-prefixed labels exactly. Numeric strings are converted to floats; records are not deleted or imputed. IDs never enter the feature matrix. Inspection records shape, types, missingness, repetitions, class balance and summary statistics.

## Leakage prevention and splitting
All four-feature measurement tuples define groups. Repeated flowers remain in the dataset but identical measurements cannot cross evaluation boundaries. An 80/20 seed-42 train_test_split stratifies the distinct groups by species and maps them back to rows. The nominal ratio applies to groups, so row counts can differ slightly. Exact row indices are saved in the experiment. The actual Kaggle split contains 120 training and 30 test rows.

Five-fold StratifiedGroupKFold shuffles training groups with seed 42. Group stratification is approximate, so fold supports can differ. Every species must have at least five distinct training groups. Validation/test observations never fit scalers or estimators. The folds are shared across algorithms and candidates. Exploratory charts use the entire dataset descriptively; no feature-selection choice or fitted exploratory PCA enters the fixed four-feature training pipeline. Test targets are not inspected to decide model settings.

## Models and tuning

| Model | Preprocessing | Search |
|---|---|---|
| Logistic regression | StandardScaler | C: 0.1, 1, 10 |
| KNN | StandardScaler | k: 3,5,7,9; weights: uniform/distance; p: 1/2 |
| SVM | StandardScaler | linear C: 0.1,1,10; RBF C: 0.1,1,10 with gamma: scale/0.1 |
| Decision tree | Passthrough | depth: 2,3,None; min leaf: 1,3 |
| Random forest | Passthrough | trees: 100,200; depth: 3,None; min leaf: 1,2 |

GridSearchCV uses both macro F1 and accuracy, refits each estimator's highest macro F1 candidate on the full training partition, and records mean, fold SD, fold scores and best parameters. The cross-model winner maximizes macro F1; numerical ties within 1e-12 use CV accuracy then a predeclared simplicity rank (linear, neighbors, SVM, tree, forest). Within-model exact ties follow the declared grid order. No test score participates. `n_jobs=1` avoids unnecessary Windows worker overhead on this tiny dataset. Estimator random states are fixed. SVM's optional probability fitting is internal to its training; it is not an independent calibration study.

## Evaluation and interpretation
Only the final winner is evaluated on the untouched holdout. Report accuracy, per-class precision/recall/F1/support, macro and weighted aggregates, raw/normalized confusion matrices, correct/incorrect counts, all predictions and real mistakes. No invented errors are added. A model-agnostic holdout permutation analysis shuffles each measurement 30 times and measures macro-F1 decrease. This post-evaluation diagnostic is not used for selection. Correlated features and small test samples limit interpretation. Standard deviations reflect variation, not confidence intervals.

The selected pipeline is saved without fitting on test or full data. Metadata includes package/Python versions, estimator class, best settings, random seed, feature/target order, source/model checksums, training-only feature ranges, exact indices and provenance. Joblib is loaded only from trusted local artifacts after checksum/schema checks; a checksum does not make an untrusted artifact safe.

## Scientific limits
The optimized CV scores are selection estimates and can be optimistic. Nested CV would be needed for a broader unbiased comparison. A 30-row holdout has high sampling variation. The tiny CV advantage of the selected tree is not evidence of meaningful universal superiority. SVM probability ordering can differ from its decision label. Extrapolation checks flag individual ranges but cannot certify joint feature familiarity. Full-dataset PCA is explicitly exploratory and does not validate model performance.

References: [Scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
