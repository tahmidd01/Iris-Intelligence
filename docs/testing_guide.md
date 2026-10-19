# Testing guide

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app.py src components views tests scripts
.\.venv\Scripts\python.exe -m pip check
```

Tests use deterministic sklearn Iris fixtures clearly separated from the authentic primary CSV. Tests do not overwrite the canonical model or reports. Integration training writes to pytest's temporary directory and uses one candidate per estimator and three folds for speed. The real portfolio run uses the full search and five folds.

Coverage includes loading/missing data; required/unexpected columns; invalid measurements/IDs/species; full feature order; repeated-observation preservation; disjoint measurement groups across train/test and CV; scaler statistics learned from training only; five-model training, persistence and inference; checksum rejection; probability bounds and normalization; actual metric recomputation; extrapolation flags; seven dashboard pages through the real navigation entry point; prediction, history, warnings and reset; empty data selections.

Run only core tests without downloaded data or trained dashboard artifacts:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_data.py tests/test_pipeline.py -q
```

The actual Kaggle-file check skips when the dataset is absent. Dashboard tests skip when artifacts are absent; train first for complete UI coverage. The main entry test can still check helpful initialization in an untrained checkout. Streamlit AppTest tests render Python-side elements and interactions; they do not replace visual browser inspection. Streamlit may log its standard missing-ScriptRunContext warning during AppTest initialization; this is a testing-context message, not an application runtime failure.

The notebook builder executes every code cell with errors disallowed. A genuine local server health response and browser inspection provide separate startup/UI evidence. See `reports/verification.md` for the actual checks performed on the delivered workspace.
