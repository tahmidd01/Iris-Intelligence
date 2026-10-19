# Windows 11 user guide

Open PowerShell in the `Task 1` project root. Python 3.11 is the tested version; check `py -3.11 --version`.

## 1. Create the local environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For the exact recorded transitive environment, use `-r requirements-lock.txt` instead. Optional activation is `.\.venv\Scripts\Activate.ps1`; if PowerShell blocks it, simply use the executable commands shown here. No policy change is necessary.

## 2. Acquire the required dataset

```powershell
.\.venv\Scripts\python.exe -m src.download
```

The command downloads the official Kaggle archive, validates `Iris.csv`, and writes provenance. On a fresh network/authentication failure, go to [the dataset page](https://www.kaggle.com/datasets/saurabh00007/iriscsv), download and unzip the archive, then run:

```powershell
.\.venv\Scripts\python.exe -m src.download --import-csv "$HOME\Downloads\Iris.csv"
```

An optional Kaggle CLI alternative is `kaggle datasets download -d saurabh00007/iriscsv -p data/raw --unzip`. Use the official Kaggle authentication instructions, store tokens outside the repository, and do not paste them into project code. Manual imports are honestly labeled user supplied. No fallback dataset is silently created.

## 3. Train, evaluate and verify

```powershell
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Training validates data, tunes all five classifiers, chooses the winner by training CV, evaluates the winner once on the reserved holdout, and writes the pipeline, metadata, figures, metrics, errors and Markdown report. Stop the running app before retraining; restart afterward. Do not tune settings in response to holdout scores and continue calling the same holdout independent.

## 4. Start the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`. Keep PowerShell running; Ctrl+C stops the server. If the port is occupied, use `--server.port 8502` and open `http://localhost:8502`. The server binds to loopback only.

## 5. Explore the workspaces

1. **Overview:** review the data, selected model, measured accuracy and species cards. Open the lab using the page link or sidebar.
2. **Dataset Explorer:** filter species and measurements; inspect the table, by-species statistics and validation audit. Download the filtered CSV. Id is never a predictor.
3. **Visual Analytics:** choose X/Y features; switch distribution views; inspect correlation, pair relationships, PCA and the 3D chart. Download interactive HTML figures or the static full-data pair plot. PCA is descriptive.
4. **Prediction Lab:** choose a real observed preset or enter positive cm measurements. Click **PREDICT FLOWER SPECIES**. Read the prediction, class estimates, submitted values and extrapolation warning. Adjust one measurement in the what-if studio; inspect similar training examples. Predictions accumulate in session history, which supports CSV export and clearing. Reset restores the first observed example. If inputs change after submission, the app labels the displayed result as stale until you predict again.
5. **Model Performance:** review candidate scores and fold variation, best settings, raw/normalized confusion matrices, class metrics, global permutation importance and every real held-out error. Download the experiment and report.
6. **ML Explained:** open the lessons and change true/false-positive/negative counts to explore precision, recall and F1 independently of the real model results.
7. **About:** review the objective, source, methodology, actual findings, limits and future work.

Use Streamlit's settings menu to select light or dark appearance. No photo upload is provided because this classifier uses numerical measurements.

## 6. Execute the notebook

```powershell
.\.venv\Scripts\python.exe -m scripts.notebook
```

Open `notebooks/exploratory_analysis.ipynb` in VS Code or a compatible notebook viewer. The generated notebook contains genuine executed results. The builder executes with this environment's Python kernel and reuses `src/` functions. It trains in a temporary output directory and leaves the main experiment intact.

## 7. Regenerate the optional PDF

The four-page report is already included at `output/pdf/iris_intelligence_report.pdf`. After retraining, regenerate it to keep it synchronized with the current experiment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe -m scripts.pdf_report
```

The optional ReportLab dependency is unnecessary for the application, training, tests or notebook. The full Markdown report is regenerated automatically by training; PDF export is a separate explicit step.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `No module named ...` | Install requirements using the same `.venv\Scripts\python.exe` used to launch the app. |
| Dataset missing | Run `src.download` or import the authentic Kaggle CSV. |
| Missing model / checksum mismatch | Retrain with `python -m src.train`; load only trusted local binaries. |
| CSV changed after training | Retrain so data and experiment checksums agree. |
| Schema or species error | Keep the six expected Kaggle columns (Id optional); use original Iris-prefixed species labels and positive finite numeric cm values. Do not silently rename unrelated data. |
| Download DNS / authentication failure | Use an allowed network or manual import; no credentials are required for the successful public download recorded in this workspace. |
| Port occupied | Launch on another port, e.g. 8502. |
| Preset looks familiar | Presets are observed dataset rows, labeled as such; they do not demonstrate independent generalization. |
| High probability but uncertain flower | These are uncalibrated model estimates; range warnings cannot detect all unusual combinations. |
| Notebook kernel blocked | Permit the local Python/Jupyter process to open its loopback sockets, then rerun the builder. |
| Different model after changing data or versions | Use the recorded source checksum, seed, split, search configuration and dependency lock to reproduce the recorded experiment. |
