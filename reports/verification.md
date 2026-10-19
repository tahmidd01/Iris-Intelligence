# Delivery verification

Verified on Windows 11 with Python 3.11.9 on 19 October 2026 (Asia/Dhaka). Results below record actual executions, not predicted outcomes.

| Check | Outcome |
|---|---|
| Official Kaggle public archive download | Successful; Iris.csv, 150 records, original six-column schema |
| Source checksum | `600ac44f23c2e6e0ae37daac8ceb2baba4df963efa580eb31b3b576b28e34c55` |
| Dataset validation | Zero missing values, balanced 50/50/50 classes, zero duplicate full rows, three repeated measurement rows retained |
| Complete five-model GridSearchCV | Successful; five grouped stratified folds, training-only tuning |
| Model selection | Decision Tree, max_depth=3, min_samples_leaf=3 |
| Selected CV macro F1 | 0.9665359477; fold SD 0.0487818198 |
| Independent holdout | 29/30 correct; accuracy 0.9666666667; macro F1 0.9665831245 |
| Model persistence and inference | Loaded saved pipeline; real Setosa prediction with normalized probabilities |
| Pytest suite | 44 passing tests; raw execution and JUnit evidence in `metrics/` |
| Python syntax/import initialization | compileall passed; every dashboard page initialized through Streamlit AppTest |
| Dependency consistency | `pip check`: No broken requirements found |
| Research notebook | All 11 code cells executed successfully; no error outputs; independently reproduced canonical holdout metrics exactly |
| Live Streamlit startup | Successful, loopback port 8501; `/_stcore/health` returned `ok` |
| Browser inspection | Overview, Prediction Lab and Model Performance loaded; real prediction produced Setosa |
| Theme / responsive checks | Desktop light and dark appearance inspected; 390x844 viewport inspected, then default viewport and project theme restored |
| Genuine screenshots | `screenshots/overview.jpg`, `prediction.jpg`, `dark.jpg`, `mobile.jpg` |
| Printable PDF | Four A4 pages, generated from actual results; Poppler rendered every page and all four final renders were visually inspected |

The execution sandbox initially prevented DNS and loopback sockets. The authorized download, package installation, live server and Jupyter kernel were rerun successfully with local execution access. These restrictions do not require a code workaround or a fallback dataset. Poppler reported missing optional display-font mappings while rendering; all final pages rendered correctly and were visually verified.

The first test pass found an artifact-schema check that assumed every pipeline's first step was an estimator. Tree pipelines begin with passthrough; the check now validates the first fitted step. An out-of-range fixture was also corrected, and navigation smoke tests now use the real app entry point. No experimental values or model predictions were changed to satisfy tests.

The report PDF uses optional ReportLab with bundled tooling; app/training dependencies remain in the verified local virtual environment. The Markdown report and all machine-readable outputs are the complete experiment record. The notebook and PDF can be regenerated using the documented commands.

Remaining scientific limits: this is one small historical benchmark split; tuned CV estimates are optimistic; probabilities lack independent calibration; feature-range checks cannot certify all joint input combinations. There are no outstanding implementation blockers.
