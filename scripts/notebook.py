"""Generate and execute the educational notebook in the current Python environment."""
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))
    md("# ✿ Iris Intelligence\n## An inspectable journey from measurements to classification\nExplore the data. Understand the model. Predict the species.\n\nThis executed notebook uses the authentic required [Kaggle CSV](https://www.kaggle.com/datasets/saurabh00007/iriscsv). It reuses the project's modular implementation and writes its independent demonstration experiment to a temporary directory.")
    md("## 1 · Problem and dataset\nSupervised multiclass classification predicts setosa, versicolor or virginica from sepal/petal length and width in cm. Id is an identifier, never a feature. We do not classify photographs.")
    code("from pathlib import Path\nimport sys, tempfile\nroot = Path.cwd()\nif not (root / 'src').exists():\n    root = root.parent\nsys.path.insert(0, str(root))\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom IPython.display import display, Image, Markdown\nfrom src.config import FEATURES, LABELS, COLORS, NAMES, DATA_PATH\nfrom src.data_loader import load_data, inspection, extract_features\nfrom src.preprocessing import split_data, measurement_groups\nfrom src.train import train\nfrom src.predict import load_bundle, predict\nfrom src.utils import read_json, sha256\nfrom src.eda import export_figures\n%matplotlib inline\nprint('Python:', sys.version.split()[0])")
    md("## 2 · Load and validate\nThe loader rejects missing/unknown fields, invalid cm values, conflicting labels and invalid IDs. It retains legitimate repeated observations. Provenance is checked against the exact CSV bytes.")
    code("frame = load_data()\nprovenance = read_json(DATA_PATH.with_suffix('.provenance.json'))\nassert provenance['sha256'] == sha256(DATA_PATH)\ndisplay(pd.Series(provenance, name='Acquisition provenance'))\ndisplay(frame.head())\naudit = inspection(frame)\ndisplay(pd.Series({k: audit[k] for k in ['rows','duplicate_full_rows','duplicate_measurement_rows','class_counts','cleaning']}))\ndisplay(frame[FEATURES].describe())")
    md("## 3 · Exploratory analysis\nThese plots are descriptive. Visually separated clusters do not validate future performance. Petal features distinguish setosa well in this sample; versicolor and virginica show more overlap.")
    code("sns.set_theme(style='whitegrid')\nfig, axes = plt.subplots(1, 2, figsize=(13, 4))\nsns.countplot(data=frame, x='Species', hue='Species', palette=COLORS, legend=False, ax=axes[0])\naxes[0].set(title='Observed species distribution', ylabel='Observations')\nsns.scatterplot(data=frame, x=FEATURES[2], y=FEATURES[3], hue='Species', palette=COLORS, ax=axes[1])\naxes[1].set(title='Petal measurement structure', xlabel=NAMES[FEATURES[2]], ylabel=NAMES[FEATURES[3]])\nplt.tight_layout(); plt.show()\ndisplay(frame.groupby('Species')[FEATURES].agg(['mean','std','min','max']).round(3))")
    code("grid = sns.pairplot(frame, vars=FEATURES, hue='Species', palette=COLORS, diag_kind='hist')\nplt.show()\nfig, ax = plt.subplots(figsize=(7, 5))\nsns.heatmap(frame[FEATURES].rename(columns=NAMES).corr(), annot=True, vmin=-1, vmax=1, cmap='RdBu_r', ax=ax)\nax.set_title('Pearson correlations · descriptive, not causal'); plt.show()")
    md("## 4 · Features and the untouched holdout\nGroup identical four-feature tuples, then stratify unique groups using an 80/20 split and seed 42. This keeps repeated measurements together and preserves every observation. Row ratios can differ from group ratios. Do not fit preprocessing on the complete dataset.")
    code("X, y = extract_features(frame)\ntraining, testing = split_data(frame)\ngroups = measurement_groups(frame)\nassert set(groups.loc[training]).isdisjoint(groups.loc[testing])\nprint(f'{len(training)} training rows; {len(testing)} held-out rows')\nprint('Features:', list(X.columns))\nprint('Train labels:', y.loc[training].value_counts().to_dict())")
    md("## 5 · Training, grouped CV and hyperparameter tuning\nThe reused training function constructs all five pipelines. Logistic regression, KNN and SVM scale within CV; trees use passthrough. Five-fold shuffled stratified group CV tunes only training rows. Macro F1 chooses the winner; exact ties use accuracy and a fixed simplicity order. Scores after tuning are selection estimates and may be optimistic. The untouched test set is evaluated only after selection.")
    code("temporary = tempfile.TemporaryDirectory(prefix='iris-notebook-')\nexperiment_dir = Path(temporary.name)\nexperiment = train(DATA_PATH, output=experiment_dir)\nmeta, metrics = experiment['metadata'], experiment['metrics']\ncomparison = pd.DataFrame(experiment['comparison'])\ndisplay(comparison[['model','cv_macro_f1','cv_std','cv_accuracy','best_params']])\nprint('Selected:', meta['selected_model'])\nprint(meta['selection'])")
    md("## 6 · Model comparison\nError bars describe variability among training folds, not confidence intervals. The tiny score differences do not establish general model superiority.")
    code("fig, ax = plt.subplots(figsize=(9, 4))\nax.barh(comparison.model, comparison.cv_macro_f1, xerr=comparison.cv_std, color='#8b5cf6')\nax.set(xlim=(0,1.05), xlabel='Training CV macro F1 ± fold SD', title='Five classical classifiers')\nplt.tight_layout(); plt.show()")
    md("## 7 · Independent final evaluation\nAccuracy is the fraction correct. Macro F1 gives every class equal weight. Weighted F1 uses class support. The confusion matrix shows actual labels by row and predictions by column. The saved model remains fitted on training rows only.")
    code("display(pd.Series({k:v for k,v in metrics.items() if not isinstance(v, (dict,list))}, name='Holdout metrics'))\ndisplay(pd.DataFrame(metrics['classification_report']).T)\nfig, axes = plt.subplots(1,2, figsize=(12,4))\nfor ax, key, fmt in zip(axes, ['confusion_matrix','normalized_confusion_matrix'], ['d','.2f']):\n    sns.heatmap(pd.DataFrame(metrics[key], index=LABELS, columns=LABELS), annot=True, fmt=fmt, cmap='Purples', ax=ax)\n    ax.set(title=key.replace('_',' ').title(), xlabel='Predicted species', ylabel='Actual species')\nplt.tight_layout(); plt.show()")
    md("## 8 · Feature relevance and real mistakes\nHoldout permutation importance is a post-selection global diagnostic, not a local or causal explanation. Correlated features can substitute for one another. Test errors below are actual errors; if empty, no error is invented.")
    code("importance = pd.read_csv(experiment_dir/'reports/metrics/permutation_importance.csv')\nfig, ax = plt.subplots(figsize=(8,4))\nax.barh(importance.feature.map(NAMES), importance['mean'], xerr=importance['std'], color='#0eaaa0')\nax.set(xlabel='Decrease in holdout macro F1 ± shuffle SD', title='Permutation importance · 30 repeats')\nplt.tight_layout(); plt.show()\nerrors = pd.read_csv(experiment_dir/'reports/metrics/errors.csv')\ndisplay(errors)\nprint(f'Actual mistakes on this split: {len(errors)}')")
    md("## 9 · Persistence and inference\nLoad only trusted local joblib files. Measurements are validated and ordered by name. Probability estimates are not guarantees. A sample from this dataset is a demonstration, not independent validation.")
    code("model, loaded_meta = load_bundle(experiment_dir/'models')\nsample = frame.iloc[0][FEATURES].to_dict()\nresult = predict(model, loaded_meta, sample)\ndisplay(pd.Series(result))\nassert set(result['probabilities']) == set(LABELS)\nassert abs(sum(result['probabilities'].values())-1) < 1e-10\ncanonical = read_json(root/'reports/metrics/evaluation.json')\nassert metrics['accuracy'] == canonical['accuracy']\nassert abs(metrics['macro_f1']-canonical['macro_f1']) < 1e-12\nprint('Reproduced canonical holdout metrics exactly.')")
    md("## 10 · Discussion, limitations and conclusion\nPetal measurements show clear descriptive structure for setosa, with more overlap between the other species. The selected model and genuine scores above document this specific seed-42 experiment. A balanced 150-flower historical benchmark is too small and narrow to prove broad botanical reliability. There is no independent probability-calibration study, and unfamiliar feature combinations can be risky even within individual feature ranges. Future work should add external observations, nested CV and measurement-noise studies.\n\nThe project demonstrates the full numerical classification lifecycle: provenance, validation, exploration, grouped splitting, leakage-safe pipelines, training-only selection, independent evaluation, error analysis and persisted inference.\n\n### References\n- [Required Kaggle dataset](https://www.kaggle.com/datasets/saurabh00007/iriscsv)\n- [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)\n- [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)")
    code("temporary.cleanup()")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3 (Iris Intelligence)", "language": "python", "name": "python3"}})
    directory = root / "notebooks"
    directory.mkdir(exist_ok=True)
    path = directory / "exploratory_analysis.ipynb"
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=240, resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
        nbf.write(notebook, path)
        print(f"Executed {sum(c.cell_type == 'code' for c in cells)} code cells successfully: {path}")
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
