import pandas as pd
import plotly.express as px
import streamlit as st
from components.state import context
from components.ui import hero, chart, file_download
from src.config import ROOT, LABELS, NAMES, COLORS
from src.eda import style, confusion

_, _, metadata, metrics, comparison = context()
hero("04 / Model comparison observatory", "Evidence before confidence.", "Five algorithms, one training-only selection process, one independent holdout. Inspect the scores, the variation, and the mistakes behind the final pipeline.")
st.subheader(f"Selected: {metadata['selected_model']}")
for col, title, key in zip(st.columns(4), ["Test accuracy", "Macro precision", "Macro recall", "Macro F1"], ["accuracy", "macro_precision", "macro_recall", "macro_f1"]):
    col.metric(title, f"{metrics[key]:.2%}", border=True)
st.caption(f"{metrics['correct']} correct / {metrics['incorrect']} incorrect · {metadata['test_rows']} held-out rows · weighted precision {metrics['weighted_precision']:.4f}, recall {metrics['weighted_recall']:.4f}, F1 {metrics['weighted_f1']:.4f}")
tabs = st.tabs(["Selection & cross-validation", "Held-out evaluation", "Feature relevance", "Error analysis", "Reproducibility"])
with tabs[0]:
    results = pd.DataFrame(comparison)
    fig = px.bar(results, x="cv_macro_f1", y="model", error_x="cv_std", orientation="h", color_discrete_sequence=["#8b5cf6"], labels={"model": "Algorithm", "cv_macro_f1": "Training CV macro F1"})
    fig.update_xaxes(range=[0, 1.05])
    chart(style(fig, "Training cross-validation · mean ± fold standard deviation"), "performance_cv", True)
    folds = pd.DataFrame([{"Algorithm": row["model"], "Fold": i + 1, "Macro F1": value} for row in comparison for i, value in enumerate(row["fold_scores"])])
    chart(style(px.line(folds, x="Fold", y="Macro F1", color="Algorithm", markers=True), "Variation across the five training folds"), "performance_folds")
    st.write(metadata["selection"])
    st.info("The chart compares tuned CV selection estimates. Reusing CV to select hyperparameters makes these estimates optimistic. Only the selected model receives a holdout evaluation; the small test set does not prove universal superiority.")
    st.dataframe(results[["model", "cv_macro_f1", "cv_std", "cv_accuracy", "candidate_count"]], hide_index=True, width="stretch")
    with st.expander("Best hyperparameters for each algorithm"):
        for row in comparison:
            st.write(row["model"])
            st.json(row["best_params"])
with tabs[1]:
    normalized = st.toggle("Normalize by actual species")
    chart(confusion(metrics["normalized_confusion_matrix"] if normalized else metrics["confusion_matrix"], normalized), "performance_cm", True)
    report = pd.DataFrame(metrics["classification_report"]).T.drop(index="accuracy")
    st.dataframe(report, width="stretch")
    per_class = report.loc[LABELS, ["precision", "recall", "f1-score"]].reset_index(names="Species").melt(id_vars="Species", var_name="Metric", value_name="Score")
    chart(style(px.bar(per_class, x="Species", y="Score", color="Metric", barmode="group", range_y=[0, 1.05]), "Per-class precision, recall and F1"), "performance_classes")
    st.caption("Precision asks how often a predicted species is right; recall asks how much of an actual species was recovered. Macro F1 treats each species equally. Accuracy is the fraction of all flowers classified correctly.")
with tabs[2]:
    importance = pd.read_csv(ROOT / "reports" / "metrics" / "permutation_importance.csv")
    importance["Measurement"] = importance.feature.map(NAMES)
    fig = px.bar(importance, x="mean", y="Measurement", error_x="std", orientation="h", labels={"mean": "Decrease in holdout macro F1"}, color_discrete_sequence=["#0eaaa0"])
    chart(style(fig, "Permutation importance · 30 held-out shuffles per feature"), "performance_importance", True)
    st.caption("A feature is shuffled while others stay fixed; the score decrease estimates global model dependence. Error bars show shuffle variation, not confidence intervals. Correlated features can substitute for each other, and negative values can occur by chance. This is not a causal or per-prediction explanation.")
with tabs[3]:
    predictions = pd.read_csv(ROOT / "reports" / "metrics" / "test_predictions.csv")
    errors = predictions[~predictions.correct]
    if errors.empty:
        st.success("No errors on this particular holdout. This does not establish perfect performance on future flowers.")
    else:
        st.dataframe(errors, width="stretch", hide_index=True)
        st.write("Overlapping measurements, especially among versicolor and virginica, are a plausible source of ambiguity. Individual mistakes do not establish the cause.")
    st.subheader("Actual versus predicted · every held-out flower")
    labels = predictions.melt(id_vars="row_index", value_vars=["actual", "predicted"], var_name="Label source", value_name="Species")
    fig = px.scatter(labels, x="row_index", y="Species", color="Species", symbol="Label source",
                     symbol_sequence=["circle-open", "x"], color_discrete_map=COLORS,
                     labels={"row_index": "Original CSV data row (zero-based)"})
    fig.update_traces(marker_size=11)
    chart(style(fig, "Actual and predicted labels · separation identifies a mistake"), "performance_actual_predicted", True)
    st.dataframe(predictions, width="stretch", hide_index=True)
    file_download(ROOT / "reports" / "metrics" / "test_predictions.csv", "Download held-out predictions")
with tabs[4]:
    st.json({key: metadata[key] for key in ["estimator_class", "features", "labels", "random_seed", "train_rows", "test_rows", "cv", "versions", "dataset_sha256", "model_sha256"]})
    st.caption("Only the training partition was used to fit this saved model. Identical measurement tuples stay in the same split and CV fold. The exact indices and all search spaces are recorded in the experiment JSON.")
    file_download(ROOT / "reports" / "metrics" / "experiment.json", "Download experiment configuration")
    file_download(ROOT / "reports" / "results.md", "Download full results report")
    file_download(ROOT / "output" / "pdf" / "iris_intelligence_report.pdf", "Download printable PDF report")
