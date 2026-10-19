"""Interactive descriptive plots and reproducible static report figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .config import COLORS, FEATURES, LABELS, NAMES
from .utils import read_json


def style(fig, title: str):
    fig.update_layout(title=title, font=dict(family="Arial", size=13),
                      margin=dict(l=25, r=25, t=65, b=35), legend_title_text="Species",
                      colorway=list(COLORS.values()), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(128,128,128,0.04)")
    return fig


def scatter(frame, x=FEATURES[2], y=FEATURES[3]):
    return style(px.scatter(frame, x=x, y=y, color="Species", color_discrete_map=COLORS,
                            labels=NAMES, hover_data=FEATURES, opacity=.85), "Measurements reveal species structure")


def distribution(frame, feature, kind="Histogram"):
    args = dict(data_frame=frame, x=feature, color="Species", color_discrete_map=COLORS, labels=NAMES)
    if kind == "Histogram":
        fig = px.histogram(**args, barmode="overlay", opacity=.65, nbins=20)
        fig.update_yaxes(title="Observations")
    elif kind == "Box":
        fig = px.box(**args, y="Species", points="all")
    else:
        fig = px.violin(**args, y="Species", box=True, points="all")
    return style(fig, f"{NAMES[feature]} · {kind.lower()}")


def correlation(frame):
    corr = frame[FEATURES].corr()
    return style(px.imshow(corr, x=[NAMES[f] for f in FEATURES], y=[NAMES[f] for f in FEATURES],
                            zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", labels={"color": "Pearson r"}), "Feature correlations · descriptive, not causal")


def pairplot(frame):
    fig = px.scatter_matrix(frame, dimensions=FEATURES, color="Species", color_discrete_map=COLORS, labels=NAMES)
    fig.update_traces(diagonal_visible=False, marker=dict(size=5, opacity=.7))
    fig.update_layout(height=760)
    return style(fig, "All four measurements, pair by pair")


def pca_plot(frame):
    scaled = StandardScaler().fit_transform(frame[FEATURES])
    pca = PCA(n_components=2)
    scores = pca.fit_transform(scaled)
    plot_data = pd.DataFrame(scores, columns=["PC1", "PC2"]).assign(Species=frame.Species.to_numpy())
    variance = pca.explained_variance_ratio_
    fig = px.scatter(plot_data, x="PC1", y="PC2", color="Species", color_discrete_map=COLORS,
                     labels={"PC1": f"PC1 ({variance[0]:.1%} variance; standardized units)",
                             "PC2": f"PC2 ({variance[1]:.1%} variance; standardized units)"})
    return style(fig, "PCA · a descriptive map of the complete dataset"), variance


def confusion(values, normalized=False):
    return style(px.imshow(values, x=LABELS, y=LABELS, text_auto=".2f" if normalized else True,
                            color_continuous_scale="Purples", zmin=0, zmax=1 if normalized else None,
                            labels={"x": "Predicted species", "y": "Actual species", "color": "Fraction" if normalized else "Count"}),
                 "Normalized confusion matrix" if normalized else "Held-out confusion matrix")


def export_figures(frame: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    directory = output / "reports" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=list(COLORS.values()))
    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=frame, x="Species", hue="Species", palette=COLORS, legend=False, ax=ax)
    ax.set(title="Species distribution", ylabel="Observations", xlabel="Species")
    save(fig, "species_distribution")
    for kind in ["histogram", "box", "violin"]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for feature, ax in zip(FEATURES, axes.flat):
            if kind == "histogram":
                sns.histplot(data=frame, x=feature, hue="Species", palette=COLORS, ax=ax, bins=15)
                ax.set(xlabel=NAMES[feature], ylabel="Observations")
            else:
                function = sns.boxplot if kind == "box" else sns.violinplot
                function(data=frame, x="Species", y=feature, hue="Species", palette=COLORS, ax=ax, legend=False, **({"cut": 0} if kind == "violin" else {}))
                ax.set(ylabel=NAMES[feature], xlabel="Species")
                ax.tick_params(axis="x", rotation=15)
        fig.suptitle(f"Feature {kind} distributions")
        fig.tight_layout()
        save(fig, kind)
    grid = sns.pairplot(frame, vars=FEATURES, hue="Species", palette=COLORS, diag_kind="hist")
    for i, row in enumerate(grid.axes):
        for j, ax in enumerate(row):
            if i == len(FEATURES) - 1:
                ax.set_xlabel(NAMES[FEATURES[j]])
            if j == 0:
                ax.set_ylabel(NAMES[FEATURES[i]])
    grid.figure.suptitle("Iris measurement relationships", y=1.02)
    save(grid.figure, "pairplot")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(frame[FEATURES].rename(columns=NAMES).corr(), annot=True, vmin=-1, vmax=1, cmap="RdBu_r", ax=ax)
    ax.set_title("Pearson correlations · descriptive")
    save(fig, "correlation")
    metrics = read_json(output / "reports" / "metrics" / "evaluation.json")
    for normalized in [False, True]:
        fig, ax = plt.subplots(figsize=(8, 6))
        key = "normalized_confusion_matrix" if normalized else "confusion_matrix"
        sns.heatmap(np.array(metrics[key]), annot=True, fmt=".2f" if normalized else "d", cmap="Purples", xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set(xlabel="Predicted species", ylabel="Actual species", title=key.replace("_", " ").title())
        save(fig, key)
    comparison = pd.DataFrame(read_json(output / "reports" / "metrics" / "model_comparison.json"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(comparison.model, comparison.cv_macro_f1, xerr=comparison.cv_std, color="#8b5cf6")
    ax.set(xlim=(0, 1.05), xlabel="Training CV macro F1 ± fold standard deviation", title="Model comparison; tuned CV scores are selection estimates")
    save(fig, "model_comparison")
    importance = pd.read_csv(output / "reports" / "metrics" / "permutation_importance.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([NAMES[f] for f in importance.feature], importance["mean"], xerr=importance["std"], color="#0eaaa0")
    ax.set(xlabel="Decrease in holdout macro F1 ± permutation standard deviation", title="Permutation importance · 30 repeats")
    save(fig, "permutation_importance")
    scores = pd.DataFrame(metrics["classification_report"]).T.loc[LABELS, ["precision", "recall", "f1-score"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    scores.plot.bar(ax=ax, color=["#8b5cf6", "#0eaaa0", "#ed9462"], rot=0)
    ax.set(ylim=(0, 1.1), xlabel="Species", ylabel="Held-out score", title="Per-class precision, recall and F1")
    save(fig, "class_metrics")
    for name, fig in [("interactive_scatter", scatter(frame)), ("interactive_pairplot", pairplot(frame)), ("pca", pca_plot(frame)[0])]:
        fig.write_html(directory / f"{name}.html", include_plotlyjs=True)
