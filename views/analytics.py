import streamlit as st
from components.state import context
from components.ui import hero, chart, file_download
from src.config import FEATURES, NAMES, LABELS, ROOT
from src.eda import scatter, correlation, distribution, pairplot, pca_plot, style
import plotly.express as px

frame, _, _, _, _ = context()
hero("02 / Visual analytics", "Find the structure.<br>See the overlap.", "Explore relationships in the observed measurements. Every view is descriptive; visual separation alone does not validate a classifier.")
selected = st.multiselect("Compare species", LABELS, default=LABELS)
data = frame[frame.Species.isin(selected)].reset_index(drop=True)
if data.empty:
    st.info("Select at least one species to render charts.")
    st.stop()
tabs = st.tabs(["Relationships", "Distributions", "Correlation", "Pair explorer", "PCA & dimensions"])
with tabs[0]:
    c1, c2 = st.columns(2)
    x = c1.selectbox("X-axis", FEATURES, index=2, format_func=NAMES.get)
    y = c2.selectbox("Y-axis", FEATURES, index=3, format_func=NAMES.get)
    chart(scatter(data, x, y), "analytics_scatter", True)
    st.caption("Setosa is visually distinct in petal space. Versicolor and virginica share some measurements; the model must combine evidence across features.")
with tabs[1]:
    feature = st.selectbox("Measurement", FEATURES, format_func=NAMES.get)
    kind = st.radio("Distribution view", ["Histogram", "Box", "Violin"], horizontal=True)
    chart(distribution(data, feature, kind), "analytics_distribution", True)
    st.caption("Box plots show medians, quartiles and spread. Violin widths are smoothed density estimates; their tails do not imply additional observed flowers.")
with tabs[2]:
    chart(correlation(data), "analytics_corr", True)
    st.caption("Pearson r measures linear association. A pooled correlation can partly reflect species differences; correlation is not causation or a feature-importance score.")
with tabs[3]:
    chart(pairplot(data), "analytics_pairs", True)
    file_download(ROOT / "reports" / "figures" / "pairplot.png", "Download static pair plot (all species)")
with tabs[4]:
    fig, variance = pca_plot(data)
    chart(fig, "analytics_pca", True)
    st.caption(f"The first two principal components explain {sum(variance):.1%} of standardized variance in this selection. PCA is fitted only for exploration; the classifier does not use this projection.")
    if len(data) > 1:
        from src.config import COLORS
        chart(style(px.scatter_3d(data, x=FEATURES[0], y=FEATURES[2], z=FEATURES[3], color="Species", color_discrete_map=COLORS, labels=NAMES), "Three dimensions of measurement"), "analytics_3d")
