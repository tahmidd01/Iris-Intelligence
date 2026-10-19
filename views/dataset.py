import pandas as pd
import plotly.express as px
import streamlit as st
from components.state import context
from components.ui import hero, chart
from src.config import FEATURES, LABELS, COLORS, NAMES
from src.data_loader import inspection
from src.eda import style

frame, _, metadata, _, _ = context()
hero("01 / Dataset explorer", "Know your observations.", "An inspectable source of truth. Filter species, compare measurements, and export exactly what you see. All four features are measured in centimeters.")
left, right = st.columns([1, 2])
species = left.multiselect("Species filter", LABELS, default=LABELS)
features = right.multiselect("Visible measurements (cm)", FEATURES, default=FEATURES, format_func=NAMES.get)
filtered = frame[frame.Species.isin(species)]
st.caption(f"Showing {len(filtered)} / {len(frame)} observations · Id is a row identifier and never a model feature.")
tabs = st.tabs(["Observations", "Statistics by species", "Data integrity"])
with tabs[0]:
    display = filtered[(["Id"] if "Id" in frame else []) + features + ["Species"]]
    st.dataframe(display, width="stretch", hide_index=True, column_config={f: st.column_config.NumberColumn(NAMES[f], format="%.1f") for f in features})
    st.download_button("Download filtered CSV", display.to_csv(index=False), "iris_filtered.csv", "text/csv")
    if not filtered.empty:
        chart(style(px.histogram(filtered, x="Species", color="Species", color_discrete_map=COLORS, labels={"count": "Observations"}), "Species in the current selection"), "dataset_counts")
    else:
        st.info("Select one or more species to explore the dataset.")
with tabs[1]:
    if features and not filtered.empty:
        st.dataframe(filtered.groupby("Species")[features].agg(["mean", "std", "min", "median", "max"]).round(3), width="stretch")
        st.caption("Standard deviations use the sample convention (n − 1). These are descriptive summaries.")
    else:
        st.info("Select a species and at least one measurement for summary statistics.")
with tabs[2]:
    audit = inspection(frame)
    a, b, c = st.columns(3)
    a.metric("Missing values", sum(audit["missing"].values()), border=True)
    b.metric("Identical full rows", audit["duplicate_full_rows"], border=True)
    c.metric("Repeated measurement rows", audit["duplicate_measurement_rows"], border=True)
    st.write(audit["cleaning"])
    st.dataframe(pd.DataFrame({"Data type": audit["dtypes"], "Missing": audit["missing"]}), width="stretch")
    st.json(metadata["provenance"])
