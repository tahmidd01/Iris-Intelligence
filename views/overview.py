import streamlit as st
from components.state import context
from components.ui import hero, chart
from src.config import FEATURES, LABELS, COLORS
from src.eda import scatter

frame, model, metadata, metrics, comparison = context()
hero("An interactive machine learning field guide", "A small flower.<br>A world of intelligence.", "Explore the Data. Understand the Model. Predict the Species.<br>Discover how four botanical measurements become a transparent, reproducible classification experiment.")
cols = st.columns(4)
for col, label, value in zip(cols, ["Verified observations", "Measurement features", "Species", "Holdout accuracy"], [len(frame), 4, 3, f"{metrics['accuracy']:.1%}"]):
    col.metric(label, value, border=True)
st.caption(f"Kaggle Iris.csv · {metadata['selected_model']} · {metadata['train_rows']} training / {metadata['test_rows']} held-out rows · seed {metadata['random_seed']}")
left, right = st.columns([1.55, 1], gap="large")
with left:
    chart(scatter(frame), "overview_scatter")
with right:
    st.subheader("From measurement to meaning")
    st.write("Iris classification is supervised learning: a model learns relationships between measured flowers and known species, then predicts a label for new measurements.")
    for title, detail in [("01 · Inspect", "Validate the authentic CSV and explore four features in cm."), ("02 · Learn", "Compare five models with training-only cross-validation."), ("03 · Verify", "Evaluate the selected pipeline on an untouched holdout."), ("04 · Experiment", "Change measurements and inspect predictions in the lab.")]:
        st.markdown(f'<div class="workflow"><b>{title}</b><br>{detail}</div>', unsafe_allow_html=True)
    st.page_link("views/lab.py", label="Open the Prediction Lab →", icon=":material/science:")
st.subheader("Meet the three species")
for i, (col, species) in enumerate(zip(st.columns(3), LABELS)):
    group = frame[frame.Species == species]
    with col:
        st.markdown(f'<div class="species-card"><span class="species-number">0{i+1}</span><h3 style="color:{COLORS[species]}">{species.removeprefix("Iris-").title()}</h3><p>{len(group)} observed flowers</p><p>Mean petal length: {group[FEATURES[2]].mean():.2f} cm<br>Mean petal width: {group[FEATURES[3]].mean():.2f} cm</p></div>', unsafe_allow_html=True)
st.caption("Observed dataset means describe this sample. Petal measurements visually separate setosa; versicolor and virginica overlap more. These are descriptive observations.")
with st.expander("Explore every workspace"):
    for page, name in [("dataset", "Dataset Explorer"), ("analytics", "Visual Analytics"), ("performance", "Model Performance"), ("learn", "ML Explained"), ("about", "About the Project")]:
        st.page_link(f"views/{page}.py", label=name)
