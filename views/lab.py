from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from components.state import context
from components.ui import hero, chart
from src.config import FEATURES, NAMES, LABELS, COLORS
from src.eda import style
from src.predict import predict, input_frame

frame, model, metadata, _, _ = context()
hero("03 / Prediction laboratory", "Four measurements.<br>A new perspective.", "Build a numerical flower profile, run the saved model, and explore how its estimates respond. This laboratory classifies measurements in cm.")
presets = {"Default / reset": frame.iloc[0][FEATURES].to_dict()}
for species in LABELS:
    row = frame[frame.Species == species].iloc[0]
    presets[f"Observed {species.removeprefix('Iris-')} · CSV row {row.name + 1}"] = row[FEATURES].to_dict()
for feature in FEATURES:
    if f"input_{feature}" not in st.session_state:
        st.session_state[f"input_{feature}"] = float(presets["Default / reset"][feature])


def apply_preset():
    for feature, value in presets[st.session_state.preset].items():
        st.session_state[f"input_{feature}"] = float(value)
    st.session_state.pop("last_prediction", None)


def reset():
    st.session_state.preset = "Default / reset"
    apply_preset()


left, right = st.columns([1, 1.45], gap="large")
with left:
    st.subheader("Compose a flower")
    st.selectbox("Quick-start observed examples", list(presets), key="preset", on_change=apply_preset)
    st.caption("Presets are actual dataset rows for demonstration; they are not independent validation examples.")
    values = {}
    for feature in FEATURES:
        values[feature] = st.number_input(NAMES[feature], min_value=.01, max_value=100., step=.1, format="%.2f", key=f"input_{feature}")
        limits = metadata["training_ranges"][feature]
        st.caption(f"Training range: {limits['min']:.1f}–{limits['max']:.1f} cm")
    submitted = st.button("PREDICT FLOWER SPECIES", type="primary", width="stretch")
    st.button("Reset measurements", on_click=reset, width="stretch")
    if submitted:
        try:
            result = predict(model, metadata, values)
            st.session_state.last_prediction = result
            history = st.session_state.setdefault("prediction_history", [])
            history.append({"time_utc": datetime.now(timezone.utc).isoformat(), **values, "predicted_species": result["species"], **{f"p_{s}": p for s, p in result["probabilities"].items()}, "extrapolation": bool(result["outside_training_range"])})
        except ValueError as exc:
            st.error(str(exc))
with right:
    result = st.session_state.get("last_prediction")
    if result:
        if result["measurements"] != values:
            st.info("Inputs have changed. The result below describes the last submitted measurements; predict again to update it.")
        species = result["species"]
        st.markdown(f'<div class="species-card"><div class="eyebrow">Predicted species</div><h1 style="color:{COLORS[species]}">{species.removeprefix("Iris-").title()}</h1><p>{metadata["selected_model"]} · persisted local pipeline</p></div>', unsafe_allow_html=True)
        if result["outside_training_range"]:
            st.warning("Extrapolation: outside observed training ranges for " + ", ".join(NAMES[f] for f in result["outside_training_range"]))
        probabilities = pd.DataFrame({"Species": list(result["probabilities"]), "Probability estimate": list(result["probabilities"].values())})
        if not probabilities.empty:
            fig = px.bar(probabilities, x="Probability estimate", y="Species", color="Species", color_discrete_map=COLORS, orientation="h", text_auto=".1%")
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            chart(style(fig, "Class probability estimates"), "lab_probability")
        st.caption("Probability estimates are not guarantees of correctness and have not been independently calibrated. The predicted class follows the estimator's decision rule; SVM probabilities may rank classes differently.")
        st.dataframe(pd.DataFrame({"Measurement": [NAMES[f] for f in FEATURES], "Submitted (cm)": [result["measurements"][f] for f in FEATURES]}), hide_index=True, width="stretch")
        # Similarity is explicitly separate from the classifier's decision mechanism.
        training = frame.loc[metadata["train_indices"]]
        scaler = StandardScaler().fit(training[FEATURES])
        query = scaler.transform(input_frame(result["measurements"]))
        distances = np.linalg.norm(scaler.transform(training[FEATURES]) - query, axis=1)
        neighbors = training.copy().assign(standardized_distance=distances).sort_values("standardized_distance").head(3)
        with st.expander("Closest training examples · similarity context"):
            st.dataframe(neighbors, hide_index=True, width="stretch")
            st.caption("Euclidean distance after scaling with training means and standard deviations. Similarity is context, not a causal or model-specific explanation.")
    else:
        st.info("Choose an observed preset or enter your own measurements, then predict to inspect the model's response.")
        st.markdown("#### What does the model see?\nSepals form the outer floral structure; petals provide two further measurements. The pipeline receives these four numbers in a fixed order, applies its fitted preprocessing, and predicts one of three labels.")
st.divider()
st.subheader("What-if studio")
st.write("Sweep one measurement while holding the other three at the current input values. This shows the model's local response, not a biological causal effect.")
sweep_feature = st.selectbox("Vary a measurement", FEATURES, index=2, format_func=NAMES.get)
limits = metadata["training_ranges"][sweep_feature]
grid = pd.DataFrame([values] * 80, columns=FEATURES)
grid[sweep_feature] = np.linspace(limits["min"], limits["max"], 80)
if hasattr(model, "predict_proba"):
    probability = pd.DataFrame(model.predict_proba(grid), columns=model.classes_)
    probability[NAMES[sweep_feature]] = grid[sweep_feature]
    long = probability.melt(id_vars=NAMES[sweep_feature], var_name="Species", value_name="Probability estimate")
    fig = px.line(long, x=NAMES[sweep_feature], y="Probability estimate", color="Species", color_discrete_map=COLORS)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.add_vline(x=values[sweep_feature], line_dash="dot", annotation_text="Current input")
    chart(style(fig, "A local measurement response curve"), "lab_sweep", True)
st.caption("The swept feature stays inside its training range, but combinations of four individually plausible measurements can still be unfamiliar to the model.")
with st.expander("Session history & configuration comparison", expanded=bool(st.session_state.get("prediction_history"))):
    history = st.session_state.get("prediction_history", [])
    if history:
        records = pd.DataFrame(history)
        st.dataframe(records, width="stretch", hide_index=True)
        st.download_button("Download session predictions", records.to_csv(index=False), "iris_predictions.csv", "text/csv")
        if len(records) > 1:
            st.caption("Each row is an input configuration. Compare measurements, predicted labels, and all three class estimates side by side.")
        if st.button("Clear session history"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.caption("Predictions stay in this browser session. Submit a flower profile to start a history.")
