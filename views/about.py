import streamlit as st
from components.state import context
from components.ui import hero, file_download
from src.config import ROOT, SOURCE

frame, _, metadata, metrics, _ = context()
hero("06 / About the project", "Small data. Serious practice.", "Iris Intelligence is an educational, reproducible machine learning platform built around a transparent numerical classification task.")
st.subheader("The objective")
st.write("Turn four measured flower characteristics into a species prediction, while making data quality, model development and evaluation visible enough to question and reproduce.")
st.link_button("View the required Kaggle dataset ↗", SOURCE)
st.write(f"The current dataset contains {len(frame)} flowers across three classes. Source status: {metadata['provenance']['method']}.")
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Methodology")
    st.write("Validate the CSV; exclude Id; retain observations; group identical measurements; stratify an 80/20 group split; tune five sklearn pipelines using grouped five-fold CV; select by macro F1; evaluate only the winner on the holdout; persist the fitted training pipeline and experiment metadata.")
    st.subheader("Technology")
    st.write("Python · Pandas · NumPy · Scikit-learn · SciPy · Plotly · Matplotlib · Seaborn · Streamlit · Joblib · Pytest · Jupyter")
    st.subheader("Reproducibility")
    st.write(f"Seed {metadata['random_seed']}. Exact source and model checksums, feature order, package versions, train/test row indices, search spaces and validation reports are recorded locally. Run `python -m src.train` to regenerate the experiment.")
with right:
    st.subheader("Findings")
    st.write(f"{metadata['selected_model']} was selected by training CV. Its holdout accuracy was {metrics['accuracy']:.2%}, with macro F1 {metrics['macro_f1']:.4f}. Petal features show strong visual separation for setosa; the other two species overlap more.")
    st.subheader("Limitations")
    st.write("A 150-row historical benchmark represents a narrow sample. Split-dependent performance and uncalibrated probability estimates limit certainty. Range checks cannot detect all unfamiliar feature combinations. The model is restricted to three labels and numerical measurements.")
    st.subheader("Future work")
    st.write("Validate externally collected flowers, quantify performance with nested CV, assess calibration on independent data, and study measurement error and distribution shift.")
st.subheader("Research artifacts")
file_download(ROOT / "reports" / "results.md", "Download experiment report")
file_download(ROOT / "output" / "pdf" / "iris_intelligence_report.pdf", "Download printable PDF report")
file_download(ROOT / "models" / "metadata.json", "Download model metadata")
st.caption("The iris mark is an original decorative vector graphic, not a scientific image. The application needs no external images or cloud services at runtime. Load joblib artifacts only from trusted local sources.")
