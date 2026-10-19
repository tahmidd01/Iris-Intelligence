"""Launch with: python -m streamlit run app.py."""
import streamlit as st
from components.ui import theme
from src.config import ROOT

st.set_page_config(page_title="Iris Intelligence", page_icon="✿", layout="wide", initial_sidebar_state="expanded")
theme()
if (ROOT / "assets" / "iris-mark.svg").exists():
    st.logo(str(ROOT / "assets" / "iris-mark.svg"), size="large")
pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/dataset.py", title="Dataset Explorer", icon=":material/table_chart:"),
    st.Page("views/analytics.py", title="Visual Analytics", icon=":material/scatter_plot:"),
    st.Page("views/lab.py", title="Prediction Lab", icon=":material/science:"),
    st.Page("views/performance.py", title="Model Performance", icon=":material/monitoring:"),
    st.Page("views/learn.py", title="ML Explained", icon=":material/school:"),
    st.Page("views/about.py", title="About the Project", icon=":material/info:"),
]
with st.sidebar:
    st.caption("BOTANICAL DATA · HUMAN UNDERSTANDING")
navigation = st.navigation(pages)
with st.sidebar:
    st.divider()
    st.caption("Four measurements. Three species.\nOne reproducible experiment.")
    st.caption("LOCAL RESEARCH WORKSPACE · v1.0")
navigation.run()
