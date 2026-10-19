"""Cache resources by file signatures, refreshing after local retraining."""
import streamlit as st
from src.config import ROOT, DATA_PATH
from src.data_loader import load_data
from src.predict import load_bundle
from src.utils import read_json, sha256


def signature(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data
def cached_data(file_signature):
    return load_data()


@st.cache_resource
def cached_bundle(file_signature):
    return load_bundle()


@st.cache_data
def cached_report(name, file_signature):
    return read_json(ROOT / "reports" / "metrics" / name)


def context():
    try:
        frame = cached_data(signature(DATA_PATH))
        model_path = ROOT / "models" / "iris_pipeline.joblib"
        metadata_path = ROOT / "models" / "metadata.json"
        model, metadata = cached_bundle((signature(model_path), signature(metadata_path)))
        if sha256(DATA_PATH) != metadata["dataset_sha256"]:
            raise ValueError("The CSV has changed since training. Run python -m src.train to synchronize the experiment.")
        def report(name):
            return cached_report(name, signature(ROOT / "reports" / "metrics" / name))
        metrics = report("evaluation.json")
        comparison = report("model_comparison.json")
        if metadata.get("quick_mode"):
            st.warning("Development experiment: one hyperparameter candidate per model. Run python -m src.train for the full benchmark search.")
        return frame, model, metadata, metrics, comparison
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        st.error(str(exc))
        st.info("From the project folder, run `python -m src.download` then `python -m src.train`. The application never trains during page refresh.")
        st.stop()
