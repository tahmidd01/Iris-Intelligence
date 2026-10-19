from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Iris.csv"
SOURCE = "https://www.kaggle.com/datasets/saurabh00007/iriscsv"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
LABELS = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
NAMES = {f: f.replace("LengthCm", " length (cm)").replace("WidthCm", " width (cm)") for f in FEATURES}
COLORS = dict(zip(LABELS, ["#8b5cf6", "#0eaaa0", "#ed9462"]))
SEED = 42
