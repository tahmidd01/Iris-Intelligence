"""Optional printable report. Install requirements-report.txt, then run this module."""
import csv
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from src.config import ROOT, LABELS, SOURCE
from src.utils import read_json


def main() -> None:
    meta = read_json(ROOT / "models" / "metadata.json")
    metrics = read_json(ROOT / "reports" / "metrics" / "evaluation.json")
    comparison = read_json(ROOT / "reports" / "metrics" / "model_comparison.json")
    audit = read_json(ROOT / "reports" / "metrics" / "data_validation.json")
    violet, teal, ink = colors.HexColor("#7c3aed"), colors.HexColor("#078c86"), colors.HexColor("#202336")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyIris", fontName="Helvetica", fontSize=9.5, leading=14, textColor=ink, spaceAfter=9))
    styles.add(ParagraphStyle(name="TitleIris", fontName="Helvetica-Bold", fontSize=32, leading=36, textColor=ink, spaceAfter=16))
    styles.add(ParagraphStyle(name="HeadingIris", fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=violet, spaceBefore=10, spaceAfter=9))
    styles.add(ParagraphStyle(name="CaptionIris", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#64677b"), spaceAfter=8))
    styles.add(ParagraphStyle(name="MetricIris", fontName="Helvetica-Bold", fontSize=20, leading=27, textColor=violet, alignment=TA_CENTER))
    content = []
    def p(text, style="BodyIris"):
        return Paragraph(text, styles[style])
    def add(text, style="BodyIris"):
        content.append(p(text, style))
    def heading(text):
        add(text, "HeadingIris")
    def table(rows, widths, header=True):
        wrapped = [[p(str(cell), "CaptionIris") for cell in row] for row in rows]
        block = Table(wrapped, colWidths=widths, hAlign="LEFT")
        rules = [("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 10),
                 ("RIGHTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 5),
                 ("BOTTOMPADDING", (0,0), (-1,-1), 4), ("LINEBELOW", (0,0), (-1,-1), .4, colors.HexColor("#e5e2ed"))]
        if header:
            rules.append(("BACKGROUND", (0,0), (-1,0), colors.HexColor("#ede9fe")))
        block.setStyle(TableStyle(rules))
        content.append(block)
        content.append(Spacer(1, 10))
    def figure(name, width):
        path = ROOT / "reports" / "figures" / name
        w, h = ImageReader(str(path)).getSize()
        content.append(Image(str(path), width=width, height=width*h/w, hAlign="CENTER"))
    add("REPRODUCIBLE MACHINE LEARNING / TASK 1", "CaptionIris")
    content.append(Spacer(1, 18))
    add("IRIS<br/>INTELLIGENCE", "TitleIris")
    add("Explore the Data. Understand the Model. Predict the Species.")
    add("An interactive Iris flower classification and machine learning analytics platform", "CaptionIris")
    content.append(Spacer(1, 16))
    cards = Table([[p(f"{metrics['accuracy']:.2%}", "MetricIris"), p(f"{metrics['macro_f1']:.4f}", "MetricIris"), p(f"{metrics['correct']}/{metrics['test_rows']}", "MetricIris")],
                   [p("Held-out accuracy", "CaptionIris"), p("Held-out macro F1", "CaptionIris"), p("Correct predictions", "CaptionIris")]], colWidths=[167]*3)
    cards.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f5f3ff")),("ALIGN",(0,0),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,0),15),("BOTTOMPADDING",(0,-1),(-1,-1),12)]))
    content.append(cards)
    heading("Abstract and objective")
    add(f"Five classical classifiers were trained and tuned to predict one of three Iris species from four measurements in centimeters. {escape(meta['selected_model'])} was selected using training-only cross-validated macro F1. The independent holdout yielded {metrics['accuracy']:.2%} accuracy. This is evidence for a specific small benchmark split, not a claim of universal botanical reliability.")
    heading("The authentic dataset")
    table([["Source", "Kaggle / saurabh00007/iriscsv"], ["Observed sample", f"{audit['rows']} flowers; 50 per species"], ["Inputs", "Sepal length, sepal width, petal length, petal width (cm)"], ["Target", "Iris-setosa, Iris-versicolor, Iris-virginica"], ["Integrity", f"No missing values; {audit['duplicate_measurement_rows']} repeated measurement rows retained; Id excluded"]], [115,386], header=False)
    heading("Complete workflow")
    add("Acquire and validate the CSV; explore measurements; group identical observations; reserve the holdout; tune five pipelines within training CV; select one model; evaluate once; save the pipeline and reports; support interactive numerical inference.")
    add(f"Acquisition: {escape(meta['provenance']['method'])}. The source checksum, exact row indices and package versions are included in the machine-readable experiment metadata.", "CaptionIris")
    content.append(PageBreak())
    heading("Methodology and experimental setup")
    add(f"Seed {meta['random_seed']}. An 80/20 stratified split of distinct four-feature measurement groups produced {meta['train_rows']} training rows and {meta['test_rows']} test rows. Identical measurements cannot cross boundaries. Five-fold shuffled StratifiedGroupKFold is used inside GridSearchCV. All observations remain; no imputation or arbitrary outlier removal occurs.")
    add("Logistic regression, KNN and SVM use StandardScaler inside their pipelines. Trees use passthrough. Scaling and tuning see training data only. The saved pipeline is fitted on the training partition and is never refitted on test observations.")
    table([["Algorithm", "Hyperparameters explored"], ["Logistic regression", "C: 0.1, 1, 10"], ["KNN", "k: 3, 5, 7, 9; uniform/distance; p: 1, 2"], ["SVM", "Linear/RBF; C: 0.1, 1, 10; RBF gamma: scale, 0.1"], ["Decision tree", "Depth: 2, 3, unlimited; min leaf: 1, 3"], ["Random forest", "Trees: 100, 200; depth: 3, unlimited; min leaf: 1, 2"]], [135,366])
    heading("Actual cross-validation comparison")
    table([["Algorithm", "Macro F1", "Fold SD", "Accuracy"]] + [[r["model"], f"{r['cv_macro_f1']:.4f}", f"{r['cv_std']:.4f}", f"{r['cv_accuracy']:.4f}"] for r in comparison], [225,92,92,92])
    figure("model_comparison.png", 380)
    add(f"Winner: <b>{escape(meta['selected_model'])}</b>. Best settings: {escape(str(meta['winning_cv']['best_params']))}. Macro F1 is the primary criterion; numerical ties use CV accuracy then a fixed simplicity order. The tiny observed advantage is not evidence of general superiority.", "CaptionIris")
    add("Fold SD describes variation, not a confidence interval. Tuned CV scores are optimistic selection estimates. Only the selected model receives a holdout evaluation.", "CaptionIris")
    content.append(PageBreak())
    heading("Independent held-out evaluation")
    table([["Metric", "Macro average", "Weighted average"], ["Precision", f"{metrics['macro_precision']:.4f}", f"{metrics['weighted_precision']:.4f}"], ["Recall", f"{metrics['macro_recall']:.4f}", f"{metrics['weighted_recall']:.4f}"], ["F1", f"{metrics['macro_f1']:.4f}", f"{metrics['weighted_f1']:.4f}"]], [181,160,160])
    add("Accuracy is the fraction correct. Precision measures correctness among predicted members of a class; recall measures recovery among actual members. F1 balances precision and recall. Macro averages treat species equally; weighted averages account for support.")
    report = metrics["classification_report"]
    table([["Species", "Precision", "Recall", "F1", "Support"]] + [[label, f"{report[label]['precision']:.3f}", f"{report[label]['recall']:.3f}", f"{report[label]['f1-score']:.3f}", int(report[label]["support"])] for label in LABELS], [181,80,80,80,80])
    figure("confusion_matrix.png", 335)
    heading("Real error analysis")
    with (ROOT / "reports" / "metrics" / "errors.csv").open(newline="", encoding="utf-8") as stream:
        errors = list(csv.DictReader(stream))
    if errors:
        for error in errors:
            add(f"CSV data row {error['row_index']} (zero-based): actual <b>{escape(error['actual'])}</b>; predicted <b>{escape(error['predicted'])}</b>. Measurements: sepal {error['SepalLengthCm']} x {error['SepalWidthCm']} cm; petal {error['PetalLengthCm']} x {error['PetalWidthCm']} cm.", "CaptionIris")
        add("Species overlap is a plausible explanation for ambiguity, not a proven cause of this particular error. All held-out examples and normalized confusion counts are available in the dashboard and CSV/JSON outputs.", "CaptionIris")
    else:
        add("No mistakes occurred on this particular holdout. No errors are invented; future performance is not guaranteed.")
    content.append(PageBreak())
    heading("Feature relevance and scientific discussion")
    figure("permutation_importance.png", 390)
    add("Permutation importance shuffles one held-out feature at a time and measures the decrease in macro F1 over 30 repetitions. Error bars show shuffle variation. Correlated features can substitute for each other; these are global model diagnostics, not causal or per-prediction explanations. Importance was computed after selection and did not inform retuning.")
    add("Descriptive EDA shows strong petal-feature separation for setosa and more overlap between versicolor and virginica. The full-dataset PCA is an exploratory map only; it is not used by the predictive pipeline. Similar training examples and what-if curves provide context and sensitivity, not causal evidence.")
    heading("Limitations and future improvements")
    add("The 150-row historical benchmark is small, balanced and restricted to three species. One 30-row holdout has substantial sampling variation. Class probability estimates have not undergone an independent calibration study. Range checks flag extrapolation but cannot detect every unfamiliar feature combination. The classifier receives measurements, never photographs.")
    add("Future work should add externally measured flowers, nested cross-validation, independent calibration and measurement-noise analysis. The current system prioritizes honest evaluation, source traceability and understandable local inference.")
    heading("Conclusion and reproducibility")
    add("The project implements the complete classification lifecycle and exposes it through seven local Streamlit workspaces. Reproduce using the pinned Python environment, authentic source CSV, seed 42 and recorded split/search settings. Run <b>python -m src.train</b> and <b>python -m pytest -q</b>. The source, metadata, executed notebook and Markdown report supply the complete audit trail.")
    heading("References")
    for title, url in [("Required Kaggle Iris.csv", SOURCE), ("Scikit-learn leakage prevention", "https://scikit-learn.org/stable/common_pitfalls.html"), ("Scikit-learn permutation importance", "https://scikit-learn.org/stable/modules/permutation_importance.html"), ("Streamlit documentation", "https://docs.streamlit.io/")]:
        add(f'<link href="{url}" color="#7c3aed">{escape(title)}</link>', "CaptionIris")
    directory = ROOT / "output" / "pdf"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "iris_intelligence_report.pdf"
    def decorate(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#e5e2ed"))
        canvas.line(47, 43, 548, 43)
        canvas.setFillColor(colors.HexColor("#64677b"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(47, 29, "IRIS INTELLIGENCE / Reproducible experiment / Seed 42")
        canvas.drawRightString(548, 29, f"{doc.page}")
    document = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=47, leftMargin=47, topMargin=40, bottomMargin=58,
                                 title="Iris Intelligence - Experimental Report", author="Iris Intelligence")
    document.build(content, onFirstPage=decorate, onLaterPages=decorate)
    print(path)


if __name__ == "__main__":
    main()
