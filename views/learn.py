import streamlit as st
from components.state import context
from components.ui import hero

_, _, metadata, _, _ = context()
hero("05 / Machine learning explained", "Understand every step.", "A field guide to classification, from labeled observations to honest evaluation. Start with the concepts; connect each one to the experiment you can inspect.")
st.graphviz_chart('digraph { rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#ede9fe" fontname="Arial"]; Measurements -> Validation -> Split; Split -> Training; Split -> "Untouched test"; Training -> "Grouped CV + tuning" -> Selection -> "Saved pipeline"; "Saved pipeline" -> "Final evaluation"; "Untouched test" -> "Final evaluation"; }')
lessons = {
    "01 · Supervised learning and classification": "Supervised learning uses examples with known answers. Here each observation pairs four flower measurements with a species label. Classification predicts a discrete category: setosa, versicolor or virginica. It does not generate a new measurement or analyze a photograph.",
    "02 · Features versus targets": "Features are inputs: sepal length, sepal width, petal length and petal width, in cm. Species is the target. Id identifies a record and has no botanical predictive role, so it is excluded. The pipeline always receives the same four named features in the same order.",
    "03 · Training versus testing": f"The {metadata['train_rows']} training rows support learning and model selection. The {metadata['test_rows']} test rows are reserved until selection is complete. An 80/20 split of distinct measurement groups is stratified by species; repeated observations stay together, so the exact row fraction can differ from 20%.",
    "04 · Why preprocessing matters": "Distance-based and regularized models depend on feature scale. StandardScaler subtracts each training feature's mean and divides by its training standard deviation. The same fitted transformation is applied to future measurements. Trees split individual features and do not need this scaling.",
    "05 · Data leakage": "Leakage occurs when information from validation or test observations influences learning. Fitting a scaler before splitting leaks information. This project places scaling inside each pipeline and fits it separately inside training CV folds. Identical measurement rows are also grouped so repeated observations cannot span folds.",
    "06 · Cross-validation and hyperparameter tuning": "Five stratified group folds rotate through training and validation roles. GridSearchCV compares predeclared settings using macro F1. The best candidate is refitted on the complete training partition. CV standard deviation describes fold variation; it is not a confidence interval. Tuning on the same folds can make the best CV score optimistic.",
    "07 · Overfitting": "A model overfits when it captures peculiarities of its training sample instead of general relationships. A deep tree can memorize many splits. Controlled search spaces, regularization, cross-validation and a separate holdout help detect or reduce this risk. A small benchmark cannot eliminate it.",
    "08 · Accuracy, precision, recall and F1": "Accuracy = correct / total. For one species, precision = true positives / predicted positives; recall = true positives / actual positives. F1 = 2 × precision × recall / (precision + recall). Macro F1 gives each species equal weight; weighted F1 weights species by their observed support. Undefined per-class scores are reported as zero.",
    "09 · How the selected model predicts": {
        "Logistic Regression": "The fitted scaler standardizes measurements. Logistic regression computes learned linear scores for each class and converts them into class probability estimates; the highest-scoring class is predicted.",
        "K-Nearest Neighbors": "The fitted scaler standardizes measurements. KNN finds nearby training flowers under its selected distance metric and combines their labels using uniform or distance weighting.",
        "Support Vector Machine": "The fitted scaler standardizes measurements. The SVM applies the selected linear or radial decision boundary. Class probabilities are separately estimated; their ordering may differ from the decision-based predicted label.",
        "Decision Tree": "The learned decision tree routes measurements through threshold questions until reaching a leaf. The leaf predicts its majority class; class fractions in that leaf yield probability estimates.",
        "Random Forest": "Each fitted tree routes the four measurements through learned thresholds. The forest averages tree class probability estimates and predicts the class with the highest mean estimate.",
    }[metadata['selected_model']],
    "10 · Uncertainty and explainability": "A high estimated probability is not proof of correctness. Sparse data, unfamiliar feature combinations and measurements outside the training range limit reliability. Permutation importance describes global model dependence. What-if curves show model sensitivity while holding inputs fixed, not causal biological effects.",
}
for title, text in lessons.items():
    with st.expander(title, expanded=title.startswith("01")):
        st.write(text)
st.subheader("A precision / recall thought experiment")
tp = st.slider("Correctly identified flowers of one species (true positives)", 0, 20, 8)
fp = st.slider("Other flowers incorrectly assigned to that species (false positives)", 0, 20, 2)
fn = st.slider("Flowers of that species missed (false negatives)", 0, 20, 3)
p = tp / (tp + fp) if tp + fp else 0
r = tp / (tp + fn) if tp + fn else 0
f = 2 * p * r / (p + r) if p + r else 0
for col, label, value in zip(st.columns(3), ["Precision", "Recall", "F1"], [p, r, f]):
    col.metric(label, f"{value:.1%}", border=True)
st.caption("This interactive arithmetic is an illustrative example, separate from the experiment's measured results.")
