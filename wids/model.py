"""Train and use the classifier (scikit-learn Random Forest or Decision Tree)."""
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from .features import FEATURES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "joys_wids.joblib"


def metrics_for(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f,
            "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
            "report": pd.DataFrame(classification_report(
                y_true, y_pred, labels=[0, 1], target_names=["Normal", "Malicious"],
                output_dict=True, zero_division=0)).T.round(3)}


def train(feat: pd.DataFrame, algo="Random Forest", window=1.0, test_size=0.3, split="Random (stratified)", seed=42):
    X, y = feat[FEATURES], feat["label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("The training data has only one class. Use more rows so it includes both normal and attack traffic.")

    if split == "Chronological":                       # train on the past, test on the future
        order = feat.sort_values("t").index
        cut = int(len(order) * (1 - test_size))
        tr, te = order[:cut], order[cut:]
        if y[tr].nunique() < 2 or y[te].nunique() < 2:
            raise ValueError("A chronological split leaves one side without attack traffic. Use the random split or more rows.")
    else:
        tr, te = train_test_split(feat.index, test_size=test_size, stratify=y, random_state=seed)

    if algo == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=seed)
    clf.fit(X.loc[tr], y.loc[tr])

    m = metrics_for(y.loc[te], clf.predict(X.loc[te]))
    bundle = {"model": clf, "algo": algo, "window": window, "split": split, "metrics": m,
              "importances": dict(zip(FEATURES, clf.feature_importances_)),
              "n_train": len(tr), "n_test": len(te), "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "tree_rules": export_text(clf, feature_names=FEATURES, max_depth=4) if algo == "Decision Tree" else None}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_saved():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict_scores(bundle, feat: pd.DataFrame) -> np.ndarray:
    """Probability that each (window, device) row is malicious."""
    return bundle["model"].predict_proba(feat[FEATURES])[:, 1]
