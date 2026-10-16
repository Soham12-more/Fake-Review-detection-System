"""
Train the fake review detection pipeline end to end.

    cd src
    python train.py

Writes:
    src/model_pipeline.joblib       - the full pipeline the Streamlit app loads
    Reports/metrics.json            - held-out test metrics, ablation, chosen C
    Reports/confusion_matrix_pipeline.png
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import CorpusSimilarity, TextStats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data", "Raw", "fakeReviewData.csv")
REPORTS = os.path.join(HERE, "..", "Reports")
MODEL_OUT = os.path.join(HERE, "model_pipeline.joblib")
SEED = 42
C_GRID = [0.3, 1, 3, 10]
LABELS = {"CG": 0, "OR": 1}  # 0 = computer-generated, 1 = original


def build_features():
    return ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000,
                                  sublinear_tf=True, dtype=np.float32), "text"),
        ("stats", Pipeline([("stats", TextStats()), ("scale", StandardScaler())]), ["text", "rating"]),
        ("similarity", Pipeline([("sim", CorpusSimilarity()), ("scale", StandardScaler())]), "text"),
    ])


def column_groups(features):
    """Column index ranges of each feature group in the transformed matrix."""
    names = features.get_feature_names_out()
    return {g: np.where(np.char.startswith(names.astype(str), g + "__"))[0]
            for g in ("tfidf", "stats", "similarity")}


def fit_lr(X, y, C):
    return LogisticRegression(C=C, solver="liblinear", max_iter=1000).fit(X, y)


def load_splits():
    """
    The fixed splits every model is trained and scored on (also used by train_dl.py).
    Returns (X_train, y_train, X_fit, y_fit, X_val, y_val, X_test, y_test);
    X_fit + X_val together make up X_train.
    """
    df = pd.read_csv(DATA).drop_duplicates()
    df = df.rename(columns={"text_": "text"})
    X = df[["text", "rating"]].reset_index(drop=True)
    y = df["label"].map(LABELS).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train,
                                                  random_state=SEED)
    return X_train, y_train, X_fit, y_fit, X_val, y_val, X_test, y_test


def main():
    t0 = time.time()
    X_train, y_train, X_fit, y_fit, X_val, y_val, X_test, y_test = load_splits()
    print(f"train={len(X_fit)} val={len(X_val)} test={len(X_test)}")

    # --- choose C per feature set on the validation split, report each on test (ablation)
    feats = build_features()
    F_fit = sparse.csr_matrix(feats.fit_transform(X_fit))
    F_val = sparse.csr_matrix(feats.transform(X_val))
    F_test = sparse.csr_matrix(feats.transform(X_test))
    groups = column_groups(feats)
    print(f"features built in {time.time() - t0:.0f}s, shape {F_fit.shape}")

    configs = {
        "tfidf_only": ["tfidf"],
        "tfidf+stats": ["tfidf", "stats"],
        "tfidf+similarity": ["tfidf", "similarity"],
        "all": ["tfidf", "stats", "similarity"],
        "stats+similarity_only": ["stats", "similarity"],
    }
    ablation = {}
    for name, gs in configs.items():
        cols = np.concatenate([groups[g] for g in gs])
        val_acc = {C: accuracy_score(y_val, fit_lr(F_fit[:, cols], y_fit, C).predict(F_val[:, cols]))
                   for C in C_GRID}
        best_C = max(val_acc, key=val_acc.get)
        test_acc = accuracy_score(y_test, fit_lr(F_fit[:, cols], y_fit, best_C).predict(F_test[:, cols]))
        ablation[name] = {"C": best_C, "val_accuracy": round(val_acc[best_C], 4), "test_accuracy": round(test_acc, 4)}
        print(f"  {name:24s} C={best_C:<4} val={val_acc[best_C]:.4f} test={test_acc:.4f}")

    # --- final model: all features, best C, refit on the full training split
    best_C = ablation["all"]["C"]
    pipeline = Pipeline([("features", build_features()), ("clf", LogisticRegression(C=best_C, solver="liblinear",
                                                                                     max_iter=1000))])
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "model": "TF-IDF (1-2 grams) + style/sentiment/repetition stats + corpus similarity -> LogisticRegression",
        "C": best_C,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test": {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
        },
        "ablation_test_accuracy": ablation,
        "labels": {"0": "Computer-generated (CG)", "1": "Original (OR)"},
    }
    print(json.dumps(metrics["test"], indent=2))
    print(classification_report(y_test, preds, target_names=["Computer-generated", "Original"], digits=4))

    os.makedirs(REPORTS, exist_ok=True)
    metrics_path = os.path.join(REPORTS, "metrics.json")
    if os.path.exists(metrics_path):
        # keep the deep learning results written by train_dl.py; rerun it to refresh the ensemble
        with open(metrics_path) as f:
            previous = json.load(f)
        metrics.update({k: previous[k] for k in ("deep_learning", "ensemble") if k in previous})
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Computer-generated", "Original"])
    ax.set_yticks([0, 1], ["Computer-generated", "Original"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Test set (n={len(y_test)}), accuracy {metrics['test']['accuracy']:.1%}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "confusion_matrix_pipeline.png"), dpi=120)

    joblib.dump(pipeline, MODEL_OUT, compress=3)
    print(f"saved {MODEL_OUT} ({os.path.getsize(MODEL_OUT) / 1e6:.1f} MB) in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
