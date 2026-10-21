"""
Train the fake review detection pipeline end to end.

    cd src
    python train.py

Writes:
    src/model_pipeline.joblib       - the full pipeline the Streamlit app loads
    Reports/metrics.json            - held-out test metrics (overall and per source), ablation, chosen C

Uses the Amazon dataset plus, if prepare_mage.py has been run, MAGE's Yelp reviews
(human-written and from 27 modern language models).
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
EXTERNAL = os.path.join(HERE, "..", "Data", "External")
MAGE_RATING = 5
# Off by default: adding MAGE's Yelp reviews lowered Amazon accuracy (ensemble 96.6% -> 95.1%) while
# catching only ~20% of modern AI reviews and none of the unseen GPT-4 ones. Modern AI text is handled
# by a pretrained transformer detector instead. Set to True to train with MAGE again.
USE_MAGE = False
REPORTS = os.path.join(HERE, "..", "Reports")
MODEL_OUT = os.path.join(HERE, "model_pipeline.joblib")
SEED = 42
C_GRID = [0.3, 1, 3, 10]
LABELS = {"CG": 0, "OR": 1}  # 0 = computer-generated, 1 = original


ALL_GROUPS = ("tfidf", "stats", "similarity")


def build_features(groups=ALL_GROUPS):
    """The feature groups to compute; train.py keeps the combination that scores best on validation."""
    transformers = {
        "tfidf": ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000,
                                           sublinear_tf=True, dtype=np.float32), "text"),
        "stats": ("stats", Pipeline([("stats", TextStats()), ("scale", StandardScaler())]), ["text", "rating"]),
        "similarity": ("similarity", Pipeline([("sim", CorpusSimilarity()), ("scale", StandardScaler())]), "text"),
    }
    return ColumnTransformer([transformers[g] for g in groups])


def column_groups(features):
    """Column index ranges of each feature group in the transformed matrix."""
    names = features.get_feature_names_out()
    return {g: np.where(np.char.startswith(names.astype(str), g + "__"))[0]
            for g in ALL_GROUPS}


def fit_lr(X, y, C):
    return LogisticRegression(C=C, solver="liblinear", max_iter=1000).fit(X, y)


def load_external(name):
    """
    A MAGE extract written by prepare_mage.py, as (X, y), or None if it isn't there.
    MAGE reviews have no star rating, so they get MAGE_RATING, the app's default too.
    """
    path = os.path.join(EXTERNAL, f"{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).dropna(subset=["text"])
    X = pd.DataFrame({"text": df["text"].astype(str), "rating": MAGE_RATING,
                      "source": "mage_yelp", "generator": df["generator"]})
    return X, df["label"].to_numpy()


def _concat(*parts):
    return (pd.concat([X for X, _ in parts], ignore_index=True),
            np.concatenate([y for _, y in parts]))


def load_splits(include_mage=USE_MAGE):
    """
    The fixed splits every model is trained and scored on (also used by train_dl.py).
    Returns (X_train, y_train, X_fit, y_fit, X_val, y_val, X_test, y_test);
    X_fit + X_val together make up X_train.

    The Amazon reviews are split 80/20 into train/test, and train again 85/15 into
    fit/validation. If prepare_mage.py has been run, MAGE's Yelp reviews are added
    using MAGE's own splits: train -> fit, valid -> validation, test -> test.
    X has a "source" column (amazon / mage_yelp) for reporting; the models ignore it.
    """
    df = pd.read_csv(DATA).drop_duplicates()
    df = df.rename(columns={"text_": "text"})
    X = df[["text", "rating"]].reset_index(drop=True).assign(source="amazon", generator="")
    y = df["label"].map(LABELS).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train,
                                                  random_state=SEED)

    mage = [load_external(f"mage_yelp_{s}") for s in ("train", "valid", "test")]
    if include_mage and all(m is not None for m in mage):
        X_fit, y_fit = _concat((X_fit, y_fit), mage[0])
        X_val, y_val = _concat((X_val, y_val), mage[1])
        X_test, y_test = _concat((X_test, y_test), mage[2])
        X_train, y_train = _concat((X_fit, y_fit), (X_val, y_val))
    return X_train, y_train, X_fit, y_fit, X_val, y_val, X_test, y_test


def scores_by_source(X, y, proba, threshold=0.5):
    """Accuracy etc. for the whole set and for each data source separately."""
    out = {}
    for name, mask in [("all", np.ones(len(X), bool))] + [(s, (X["source"] == s).to_numpy())
                                                          for s in sorted(X["source"].unique())]:
        preds = (proba[mask] >= threshold).astype(int)
        out[name] = {"n": int(mask.sum()),
                     "accuracy": round(accuracy_score(y[mask], preds), 4),
                     "precision": round(precision_score(y[mask], preds, zero_division=0), 4),
                     "recall": round(recall_score(y[mask], preds, zero_division=0), 4),
                     "f1": round(f1_score(y[mask], preds, zero_division=0), 4)}
    return out


def main():
    t0 = time.time()
    X_train, y_train, X_fit, y_fit, X_val, y_val, X_test, y_test = load_splits()
    print(f"train={len(X_fit)} val={len(X_val)} test={len(X_test)}")
    print("  by source (train/test):", X_train["source"].value_counts().to_dict(), "/",
          X_test["source"].value_counts().to_dict())

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

    # --- final model: the feature combination and C that scored best on validation, refit on all training data
    best_config = max(ablation, key=lambda name: ablation[name]["val_accuracy"])
    best_groups, best_C = configs[best_config], ablation[best_config]["C"]
    print(f"chosen on validation: {best_config} (C={best_C})")
    pipeline = Pipeline([("features", build_features(best_groups)),
                         ("clf", LogisticRegression(C=best_C, solver="liblinear", max_iter=1000))])
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "model": f"{' + '.join(best_groups)} features -> LogisticRegression",
        "feature_groups": list(best_groups),
        "C": best_C,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test": {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
        },
        "test_by_source": scores_by_source(X_test, y_test, proba),
        "data": {"train": X_train["source"].value_counts().to_dict(),
                 "test": X_test["source"].value_counts().to_dict()},
        "ablation_test_accuracy": ablation,
        "labels": {"0": "Computer-generated (CG)", "1": "Original (OR)"},
    }
    print(json.dumps(metrics["test_by_source"], indent=2))
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
