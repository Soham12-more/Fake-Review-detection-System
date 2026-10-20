"""
Predict a whole CSV of reviews at once (also used by the app's batch tab).

    cd src
    python predict.py reviews.csv --text-column text_ --rating-column rating -o predictions.csv

Options:
    --rating-column      column with 1-5 star ratings (omit to use --default-rating for every row)
    --default-rating     rating used when the rating column is missing or empty (default 5)
    --label-column       column with true labels (CG/OR or 0/1); prints accuracy if given
    --threshold          probability of "Original" needed to predict Original (default 0.5)

The output CSV is the input with four columns added:
    prob_original_lr, prob_original_dl, prob_original (the ensemble), prediction
"""
import argparse
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from features import to_frame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.joblib")
DL_MODEL_PATH = os.path.join(BASE_DIR, "dl_model.keras")

LABEL_NAMES = {0: "Computer-generated", 1: "Original"}
LABEL_ALIASES = {"cg": 0, "computer-generated": 0, "computer generated": 0, "0": 0, "0.0": 0,
                 "or": 1, "original": 1, "1": 1, "1.0": 1}


def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_dl_model():
    """
    The BiLSTM is optional: returns a function mapping a list of texts to
    probabilities of "Original", or None if the model or TensorFlow is unavailable.
    """
    if not os.path.exists(DL_MODEL_PATH):
        return None
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras
        import tensorflow as tf
        model = keras.models.load_model(DL_MODEL_PATH)
    except Exception:
        return None

    def predict(texts):
        inputs = tf.constant([[t] for t in texts], dtype=tf.string)
        return model.predict(inputs, batch_size=256, verbose=0).ravel().astype(float)

    return predict


def clean_ratings(ratings, default_rating):
    """Numeric 1-5 ratings; missing or unreadable values become default_rating."""
    r = pd.to_numeric(pd.Series(ratings), errors="coerce").fillna(default_rating)
    return r.clip(1, 5).to_numpy()


def predict_texts(pipeline, dl_predict, texts, ratings, threshold=0.5, chunk_size=1000, progress=None):
    """
    Predict many reviews in chunks. `progress(done, total)` is called after each chunk.
    Empty texts get no prediction (NaN). Returns a DataFrame aligned with `texts`.
    """
    texts = pd.Series(texts).fillna("").astype(str).reset_index(drop=True)
    ratings = pd.Series(ratings).reset_index(drop=True)
    out = pd.DataFrame(np.nan, index=texts.index,
                       columns=["prob_original_lr", "prob_original_dl", "prob_original"])
    valid = texts.str.strip().ne("").to_numpy()
    idx = np.flatnonzero(valid)

    for start in range(0, len(idx), chunk_size):
        rows = idx[start:start + chunk_size]
        lr = pipeline.predict_proba(to_frame(texts[rows], ratings[rows]))[:, 1]
        out.loc[rows, "prob_original_lr"] = lr
        if dl_predict is not None:
            dl = dl_predict(list(texts[rows]))
            out.loc[rows, "prob_original_dl"] = dl
            out.loc[rows, "prob_original"] = (lr + dl) / 2
        else:
            out.loc[rows, "prob_original"] = lr
        if progress is not None:
            progress(min(start + chunk_size, len(idx)), len(idx))

    out["prediction"] = np.where(out["prob_original"].isna(), "",
                                 np.where(out["prob_original"] >= threshold, LABEL_NAMES[1], LABEL_NAMES[0]))
    if dl_predict is None:
        out = out.drop(columns="prob_original_dl")
    return out


def evaluate(true_labels, predictions):
    """
    Accuracy etc. against true labels given as CG/OR or 0/1.
    Returns None if the labels can't be read.
    """
    y_true = pd.Series(true_labels).astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    y_pred = pd.Series(predictions).map({v: k for k, v in LABEL_NAMES.items()})
    keep = y_true.notna() & y_pred.notna()
    if keep.sum() == 0:
        return None
    y_true, y_pred = y_true[keep].astype(int), y_pred[keep].astype(int)
    return {
        "rows": int(keep.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict a CSV of reviews.")
    parser.add_argument("csv")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--rating-column")
    parser.add_argument("--default-rating", type=float, default=5)
    parser.add_argument("--label-column")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("-o", "--output", default="predictions.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    t0 = time.time()
    df = pd.read_csv(args.csv)
    for col in (args.text_column, args.rating_column, args.label_column):
        if col and col not in df.columns:
            parser.error(f"column '{col}' not in the CSV; columns are: {', '.join(df.columns)}")

    pipeline = load_pipeline()
    if pipeline is None:
        parser.error("model_pipeline.joblib not found; run train.py first")
    dl_predict = load_dl_model()
    print("models: logistic regression" + (" + BiLSTM (ensemble)" if dl_predict else " only (BiLSTM unavailable)"))

    ratings = clean_ratings(df[args.rating_column] if args.rating_column else [np.nan] * len(df),
                            args.default_rating)
    result = predict_texts(pipeline, dl_predict, df[args.text_column], ratings, args.threshold,
                           progress=lambda done, total: print(f"  {done}/{total} reviews", flush=True))
    out = pd.concat([df.reset_index(drop=True), result], axis=1)
    out.to_csv(args.output, index=False)

    counts = result["prediction"].value_counts()
    print(f"\n{len(df)} rows -> {args.output} in {time.time() - t0:.0f}s")
    for name in LABEL_NAMES.values():
        print(f"  {name}: {counts.get(name, 0)}")
    skipped = (result["prediction"] == "").sum()
    if skipped:
        print(f"  skipped (empty text): {skipped}")

    if args.label_column:
        m = evaluate(df[args.label_column], result["prediction"])
        if m is None:
            print("\nCould not read the labels (expected CG/OR or 0/1).")
        else:
            print(f"\nAgainst '{args.label_column}' ({m['rows']} rows): accuracy {m['accuracy']:.2%}, "
                  f"precision {m['precision']:.2%}, recall {m['recall']:.2%}, F1 {m['f1']:.2%}")
            print("confusion matrix (rows = true CG/OR, columns = predicted CG/OR):")
            print(m["confusion_matrix"])


if __name__ == "__main__":
    main()
