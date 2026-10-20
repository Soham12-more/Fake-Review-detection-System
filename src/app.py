import streamlit as st
st.set_page_config(
    page_title="Fake Review Detection - Prediction",
    page_icon="🤖",
    layout="wide"
)

import os
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import altair as alt

from features import explain, to_frame
from predict import LABEL_NAMES, clean_ratings, evaluate, load_dl_model, load_pipeline, predict_texts

BASE_DIR = os.path.dirname(__file__)
METRICS_PATH = os.path.join(BASE_DIR, "..", "Reports", "metrics.json")
COLORS = alt.Scale(domain=["Original", "Computer-generated"], range=["#2a9d8f", "#e76f51"])
NONE = "(none)"

SIGNAL_LABELS = {
    "max_similarity": "Similarity to closest training review",
    "near_duplicate_count": "Near-duplicate training reviews",
    "sentiment_polarity": "Sentiment polarity (-1 to 1)",
    "sentiment_rating_mismatch": "Sentiment vs star-rating mismatch",
    "subjectivity": "Subjectivity (0 to 1)",
    "repeated_bigram_ratio": "Repeated phrases within the review",
    "lexical_diversity": "Lexical diversity (unique / total words)",
    "word_count": "Word count",
    "avg_word_length": "Average word length",
    "avg_sentence_length": "Average sentence length (words)",
    "flesch_reading_ease": "Flesch reading ease",
    "punctuation_ratio": "Punctuation ratio",
    "uppercase_ratio": "Uppercase ratio",
    "exclamation_count": "Exclamation marks",
    "rating": "Star rating",
}


# -------------------------------
# Load Models
# -------------------------------
pipeline = st.cache_resource(load_pipeline)()
if pipeline is None:
    st.error("Model file not found. Run `python train.py` inside the src folder to create 'model_pipeline.joblib'.")
    st.stop()
dl_predict = st.cache_resource(load_dl_model)()


def direction(value):
    return "Original" if value > 0 else "Computer-generated"


def guess_column(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


st.title("Fake Review Detection")
st.caption("Classifies reviews as written by a person (Original) or computer-generated, and explains why.")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    test = metrics["ensemble"]["test"] if dl_predict is not None and "ensemble" in metrics else metrics["test"]
    model_name = "ensemble (logistic regression + BiLSTM)" if dl_predict is not None else "logistic regression"
    st.caption(f"Model: {model_name}. Held-out test set: accuracy {test['accuracy']:.1%}, "
               f"precision {test['precision']:.1%}, recall {test['recall']:.1%}, F1 {test['f1']:.1%}")

tab_single, tab_batch = st.tabs(["Single review", "Batch (CSV file)"])

# -------------------------------
# Single Prediction
# -------------------------------
with tab_single:
    review_text = st.text_area("Enter review text:", height=150)
    col_rating, col_threshold = st.columns(2)
    rating = col_rating.select_slider("Star rating given with the review", options=[1, 2, 3, 4, 5], value=5)
    threshold = col_threshold.slider("Prediction threshold (probability of Original)", 0.0, 1.0, 0.5, 0.05)

    if st.button("Predict"):
        if not review_text.strip():
            st.error("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                lr_probabilities = pipeline.predict_proba(to_frame([review_text], [rating]))[0]
                probabilities = lr_probabilities
                if dl_predict is not None:
                    dl_original = float(dl_predict([review_text])[0])
                    dl_probabilities = np.array([1 - dl_original, dl_original])
                    probabilities = (lr_probabilities + dl_probabilities) / 2
                prediction = 1 if probabilities[1] >= threshold else 0
                words, signals = explain(pipeline, review_text, rating)
                similarity_step = pipeline.named_steps["features"].named_transformers_["similarity"].named_steps["sim"]
                neighbours = similarity_step.most_similar(review_text, k=3)

            result = LABEL_NAMES[prediction]
            confidence = probabilities[prediction]
            (st.success if prediction == 1 else st.warning)(f"Prediction: {result} ({confidence:.0%} confidence)")

            left, right = st.columns(2)

            with left:
                model_probs = {"Logistic regression": lr_probabilities}
                if dl_predict is not None:
                    model_probs["BiLSTM (deep learning)"] = dl_probabilities
                    model_probs["Ensemble (final)"] = probabilities
                df_probs = pd.DataFrame([
                    {"Model": name, "Label": label, "Probability": p[i]}
                    for name, p in model_probs.items()
                    for i, label in enumerate(["Computer-generated", "Original"])
                ])
                bar_chart = alt.Chart(df_probs).mark_bar().encode(
                    x=alt.X("Model", axis=alt.Axis(labelAngle=0, labelLimit=160), sort=list(model_probs), title=None),
                    xOffset="Label",
                    y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Label", scale=COLORS),
                    tooltip=["Model", "Label", alt.Tooltip("Probability", format=".1%")],
                ).properties(title="Prediction probability by model", height=300)
                st.altair_chart(bar_chart, use_container_width=True)

            with right:
                if not words.empty:
                    words["Pushes toward"] = words["contribution"].apply(direction)
                    word_chart = alt.Chart(words).mark_bar().encode(
                        x=alt.X("contribution", title="Contribution to decision"),
                        y=alt.Y("feature", sort="-x", title=None),
                        color=alt.Color("Pushes toward", scale=COLORS),
                        tooltip=["feature", alt.Tooltip("contribution", format=".3f"), "Pushes toward"],
                    ).properties(title="Words and phrases that influenced the decision", height=300)
                    st.altair_chart(word_chart, use_container_width=True)

            st.subheader("Why this result")
            top = signals.head(5)
            reasons = [f"- **{SIGNAL_LABELS.get(r.feature, r.feature)}** = {r.value:.2f} "
                       f"→ pushes toward *{direction(r.contribution)}*" for r in top.itertuples()]
            if not words.empty:
                top_word = words.iloc[0]
                reasons.append(f"- Strongest phrase: **\"{top_word['feature']}\"** "
                               f"→ pushes toward *{direction(top_word['contribution'])}*")
            st.markdown("\n".join(reasons))

            with st.expander("All signals"):
                table = signals.assign(
                    Signal=signals["feature"].map(SIGNAL_LABELS).fillna(signals["feature"]),
                    Value=signals["value"].round(3),
                    Contribution=signals["contribution"].round(3),
                    **{"Pushes toward": signals["contribution"].apply(direction)},
                )[["Signal", "Value", "Contribution", "Pushes toward"]]
                st.dataframe(table, hide_index=True, use_container_width=True)

            with st.expander("Most similar reviews in the training data"):
                for score, text in neighbours:
                    st.markdown(f"**Similarity {score:.2f}**")
                    st.write(text)

# -------------------------------
# Batch Prediction
# -------------------------------
with tab_batch:
    st.write("Upload a CSV with one review per row. Predictions run at roughly 100 reviews per second.")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            data = pd.read_csv(uploaded, encoding="latin-1")
        columns = list(data.columns)
        st.caption(f"{len(data):,} rows, {len(columns)} columns")
        st.dataframe(data.head(5), hide_index=True, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        text_guess = guess_column(columns, ["text_", "text", "review", "review_text", "content", "body"])
        text_col = c1.selectbox("Review text column", columns,
                                index=columns.index(text_guess) if text_guess else 0)
        rating_options = [NONE] + columns
        rating_guess = guess_column(columns, ["rating", "stars", "star_rating", "score"])
        rating_col = c2.selectbox("Star rating column (1-5)", rating_options,
                                  index=rating_options.index(rating_guess) if rating_guess else 0)
        label_guess = guess_column(columns, ["label", "class", "target"])
        label_col = c3.selectbox("True label column, optional (CG/OR or 0/1)", rating_options,
                                 index=rating_options.index(label_guess) if label_guess else 0)

        c4, c5 = st.columns(2)
        default_rating = c4.select_slider("Rating to use when missing", options=[1, 2, 3, 4, 5], value=5,
                                          help="The model uses the star rating; rows without one get this value.")
        batch_threshold = c5.slider("Prediction threshold (probability of Original)", 0.0, 1.0, 0.5, 0.05,
                                    key="batch_threshold")
        if rating_col == NONE:
            st.info("No rating column selected: every review is scored as if it had "
                    f"{default_rating} stars, which can change some predictions.")

        if st.button("Run predictions", type="primary"):
            bar = st.progress(0.0, text="Starting...")
            ratings = clean_ratings(data[rating_col] if rating_col != NONE else [np.nan] * len(data),
                                    default_rating)
            result = predict_texts(
                pipeline, dl_predict, data[text_col], ratings, batch_threshold,
                progress=lambda done, total: bar.progress(done / total, text=f"{done:,} / {total:,} reviews"))
            bar.empty()
            st.session_state["batch"] = {
                "file": uploaded.name,
                "output": pd.concat([data.reset_index(drop=True), result], axis=1),
                "text_col": text_col,
                "label_col": label_col,
            }

        batch = st.session_state.get("batch")
        if batch is not None and batch["file"] == uploaded.name:
            output = batch["output"]
            predicted = output["prediction"]
            n_cg = int((predicted == LABEL_NAMES[0]).sum())
            n_or = int((predicted == LABEL_NAMES[1]).sum())
            n_skipped = int((predicted == "").sum())

            st.subheader("Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Reviews scored", f"{n_cg + n_or:,}")
            m2.metric("Computer-generated", f"{n_cg:,}", f"{n_cg / max(n_cg + n_or, 1):.1%} of scored",
                      delta_color="off")
            m3.metric("Original", f"{n_or:,}", f"{n_or / max(n_cg + n_or, 1):.1%} of scored", delta_color="off")
            if n_skipped:
                st.caption(f"{n_skipped:,} rows had empty text and were skipped.")

            hist = alt.Chart(output.dropna(subset=["prob_original"])).mark_bar().encode(
                x=alt.X("prob_original", bin=alt.Bin(maxbins=20), title="Probability of Original (ensemble)"),
                y=alt.Y("count()", title="Reviews"),
                color=alt.Color("prediction", title="Prediction", scale=COLORS),
            ).properties(title="How confident the model is across the file", height=260)
            st.altair_chart(hist, use_container_width=True)

            if batch["label_col"] != NONE:
                m = evaluate(output[batch["label_col"]], predicted)
                if m is None:
                    st.warning(f"Couldn't read the labels in '{batch['label_col']}' (expected CG/OR or 0/1).")
                else:
                    st.subheader("Accuracy against your labels")
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Accuracy", f"{m['accuracy']:.2%}")
                    e2.metric("Precision", f"{m['precision']:.2%}")
                    e3.metric("Recall", f"{m['recall']:.2%}")
                    e4.metric("F1", f"{m['f1']:.2%}")
                    cm = pd.DataFrame(m["confusion_matrix"],
                                      index=["Actually computer-generated", "Actually original"],
                                      columns=["Predicted computer-generated", "Predicted original"])
                    st.dataframe(cm, use_container_width=True)
                    st.caption(f"Based on {m['rows']:,} rows with readable labels. If this file contains reviews "
                               "the model was trained on (such as the project's own dataset), these numbers "
                               "will be higher than on genuinely new reviews.")

            st.subheader("Predictions")
            show = output[[batch["text_col"], "prediction", "prob_original"]].rename(
                columns={batch["text_col"]: "Review", "prediction": "Prediction",
                         "prob_original": "Probability of Original"})
            st.dataframe(show.head(1000), hide_index=True, use_container_width=True)
            if len(show) > 1000:
                st.caption("Showing the first 1,000 rows; the download has all of them.")

            st.download_button(
                "Download predictions as CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name=os.path.splitext(batch["file"])[0] + "_predictions.csv",
                mime="text/csv",
            )
