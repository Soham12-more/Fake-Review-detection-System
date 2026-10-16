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
import joblib
import numpy as np
import pandas as pd
import altair as alt

from features import explain, to_frame

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.joblib")
DL_MODEL_PATH = os.path.join(BASE_DIR, "dl_model.keras")
METRICS_PATH = os.path.join(BASE_DIR, "..", "Reports", "metrics.json")

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
# Load Model
# -------------------------------
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_dl_model():
    """The BiLSTM is optional: without it (or without TensorFlow) the app uses the pipeline alone."""
    if not os.path.exists(DL_MODEL_PATH):
        return None
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras
        import tensorflow as tf
        model = keras.models.load_model(DL_MODEL_PATH)
        return lambda text: float(model(tf.constant([[text]], dtype=tf.string), training=False).numpy()[0][0])
    except Exception:
        return None


pipeline = load_pipeline()
if pipeline is None:
    st.error("Model file not found. Run `python train.py` inside the src folder to create 'model_pipeline.joblib'.")
    st.stop()
dl_model = load_dl_model()


def direction(value):
    return "Original" if value > 0 else "Computer-generated"


# -------------------------------
# Single Prediction Section
# -------------------------------
st.title("Fake Review Detection - Single Prediction")
st.caption("Classifies a review as written by a person (Original) or computer-generated, and explains why.")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    test = metrics["ensemble"]["test"] if dl_model is not None and "ensemble" in metrics else metrics["test"]
    model_name = "ensemble (logistic regression + BiLSTM)" if dl_model is not None else "logistic regression"
    st.caption(f"Model: {model_name}. Held-out test set: accuracy {test['accuracy']:.1%}, "
               f"precision {test['precision']:.1%}, recall {test['recall']:.1%}, F1 {test['f1']:.1%}")

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
            if dl_model is not None:
                dl_original = dl_model(review_text)
                dl_probabilities = np.array([1 - dl_original, dl_original])
                probabilities = (lr_probabilities + dl_probabilities) / 2
            prediction = 1 if probabilities[1] >= threshold else 0
            words, signals = explain(pipeline, review_text, rating)
            similarity_step = pipeline.named_steps["features"].named_transformers_["similarity"].named_steps["sim"]
            neighbours = similarity_step.most_similar(review_text, k=3)

        result = "Original" if prediction == 1 else "Computer-generated"
        confidence = probabilities[prediction]
        (st.success if prediction == 1 else st.warning)(f"Prediction: {result} ({confidence:.0%} confidence)")

        left, right = st.columns(2)

        with left:
            model_probs = {"Logistic regression": lr_probabilities}
            if dl_model is not None:
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
                color=alt.Color("Label", scale=alt.Scale(
                    domain=["Original", "Computer-generated"], range=["#2a9d8f", "#e76f51"])),
                tooltip=["Model", "Label", alt.Tooltip("Probability", format=".1%")],
            ).properties(title="Prediction probability by model", height=300)
            st.altair_chart(bar_chart, use_container_width=True)

        with right:
            if not words.empty:
                words["Pushes toward"] = words["contribution"].apply(direction)
                word_chart = alt.Chart(words).mark_bar().encode(
                    x=alt.X("contribution", title="Contribution to decision"),
                    y=alt.Y("feature", sort="-x", title=None),
                    color=alt.Color("Pushes toward", scale=alt.Scale(
                        domain=["Original", "Computer-generated"], range=["#2a9d8f", "#e76f51"])),
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
