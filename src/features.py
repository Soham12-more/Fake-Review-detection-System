"""
Feature extraction shared by training (train.py) and the Streamlit app (app.py).

Everything the model needs is built from two inputs: the review text and its
star rating. Keeping the code in one module means the app computes features
exactly the way the model was trained on them.
"""
import re
import string

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_RE = re.compile(r"[.!?]+")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


def _syllables(word):
    """Rough syllable count: groups of vowels, minus a silent trailing 'e'."""
    word = word.lower()
    count = len(VOWEL_GROUP_RE.findall(word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _repeated_bigram_ratio(words):
    """Share of word bigrams in the review that occur more than once in it."""
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    counts = pd.Series(bigrams).value_counts()
    return float(counts[counts > 1].sum()) / len(bigrams)


STAT_FEATURES = [
    "word_count",
    "avg_word_length",
    "lexical_diversity",
    "repeated_bigram_ratio",
    "punctuation_ratio",
    "uppercase_ratio",
    "exclamation_count",
    "avg_sentence_length",
    "flesch_reading_ease",
    "sentiment_polarity",
    "subjectivity",
    "rating",
    "sentiment_rating_mismatch",
]


def text_stats(text, rating):
    """Hand-crafted style, sentiment and repetition features for one review."""
    text = text if isinstance(text, str) else ""
    words = WORD_RE.findall(text)
    lower_words = [w.lower() for w in words]
    n_words = len(words)
    n_chars = max(len(text), 1)
    n_sentences = max(len([s for s in SENTENCE_RE.split(text) if s.strip()]), 1)
    syllables = sum(_syllables(w) for w in words)

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    # Map 1-5 stars onto the same -1..1 scale as TextBlob polarity.
    rating_scaled = (float(rating) - 3.0) / 2.0

    return [
        n_words,
        (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        (len(set(lower_words)) / n_words) if n_words else 0.0,
        _repeated_bigram_ratio(lower_words),
        sum(c in string.punctuation for c in text) / n_chars,
        sum(c.isupper() for c in text) / n_chars,
        text.count("!"),
        n_words / n_sentences,
        (206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (syllables / n_words)) if n_words else 0.0,
        polarity,
        blob.sentiment.subjectivity,
        float(rating),
        abs(polarity - rating_scaled),
    ]


class TextStats(BaseEstimator, TransformerMixin):
    """Turns a DataFrame with 'text' and 'rating' columns into STAT_FEATURES."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([text_stats(t, r) for t, r in zip(X["text"], X["rating"])], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(STAT_FEATURES)


SIMILARITY_FEATURES = ["max_similarity", "near_duplicate_count"]


class CorpusSimilarity(BaseEstimator, TransformerMixin):
    """
    Compares each review with the reviews seen in training.

    max_similarity       - cosine similarity to the closest training review
    near_duplicate_count - training reviews with similarity >= near_duplicate_threshold

    During fit_transform each training review is compared with the *other*
    training reviews only, so a review never matches itself.
    """

    def __init__(self, near_duplicate_threshold=0.5, chunk_size=500):
        self.near_duplicate_threshold = near_duplicate_threshold
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        self.vectorizer_ = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english", dtype=np.float32)
        self.corpus_ = self.vectorizer_.fit_transform(list(X)).tocsr()
        self.corpus_text_ = list(X)
        return self

    def _scores(self, query, exclude_self=False):
        corpus_t = self.corpus_.T.tocsc()
        rows = []
        for start in range(0, query.shape[0], self.chunk_size):
            sims = (query[start:start + self.chunk_size] @ corpus_t).toarray()
            if exclude_self:
                idx = np.arange(sims.shape[0])
                sims[idx, start + idx] = 0.0
            rows.append(np.column_stack([
                sims.max(axis=1),
                (sims >= self.near_duplicate_threshold).sum(axis=1),
            ]))
        return np.vstack(rows).astype(np.float64)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self._scores(self.corpus_, exclude_self=True)

    def transform(self, X):
        return self._scores(self.vectorizer_.transform(list(X)))

    def most_similar(self, text, k=3):
        """The k closest training reviews to `text`, as (similarity, review) pairs."""
        sims = (self.vectorizer_.transform([text]) @ self.corpus_.T).toarray().ravel()
        top = np.argsort(sims)[::-1][:k]
        return [(float(sims[i]), self.corpus_text_[i]) for i in top]

    def get_feature_names_out(self, input_features=None):
        return np.array(SIMILARITY_FEATURES)


def to_frame(texts, ratings):
    """Build the two-column input the pipeline expects."""
    return pd.DataFrame({"text": list(texts), "rating": list(ratings)})


def explain(pipeline, text, rating, top_k=8):
    """
    Per-feature contributions to the model's decision for one review.

    The classifier is logistic regression, so each feature's contribution is
    simply its (transformed) value times its coefficient. Positive values push
    toward class 1 (Original), negative values toward class 0 (Computer-generated).
    Returns (word_contributions, feature_contributions) as DataFrames.
    """
    features = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    x = features.transform(to_frame([text], [rating]))
    x = sparse.csr_matrix(x)
    names = features.get_feature_names_out()
    idx = x.indices
    df = pd.DataFrame({"feature": names[idx], "contribution": x.data * clf.coef_[0][idx]})
    is_word = df["feature"].str.startswith("tfidf__")
    words = df[is_word].copy()
    words["feature"] = words["feature"].str.replace("tfidf__", "", regex=False)
    words = words.reindex(words["contribution"].abs().sort_values(ascending=False).index).head(top_k)

    raw = dict(zip(STAT_FEATURES + SIMILARITY_FEATURES,
                   np.concatenate([TextStats().transform(to_frame([text], [rating]))[0],
                                   features.named_transformers_["similarity"].named_steps["sim"]
                                   .transform([text])[0]])))
    other = df[~is_word].copy()
    other["feature"] = other["feature"].str.split("__").str[-1]
    other["value"] = other["feature"].map(raw)
    other = other.reindex(other["contribution"].abs().sort_values(ascending=False).index)
    return words.reset_index(drop=True), other.reset_index(drop=True)
