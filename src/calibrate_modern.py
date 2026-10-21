"""
Choose the threshold for the modern-AI check and measure the combined system.

    cd src
    python calibrate_modern.py            # scores ~1,800 reviews with the Longformer (~45 min on a CPU)
    python calibrate_modern.py --reuse    # reuse Reports/modern_detector_scores.csv

A review is flagged as computer-generated if EITHER check flags it:
  - the project's ensemble (GPT-2-era patterns): probability of Original below the threshold
    chosen by choose_threshold.py, or
  - the modern-AI detector (modern_detector.py): probability of AI at or above the threshold chosen here.

The modern threshold is the lowest one that flags at most 1% of the validation human reviews
(Amazon and Yelp). It is then checked on test reviews: Amazon, Yelp (MAGE), and GPT-4 IMDb reviews
from MAGE's out-of-distribution set, which no model in this project was trained on.

Writes Reports/modern_detector_scores.csv and the "modern_ai" section of Reports/metrics.json.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from modern_detector import REPO, ModernAIDetector
from predict import default_threshold, load_dl_model, load_metrics, load_pipeline, predict_texts
from train import REPORTS, load_external, load_splits

SCORES = os.path.join(REPORTS, "modern_detector_scores.csv")
MAX_FALSE_FLAG_RATE = 0.01
SEED = 0


def sample_reviews():
    _, _, _, _, X_val, y_val, X_test, y_test = load_splits(include_mage=False)
    val, test = X_val.assign(label=y_val), X_test.assign(label=y_test)
    mv, mt, oo = (load_external(n) for n in ("mage_yelp_valid", "mage_yelp_test", "mage_ood_gpt4_imdb"))
    if mv is None or mt is None or oo is None:
        raise SystemExit("MAGE files missing: run prepare_mage.py first")
    mv, mt, oo = (X.assign(label=y) for X, y in (mv, mt, oo))
    chatgpt = (mt.label == 0) & (mt.generator == "gpt-3.5-turbo")

    def take(df, n, split, group):
        d = df.sample(min(n, len(df)), random_state=SEED) if n else df
        return d.assign(split=split, group=group)[["text", "rating", "label", "split", "group"]]

    return pd.concat([
        take(val[val.label == 1], 300, "validation", "amazon_human"),
        take(mv[mv.label == 1], 150, "validation", "yelp_human"),
        take(mv[mv.label == 0], 150, "validation", "yelp_ai"),
        take(test[test.label == 1], 300, "test", "amazon_human"),
        take(test[test.label == 0], 150, "test", "amazon_gpt2_ai"),
        take(mt[mt.label == 1], 150, "test", "yelp_human"),
        take(mt[chatgpt], 0, "test", "yelp_ai_chatgpt"),
        take(mt[(mt.label == 0) & ~chatgpt], 100, "test", "yelp_ai_other"),
        take(oo[oo.label == 1], 0, "test", "imdb_human"),
        take(oo[oo.label == 0], 0, "test", "imdb_gpt4_ai"),
    ], ignore_index=True)


def score(rows):
    t0 = time.time()
    rows["p_original_ensemble"] = predict_texts(load_pipeline(), load_dl_model(), rows.text,
                                                rows.rating)["prob_original"].to_numpy()
    print(f"loading {REPO} ...")
    rows["p_ai_longformer"] = ModernAIDetector().predict(
        rows.text.tolist(), progress=lambda done, total: done % 200 == 0 and print(
            f"  {done}/{total} reviews ({time.time() - t0:.0f}s)", flush=True))
    out = rows.assign(text=rows.text.str[:300])
    out.to_csv(SCORES, index=False)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="reuse saved scores instead of re-scoring")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    rows = pd.read_csv(SCORES) if args.reuse else score(sample_reviews())
    val, test = rows[rows.split == "validation"], rows[rows.split == "test"]
    human = val[val.label == 1].p_ai_longformer.to_numpy()

    candidates = np.unique(np.round(human, 4))
    ok = [t for t in np.r_[candidates, 1.0] if (human >= t).mean() <= MAX_FALSE_FLAG_RATE]
    modern_t = float(min(ok))
    classic_t = default_threshold()

    by_group = {}
    for group, df in test.groupby("group"):
        classic = df.p_original_ensemble < classic_t
        modern = df.p_ai_longformer >= modern_t
        is_human = int(df.label.iloc[0]) == 1
        by_group[group] = {"n": int(len(df)), "kind": "genuine" if is_human else "computer-generated",
                           "flagged_by_classic": round(float(classic.mean()), 4),
                           "flagged_by_modern": round(float(modern.mean()), 4),
                           "flagged_combined": round(float((classic | modern).mean()), 4)}

    metrics = load_metrics()
    metrics["modern_ai"] = {
        "detector": REPO,
        "threshold": modern_t,
        "classic_threshold": classic_t,
        "rule": "computer-generated if the ensemble's probability of Original < classic_threshold "
                "OR the modern detector's probability of AI >= threshold",
        "chosen_on": f"validation human reviews (n={len(human)}): flag at most {MAX_FALSE_FLAG_RATE:.0%}",
        "validation_humans_flagged": round(float((human >= modern_t).mean()), 4),
        "test": by_group,
    }
    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"modern-AI threshold: {modern_t} (validation humans flagged "
          f"{metrics['modern_ai']['validation_humans_flagged']:.2%}); classic threshold {classic_t}")
    for group, m in by_group.items():
        verb = "wrongly flagged" if m["kind"] == "genuine" else "caught"
        print(f"  {group:16s} n={m['n']:<4} {verb:16s} {m['flagged_combined']:6.1%}  "
              f"(classic {m['flagged_by_classic']:.1%}, modern {m['flagged_by_modern']:.1%})")


if __name__ == "__main__":
    main()
