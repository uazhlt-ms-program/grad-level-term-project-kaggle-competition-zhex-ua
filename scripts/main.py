#!/usr/bin/env python3
"""TF-IDF (word + char_wb) + Logistic Regression baseline for the review classifier."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def build_pipeline(word_max_features: int, char_max_features: int) -> Pipeline:
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=word_max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        max_features=char_max_features,
        sublinear_tf=True,
        lowercase=True,
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline([("features", features), ("clf", clf)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--quick",
        type=int,
        default=None,
        help="Subsample training to ~N rows (stratified) for fast iteration.",
    )
    parser.add_argument("--no-cv", action="store_true", help="Skip CV; fit + predict only.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--word-max-features", type=int, default=200_000)
    parser.add_argument("--char-max-features", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"loading data from {args.data_dir}…")
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    print(f"  train: {train.shape}, test: {test.shape}")
    print("  label distribution:")
    print(train["LABEL"].value_counts().sort_index().to_string())

    if args.quick:
        per_class = max(1, args.quick // train["LABEL"].nunique())
        train = (
            train.groupby("LABEL", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_class), random_state=args.seed))
            .reset_index(drop=True)
        )
        print(f"--quick: subsampled training to {len(train)} rows (stratified)")

    print("cleaning text…")
    train["TEXT"] = train["TEXT"].fillna("").map(clean_text)
    test["TEXT"] = test["TEXT"].fillna("").map(clean_text)

    X_train = train["TEXT"].to_numpy()
    y_train = train["LABEL"].to_numpy()
    X_test = test["TEXT"].to_numpy()

    pipeline = build_pipeline(args.word_max_features, args.char_max_features)

    if not args.no_cv:
        print(f"\nrunning {args.cv_folds}-fold stratified CV (macro-F1)…")
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        t0 = time.time()
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="f1_macro", n_jobs=1, verbose=1,
        )
        print(
            f"CV macro-F1: {scores.mean():.4f} ± {scores.std():.4f}  "
            f"(folds: {np.round(scores, 4).tolist()})  "
            f"[{time.time() - t0:.1f}s]"
        )

    print("\nfitting on full training set…")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  fit done in {time.time() - t0:.1f}s")

    print("predicting test set…")
    preds = pipeline.predict(X_test)
    submission = pd.DataFrame({"ID": test["ID"], "LABEL": preds})
    submission.to_csv(args.output, index=False)
    print(f"wrote {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
