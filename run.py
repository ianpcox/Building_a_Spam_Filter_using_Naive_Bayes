"""
Single entry point: SMS spam classification with baseline and Multinomial NB.
Reproducible pipeline for Project Elevate (Phases 1-4).
Usage: python run.py [--data PATH]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

RANDOM_STATE = 42
TEST_SIZE = 0.2
SPAM_POSITIVE_LABEL = "spam"


def load_data(path: str) -> pd.DataFrame:
    """Load SMSSpamCollection (tab-separated, Label, SMS)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["Label", "SMS"], encoding="utf-8")
    df["Label"] = df["Label"].str.strip().str.lower()
    return df


def preprocess(text: str) -> str:
    """Lowercase and remove non-word chars (space-separated tokens)."""
    return re.sub(r"\W", " ", str(text).lower()).strip()


def majority_baseline(y_true: np.ndarray, classes: list) -> np.ndarray:
    """Predict majority class (ham) for every sample."""
    majority = max(set(classes), key=classes.count)
    return np.array([majority] * len(y_true))


def keyword_baseline(X_text: pd.Series, spam_keywords: set) -> np.ndarray:
    """Rule-based: if any spam keyword in message, predict spam else ham."""
    pred = []
    for msg in X_text:
        tokens = set(preprocess(msg).split())
        pred.append(SPAM_POSITIVE_LABEL if tokens & spam_keywords else "ham")
    return np.array(pred)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, name: str) -> dict:
    """Compute metrics; y_true/y_pred use string labels ham/spam."""
    # For sklearn metrics we need binary: spam=1, ham=0
    bin_true = (np.array(y_true) == SPAM_POSITIVE_LABEL).astype(int)
    bin_pred = (np.array(y_pred) == SPAM_POSITIVE_LABEL).astype(int)

    metrics = {
        "accuracy": accuracy_score(bin_true, bin_pred),
        "precision": precision_score(bin_true, bin_pred, zero_division=0),
        "recall": recall_score(bin_true, bin_pred, zero_division=0),
        "f1": f1_score(bin_true, bin_pred, zero_division=0),
    }
    if y_prob is not None and len(y_prob.shape) >= 2:
        prob_spam = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.ravel()
        metrics["roc_auc"] = roc_auc_score(bin_true, prob_spam)
    else:
        metrics["roc_auc"] = roc_auc_score(bin_true, bin_pred)  # use pred as proxy if no proba

    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("  Confusion matrix (TN FP / FN TP):")
    cm = confusion_matrix(bin_true, bin_pred)
    print(cm)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SMS spam classification pipeline")
    parser.add_argument(
        "--data",
        default="SMSSpamCollection",
        help="Path to SMSSpamCollection (tab-separated)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write metrics JSON",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    np.random.seed(RANDOM_STATE)
    df = load_data(str(data_path))
    df["text_clean"] = df["SMS"].apply(preprocess)

    X = df["text_clean"]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline 1: majority class
    y_pred_majority = majority_baseline(y_test.values, list(y_train))
    evaluate(y_test.values, y_pred_majority, None, "Baseline: majority (ham)")

    # Baseline 2: keyword rule (simple set of spam-indicative words)
    spam_keywords = {
        "free", "win", "prize", "cash", "urgent", "click", "call", "text", "stop",
        "claim", "winner", "congratulations", "selected", "offer", "guaranteed",
    }
    y_pred_keyword = keyword_baseline(X_test, spam_keywords)
    evaluate(y_test.values, y_pred_keyword, None, "Baseline: keyword rule")

    # Main: Multinomial NB with CountVectorizer
    vectorizer = CountVectorizer(min_df=2, max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)
    y_pred_nb = model.predict(X_test_vec)
    y_prob_nb = model.predict_proba(X_test_vec)

    metrics_nb = evaluate(y_test.values, y_pred_nb, y_prob_nb, "Main: Multinomial Naive Bayes")

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({"MultinomialNB": metrics_nb}, f, indent=2)
        print(f"\nMetrics written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
