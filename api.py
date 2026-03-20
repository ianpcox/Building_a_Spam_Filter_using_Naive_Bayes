"""
Spam Filter API — FastAPI microservice
Project Elevate: Building_a_Spam_Filter_using_Naive_Bayes

Usage:
    uvicorn api:app --reload

Endpoints:
    GET  /health         — Health check
    GET  /model-info     — Model metadata and feature info
    POST /predict        — Classify a single message
    POST /predict-batch  — Classify a list of messages
"""

import re
import time
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = Path("SMSSpamCollection")
MODEL_PATH = Path("model.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Spam Filter API",
    description=(
        "A production-ready SMS spam classification microservice. "
        "Powered by Logistic Regression with TF-IDF and bigrams, trained on the "
        "UCI SMS Spam Collection (5,572 messages). Achieves 98.65% accuracy and "
        "0.9903 ROC-AUC. See /model-info for full performance details."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, example="URGENT! You have won a £1000 prize. Call now!")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Spam probability threshold (default 0.5)")


class PredictResponse(BaseModel):
    prediction: str = Field(..., example="spam")
    spam_probability: float = Field(..., example=0.98)
    ham_probability: float = Field(..., example=0.02)
    latency_ms: float = Field(..., example=1.4)
    threshold_used: float = Field(..., example=0.5)


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_latency_ms: float


# ── Model Loading / Training ──────────────────────────────────────────────────
def preprocess(text: str) -> str:
    return re.sub(r"\W", " ", str(text).lower()).strip()


def train_and_save_model():
    """Train the Logistic Regression pipeline and save it to disk."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at '{DATA_PATH}'. "
            "Please ensure 'SMSSpamCollection' is in the project root."
        )
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "sms"], encoding="utf-8")
    df["label"] = df["label"].str.strip().str.lower()
    df["text_clean"] = df["sms"].apply(preprocess)

    X_train, _, y_train, _ = train_test_split(
        df["text_clean"], df["label"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, max_features=15000, sublinear_tf=True, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(C=5.0, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return pipeline


def load_model():
    """Load model from disk, or train it if not found."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    print("No saved model found — training now...")
    return train_and_save_model()


# ── Startup ───────────────────────────────────────────────────────────────────
model: Pipeline = None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()
    print("Spam Filter API ready.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info", tags=["System"])
def model_info():
    return {
        "model": "Logistic Regression",
        "vectorizer": "TF-IDF (unigrams + bigrams, max_features=15000)",
        "training_data": "UCI SMS Spam Collection (5,572 messages)",
        "performance": {
            "accuracy": 0.9865,
            "precision_spam": 0.9653,
            "recall_spam": 0.9329,
            "f1_spam": 0.9488,
            "roc_auc": 0.9903,
            "cv_f1_macro_5fold": "0.9712 ± 0.0059",
        },
        "report": "See docs/report.md for full analysis.",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    clean_text = preprocess(request.text)
    proba = model.predict_proba([clean_text])[0]
    classes = list(model.classes_)
    spam_idx = classes.index("spam")
    ham_idx  = classes.index("ham")
    spam_prob = float(proba[spam_idx])
    ham_prob  = float(proba[ham_idx])
    prediction = "spam" if spam_prob >= request.threshold else "ham"
    latency = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        prediction=prediction,
        spam_probability=round(spam_prob, 4),
        ham_probability=round(ham_prob, 4),
        latency_ms=round(latency, 2),
        threshold_used=request.threshold,
    )


@app.post("/predict-batch", response_model=BatchPredictResponse, tags=["Classification"])
def predict_batch(request: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    clean_texts = [preprocess(t) for t in request.texts]
    probas = model.predict_proba(clean_texts)
    classes = list(model.classes_)
    spam_idx = classes.index("spam")
    ham_idx  = classes.index("ham")

    results = []
    for i, proba in enumerate(probas):
        spam_prob = float(proba[spam_idx])
        ham_prob  = float(proba[ham_idx])
        prediction = "spam" if spam_prob >= request.threshold else "ham"
        results.append(PredictResponse(
            prediction=prediction,
            spam_probability=round(spam_prob, 4),
            ham_probability=round(ham_prob, 4),
            latency_ms=0.0,
            threshold_used=request.threshold,
        ))

    total_latency = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(results=results, total_latency_ms=round(total_latency, 2))
