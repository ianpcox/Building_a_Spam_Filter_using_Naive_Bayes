# Spam Filter API (Project Elevate)

This repository transforms a basic tutorial notebook on SMS spam classification into a **production-ready machine learning pipeline and microservice**. 

Using the UCI SMS Spam Collection, we elevated the project from manual probability calculations to a robust NLP pipeline comparing **Multinomial Naive Bayes** against **Logistic Regression** (with TF-IDF and bigrams). The project now includes comprehensive metrics, cross-validation, calibration analysis, and a deployable FastAPI microservice.

## Project Structure

* `run.py` — The core reproducible ML pipeline. Runs baselines, trains models, and generates metrics/visualizations.
* `api.py` — A FastAPI microservice that wraps the trained Logistic Regression model for real-time text classification.
* `docs/report.md` — A comprehensive paper-style report detailing methodology, model comparison, feature importance, and error analysis.
* `docs/assets/` — Generated charts and visualizations supporting the report.
* `Building a Spam Filter with Multinomial Naive Bayes.ipynb` — The original exploratory tutorial notebook.

## Key Findings

Logistic Regression (TF-IDF + Bigrams, balanced weights) proved superior for production over Naive Bayes, achieving:
* **Accuracy:** 98.65%
* **F1-Macro (5-fold CV):** 0.9712
* **ROC-AUC:** 0.9903
* Superior probability calibration, allowing for flexible API thresholding.

Read the full analysis in [docs/report.md](docs/report.md).

## How to Run

### 1. Run the ML Pipeline
Generates all metrics, trains the models, and outputs visualizations to `docs/assets/`.
```bash
pip install -r requirements.txt
python run.py
```

### 2. Start the API Server
Starts the FastAPI microservice on `localhost:8000`.
```bash
uvicorn api:app --reload
```

### 3. Test the API
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"}'
```
*Expected Response:*
```json
{
  "prediction": "spam",
  "spam_probability": 0.98,
  "latency_ms": 1.2
}
```
