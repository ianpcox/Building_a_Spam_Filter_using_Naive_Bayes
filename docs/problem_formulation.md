# Problem Formulation: SMS Spam Classification

## Research question

Can a multinomial Naive Bayes classifier trained on labeled SMS messages reliably distinguish spam from ham (non-spam) so that the system achieves high precision and recall and is suitable for deployment as a filter?

## Success criteria

- **Primary metric:** F1 score (binary, spam positive class) on a held-out test set. Target: F1 ≥ 0.90.
- **Secondary metrics:** Precision and recall for spam; ROC-AUC; accuracy; calibration where applicable.
- **Baseline:** At least one non-ML baseline (e.g. majority-class predictor or simple keyword rule) reported on the same metrics and split.

## Stakeholders and decisions

- **Product/engineering:** Decision to deploy an SMS/mail filter or API; choice of threshold (precision vs recall trade-off).
- **End users:** Fewer spam messages in inbox; acceptable false positive rate (ham marked as spam).
- **Research/portfolio:** Demonstrates hypothesis-driven text classification, proper train/val/test, baselines, and metrics.
