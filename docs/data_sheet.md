# Data Sheet: SMS Spam Collection

## Dataset

- **Name:** SMS Spam Collection (UCI / Tiago A. Almeida and José María Gómez Hidalgo).
- **Source:** [UCI Machine Learning Repository – SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). Composition and papers: [dt.fee.unicamp.br](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/#composition).
- **License:** Publicly available for research and education; verify current UCI terms for redistribution.
- **Version / date:** Dataset as used in-project; original collection circa 2012 (see source page for exact dates).

## Schema and format

- **File:** `SMSSpamCollection` (tab-separated, no header).
- **Columns:** `Label` (ham/spam), `SMS` (raw message text).
- **Size:** 5,572 SMS messages.

## Splits

- **Train:** 80% (4,458 messages), **Test:** 20% (1,114 messages), with fixed random seed (e.g. 1 or 42) for reproducibility.
- Optional: reserve a validation subset from training for hyperparameter tuning (e.g. 80/10/10 train/val/test).

## Demographics and collection

- Messages are in English; collection methodology and demographics of senders/recipients are not fully documented.
- Labels are human-assigned (spam vs ham). Possible noise or subjectivity in edge cases.

## Known biases and limitations

- **Class imbalance:** ~87% ham, ~13% spam. Metrics should be reported per class and F1/ROC-AUC used alongside accuracy.
- **Temporal:** Data is from a specific period; spam language and tactics evolve; performance may degrade over time without retraining.
- **Language:** English-only; not representative of multilingual or code-mixed SMS.

## Optional dataset expansion (DATASET_EXPANSIONS.md)

- **SpamDam** (~76k SMS, 2018–2023, adversarial) or **UCI Spam Email** could be used as additional benchmarks or for robustness checks. Document in this data sheet if added; keep UCI SMS as primary for reproducibility.
