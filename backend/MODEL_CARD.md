# Model Card — IMDB Sentiment Classifier

## Summary

| | |
|---|---|
| **Task** | Binary sentiment classification (positive / negative) |
| **Architecture** | TF-IDF (uni- + bigrams, 100k features, sublinear TF) → Logistic Regression (C=4.0) |
| **Training data** | IMDB 50k movie reviews (Maas et al., 2011), 49,582 after de-duplication |
| **Split** | 80/20 stratified train/test; all reported metrics are on the unseen test split |
| **Artifact** | `artifacts/sentiment_pipeline.joblib` (~2 MB), reproducible via `python train.py` |

## Evaluation

Metrics on the 9,917-review held-out test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority-class baseline | 0.502 | — | — | — | — |
| Multinomial Naive Bayes | 0.8865 | 0.8878 | 0.8857 | 0.8867 | 0.9544 |
| **Logistic Regression (selected)** | **0.9132** | **0.9077** | **0.9206** | **0.9141** | **0.9714** |

5-fold cross-validation on the training split: **0.9109 ± 0.0013** accuracy —
consistent with the test score, so the model is not overfit to the split.
Hyperparameters were chosen by grid search (`C ∈ {0.25, 1, 4}`, 3-fold CV)
on the training split only; the test set was touched exactly once.

Full numbers, confusion matrices, and training metadata live in
[`artifacts/metrics.json`](artifacts/metrics.json), written by `train.py`.

## Design decisions

- **Why a linear model and not a transformer?** On this dataset a fine-tuned
  DistilBERT reaches ~93%, roughly 2 points above this model — at the cost of
  ~250 MB of weights, GPU-bound inference, and a black-box decision process.
  The linear pipeline serves predictions in ~1 ms on CPU and is *exactly*
  explainable: the logit is a sum of per-token contributions, which the API
  returns with every prediction. For a review-scoring service, that trade-off
  favours the linear model; the transformer is the right upgrade path if the
  accuracy gap ever matters more than latency and interpretability.
- **No stop-word removal.** Standard English stop-word lists contain *not*,
  *no*, and *never*. Removing them destroys negation. Instead, bigrams let the
  model learn negation directly — `"not good"` carries a strong negative
  weight (−2.1) even though `"good"` alone is positive (+0.8).
- **Sublinear TF + min_df=2.** Repeating a word ten times shouldn't make a
  review ten times more positive; dropping hapax features cuts noise and
  artifact size.
- **Vectorizer and classifier ship as one `Pipeline`.** Training and serving
  share a single artifact and a single preprocessing function
  (`sentiment/preprocess.py`), so there is no possibility of training/serving
  skew.

## Limitations & known failure modes

- **Long-range negation and sarcasm.** Bigrams only see adjacent words.
  *"This was not bad at all, actually quite fun"* is misclassified as negative
  (0.78): `bad` (−1.9) and `at all` (−1.3) outweigh `fun` (+1.4), and the
  rescuing signal `not bad` (+0.5) is too weak.
- **Domain shift.** Trained exclusively on movie reviews. Vocabulary weights
  (e.g. *plot*, *acting*, *director*) will not transfer cleanly to product
  reviews or tweets; re-train on in-domain data before reusing.
- **English only**, and binary only — there is no neutral class, so mixed or
  factual text is forced into one of two labels.
- **Probability calibration.** Logistic regression's probabilities are
  reasonably calibrated by construction, but were not explicitly calibrated
  (e.g. via isotonic regression) against a held-out set; treat confidence
  scores as a ranking signal, not exact frequencies.

## Intended use

Demonstration / portfolio system for classifying English movie reviews.
Not intended for moderation, hiring, or any decision affecting people.
