# End-to-End NLP Pipeline for Customer Feedback Analysis

## Overview

Built an end-to-end NLP sentiment classification pipeline using the IMDb dataset, covering preprocessing, baseline benchmarking, transformer fine-tuning, evaluation, and inference.

This project compares a classical machine learning approach (TF-IDF + Logistic Regression) with a transformer-based model (DistilBERT) under controlled experimental conditions.

---

## Dataset

* Source: IMDb movie reviews dataset
* Total size: 25,000 labeled samples
* Training subset: 5,000 balanced samples (2,500 positive, 2,500 negative)

---

## Pipeline

1. Text preprocessing (cleaning, normalization)
2. Baseline model:

   * TF-IDF vectorization
   * Logistic Regression classifier
3. Transformer model:

   * DistilBERT fine-tuning using Hugging Face Transformers
4. Evaluation:

   * Accuracy
   * F1 Score
   * Confusion Matrix
   * Training & Validation Curves
5. Inference on unseen text inputs

---

## Results

### Baseline (TF-IDF + Logistic Regression)

* Accuracy: **88.37%**
* F1 Score: **0.8849**

### DistilBERT

* Accuracy: **85.12%**
* F1 Score: **0.8512**

---

## Key Observations

* The baseline model outperformed DistilBERT in this experiment, achieving higher accuracy and F1 score.
* DistilBERT showed consistent learning (training loss decreased significantly), but validation loss increased slightly, indicating mild overfitting.
* Balanced sampling improved prediction stability across both classes.
* Results highlight that classical models remain highly competitive for sentiment classification tasks, especially under limited training budgets.

### Insight 

Despite the expressive power of transformer models, the TF-IDF baseline achieved better performance in this constrained setup. This highlights the importance of selecting models based on data size, computational budget, and problem complexity rather than assuming deep learning will always outperform simpler approaches.

---

## Visualizations

* Baseline Confusion Matrix
* BERT Confusion Matrix
* BERT Training Loss Curve
* BERT Validation Loss Curve
* BERT Validation Accuracy Curve
* BERT Validation F1 Curve

(See `results/plots/` directory)

---

## Sample Inference

* "The product is amazing and works perfectly." → Positive
* "The experience was okay, nothing special." → Negative
* "Very disappointed with the service and quality." → Negative

---

## Project Structure

```
data/
models/
results/
  ├── baseline_metrics.json
  ├── bert_metrics.json
  ├── plots/
src/
```

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Hugging Face Transformers
* PyTorch
* Matplotlib

---

## Future Improvements

* Hyperparameter tuning for DistilBERT
* Early stopping based on validation loss
* Training on the full dataset instead of a subset
* Model calibration to reduce overconfidence

---

## Summary

This project demonstrates an end-to-end applied machine learning workflow, including model benchmarking, evaluation, and interpretation. It emphasizes practical trade-offs between classical ML models and transformer-based approaches, reinforcing that model effectiveness depends on context, not just model complexity.
