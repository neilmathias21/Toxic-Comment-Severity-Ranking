# Toxic Comment Severity Ranking Using Machine Learning

## Project Overview

This project develops a machine learning system for ranking online comments according to their toxicity severity using Natural Language Processing (NLP) techniques.

Unlike traditional text classification tasks that predict fixed toxicity labels, this project focuses on ranking comments from less toxic to more toxic. The objective is based on the Kaggle **Jigsaw Toxic Severity Rating** competition, where models are evaluated on their ability to correctly order pairs of comments by toxicity.

The project uses TF-IDF feature engineering and multiple machine learning models to learn toxicity severity patterns from pairwise comparisons.

---

## Objectives

* Build an end-to-end toxicity ranking system
* Apply NLP preprocessing techniques to raw text data
* Convert text into numerical features using TF-IDF
* Train and compare multiple machine learning models
* Evaluate models using Pairwise Ranking Accuracy
* Interpret model behavior using feature importance analysis
* Generate toxicity scores for unseen comments

---

## Dataset

Dataset Source:

**Kaggle - Jigsaw Toxic Severity Rating**

Files Used:

### comments_to_score.csv

Contains comments that require toxicity scores.

### validation_data.csv

Contains pairs of comments labeled as:

* less_toxic
* more_toxic

These pairs are used to train and evaluate ranking models.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

### NLP Techniques

* Text Cleaning
* TF-IDF Vectorization
* N-Gram Feature Extraction

### Machine Learning Models

* Logistic Regression
* Linear SVC
* SGD Classifier
* Passive Aggressive Classifier
* Ridge Classifier

---

## Methodology

### 1. Data Loading

* Load Kaggle datasets
* Verify data quality
* Inspect structure and columns

### 2. Exploratory Data Analysis

* Comment length analysis
* Word count analysis
* Distribution visualization
* Outlier detection

### 3. Text Preprocessing

* Lowercase conversion
* URL removal
* HTML tag removal
* Special character removal
* Whitespace normalization

### 4. Feature Engineering

TF-IDF Vectorization:

* Maximum Features: 50,000
* N-Grams: (1,3)
* Sublinear TF Scaling

### 5. Pairwise Ranking Dataset Creation

Validation pairs are transformed into a binary ranking dataset:

* Less Toxic → 0
* More Toxic → 1

### 6. Model Training

Five ranking models are trained and evaluated:

1. Logistic Regression
2. Linear SVC
3. SGD Classifier
4. Passive Aggressive Classifier
5. Ridge Classifier

### 7. Model Evaluation

Evaluation Metric:

**Pairwise Ranking Accuracy**

A ranking is considered correct when:

```text
Score(More Toxic) > Score(Less Toxic)
```

### 8. Model Interpretation

Feature coefficients are analyzed to identify:

* Most toxic words
* Least toxic words
* Important ranking signals

---

## Results

| Model                         | Ranking Accuracy |
| ----------------------------- | ---------------: |
| Linear SVC                    |           80.19% |
| Ridge Classifier              |           79.63% |
| Logistic Regression           |           78.21% |
| Passive Aggressive Classifier |           77.96% |
| SGD Classifier                |           75.73% |

### Best Model

**Linear SVC**

### Best Ranking Accuracy

**80.19%**

---

## Key Findings

* Linear models performed exceptionally well on sparse TF-IDF text features.
* Linear SVC achieved the highest ranking accuracy among all evaluated models.
* TF-IDF combined with pairwise ranking provided a strong baseline for toxicity severity prediction.
* Feature importance analysis revealed meaningful linguistic indicators of toxic behavior.

---

## Outputs Generated

### Submission File

```text
results/submission.csv
```

Contains toxicity scores for all comments in the scoring dataset.

### Toxic Word Analysis

```text
results/toxic_words.csv
results/non_toxic_words.csv
```

### Final Project Summary

```text
results/final_summary.csv
```

---

## Future Improvements

Potential enhancements include:

* Hyperparameter Optimization
* Cross Validation
* Feature Selection
* Ensemble Ranking Methods
* Word Embeddings (Word2Vec, FastText, GloVe)
* Transformer Models (BERT, RoBERTa, DeBERTa)

---

## Conclusion

This project successfully developed an end-to-end toxicity severity ranking system using NLP and Machine Learning techniques.

Among the evaluated models, Linear SVC achieved the highest ranking accuracy of 80.19%, demonstrating strong performance for high-dimensional sparse text data.

The project provides a robust baseline solution for toxicity severity ranking and showcases the complete machine learning workflow from data preparation to model interpretation and submission generation.
