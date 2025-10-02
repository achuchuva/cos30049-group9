# AI-based Project for Cybersecurity (AI4Cyber)

## Project Description

This project involves developing a comprehensive machine-learning solution that integrates project management, design elements, and technical implementation to address real-world cybersecurity challenges.

The project consists of three phases: 
1. Creating a detailed project management plan.
2. Implementing a machine learning model.
3. Developing a dynamic website to showcase their results.

The Machine Learning Web Application aims to deliver an interactive platform for users to engage with machine learning models and visualize data insights. The primary goal is to demonstrate practical machine learning applications in real-world cybersecurity-related scenarios, enhancing user interaction and understanding of the underlying models.

## Spam Detection Pipeline

### Repository Structure
```
ai4cyber/
  data/                     # Folder containing raw datasets
  data/emails.csv           # Provided raw dataset (text, spam)
  data_processing.py        # Loading, cleaning, TF-IDF feature generation
  eda.py                    # Generates figures for exploratory analysis
  models.py                 # Model factory functions
  train.py                  # Trains and saves models & artifacts
  evaluate.py               # Evaluates saved models on test set
  main.py                   # CLI orchestrator (eda, train, evaluate)
  models/                   # Persisted trained models (.joblib)
  artifacts/                # Vectorizer & dataset splits
  reports/figures/          # Saved plots & evaluation charts
```

### Installation (uv)
Ensure you have `uv` installed, then run inside project root:
```
uv sync
```

### Commands
Preprocess data based on csv file
```
python main.py preprocess --data data/emails.csv
```

Run EDA (generates plots under `reports/data_figures`):
```
python main.py eda --data data/emails.csv
```

Train models (Logistic Regression, Naive Bayes, Linear SVM, Random Forest + KMeans clustering):
```
python main.py train --data data/emails.csv
```

Evaluate models on held-out test set (prints metrics, saves ROC & confusion matrices):
Argument determines how the trained model names are prefixed
```
python main.py evaluate --prefix spam
```

Run the full pipeline: preprocess, eda, train, evaluate
```
python main.py all --data data/emails.csv
```

If data isn't specified, the default `data/emails.csv` will be used.
If prefix isn't specified, the default `spam` will be used.

### Implemented Machine Learning Methods
Classification models:
1. Logistic Regression (baseline linear classifier with TF-IDF sparse features)
2. Multinomial Naive Bayes (probabilistic model well-suited to word counts)
3. Linear SVM (margin-based, robust with high-dimensional sparse data)
4. Random Forest (ensemble of decision trees for non-linear relationships)

Unsupervised (Exploratory) model:
- KMeans (after PCA dimensionality reduction) to inspect natural grouping of emails.

### Evaluation Metrics
For classification we compute: Accuracy, Precision, Recall, F1 Score, ROC AUC (when probabilistic scores available), and Confusion Matrix.
For clustering we compute: Silhouette Score (interpretive only; clustering labels are not aligned with spam/ham classes without post-hoc mapping).

### Artifacts Saved
| Artifact | Path | Description |
|----------|------|-------------|
| Vectorizer | `artifacts/spam_vectorizer.joblib` | Fitted TF-IDF vocabulary + IDF weights |
| Train Split | `artifacts/spam_train.joblib` | (X_train sparse matrix, y_train) |
| Test Split | `artifacts/spam_test.joblib` | (X_test sparse matrix, y_test) |
| Models | `models/*.joblib` | Trained classifier / clustering models |
| Figures | `reports/data_figures/*.png` `reports/evaluation_figures/*.png` | EDA and evaluation plots |

### Dependencies
Key Python libraries used:
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib (persistence of models & artifacts)

### Justification of Model Choices
- Logistic Regression & Linear SVM: Strong baselines for linearly separable high-dimensional sparse text features.
- Multinomial Naive Bayes: Often excels with word frequency data and provides probabilistic outputs for ROC.
- Random Forest: Introduces non-linear decision boundaries and feature interaction modeling.
- KMeans + PCA: Offers unsupervised perspective; helps verify whether spam vs ham separation emerges naturally.

### Reproducibility
Fixed random_state=42 for model reproducibility where applicable. Train/test split stratified to preserve class distribution.

### License / Use
Academic assignment use only.