# AI-based Project for Cybersecurity (AI4Cyber)

## Project Description

This project involves developing a comprehensive machine-learning solution that integrates project management, design elements, and technical implementation to address real-world cybersecurity challenges.

The project consists of three phases: 
1. Creating a detailed project management plan.
2. Implementing a machine learning model.
3. Developing a dynamic website to showcase their results.

The Machine Learning Web Application aims to deliver an interactive platform for users to engage with machine learning models and visualise data insights. The primary goal is to demonstrate practical machine learning applications in real-world cybersecurity-related scenarios, enhancing user interaction and understanding of the underlying models.

## Spam Detection Pipeline

### Folder/File Structure
```
ai4cyber/
  data/raw_datasets/          # Folder containing all raw datasets
  data/spam_featured.csv      # Final, processed, feature engineered dataset
  dataset_processing/         # Folder for misc files that processed and added features to cleaned dataset
  data_processing.py          # Loading, cleaning, TF-IDF feature generation
  eda.py                      # Generates figures for exploratory analysis
  models.py                   # Model factory functions
  train.py                    # Trains and saves models & artifacts
  evaluate.py                 # Evaluates saved models on test set
  predict.py                  # Predicts on a single .txt file input
  main.py                     # CLI orchestrator (eda, train, evaluate)
  models/                     # Persisted trained models (.joblib)
  artifacts/                  # Vectorizer & dataset splits (.joblib)
  reports/data_figures/       # Various saved data figures and plots
  reports/evaluation_figures/ # Evaluation figures of the classification and clustering models
```

### Installation (uv)
This project uses `uv` to for package syncing.
Ensure you have `uv` installed, then run inside project root:
```
uv sync
```

### Commands
Preprocess data based on csv file
```
python main.py preprocess --data data/spam_featured.csv
```

Run EDA (generates plots under `reports/data_figures`):
```
python main.py eda --data data/spam_featured.csv
```

Train models (Logistic Regression, Naive Bayes, Random Forest, KMeans clustering):
```
python main.py train --data data/spam_featured.csv
```

Evaluate models on held-out test set (prints metrics, saves ROC & confusion matrices):
Argument determines how the trained model names are prefixed
```
python main.py evaluate --prefix spam
```

Predict on a single text file:
```
python predict.py sample_input.txt
```

Run the full pipeline: preprocess, eda, train, evaluate
```
python main.py all --data data/spam_featured.csv
```

If data isn't specified, the default `data/spam_featured.csv` will be used.
If prefix isn't specified, the default `spam` will be used.

### Implemented Machine Learning Methods
Classification models:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest

Clustering model:
- KMeans (after PCA dimensionality reduction).

### Evaluation Metrics
For classification we compute: Accuracy, Precision, Recall, R2 Score, ROC AUC (when probabilistic scores available), and Confusion Matrix.
For clustering we compute: Silhouette Score, and Scatter Plots for model predictions and true labels.

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

### Reproducibility
Fixed random_state=42 for model reproducibility where applicable. Train/test split stratified to preserve class distribution.

### License / Use
Academic assignment use only.
