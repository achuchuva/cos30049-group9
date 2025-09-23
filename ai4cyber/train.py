"""Train multiple models for spam detection.
Saves trained classification & clustering models into models/ directory.
"""
from __future__ import annotations
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, f1_score

from data_processing import load_spam_dataset, preprocess, save_artifacts
from models import get_classification_models, get_clustering_model

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train(csv_path: str = "data/emails.csv"):
    df = load_spam_dataset(csv_path)
    processed = preprocess(df)
    save_artifacts(processed)

    X_train, y_train = processed.X_train, processed.y_train
    class_models = get_classification_models()

    metrics_report = {}
    for name, model in class_models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        preds = model.predict(X_train)
        metrics_report[name] = {
            "train_accuracy": float(accuracy_score(y_train, preds)),
            "train_f1": float(f1_score(y_train, preds)),
        }

    # clustering (unsupervised) just to explore separation
    cluster_model = get_clustering_model()
    cluster_model.fit(X_train)
    joblib.dump(cluster_model, MODELS_DIR / "kmeans_pca.joblib")

    print("Training complete. Models saved to ./models")
    print("Training metrics (on train set):")
    for m, vals in metrics_report.items():
        print(m, vals)


if __name__ == "__main__":
    train()
