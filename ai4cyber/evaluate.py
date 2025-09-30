"""Evaluate saved models on held-out test data.
Produces metrics & confusion matrices; saves ROC curve for probabilistic models.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    silhouette_score,
    classification_report,
)

from data_processing import load_artifacts
from models import get_classification_models

FIG_DIR = Path("reports/evaluation_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")

# Function for evaluation
def evaluate_model(y_test, y_pred, model_name):
    print(f"{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print('-'*50)

def evaluate(prefix: str = "spam"):
    processed = load_artifacts(prefix=prefix)
    X_test, y_test = processed.X_test, processed.y_test

    # iterate through expected model filenames
    results = {}
    for name in get_classification_models().keys():
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        preds = model.predict(X_test)

        # Use the new evaluation function to print detailed metrics
        evaluate_model(y_test, preds, name)

        probas = None
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_test)[:, 1]
            except Exception:
                probas = None
        
        roc_auc = None
        if probas is not None:
            roc_auc = roc_auc_score(y_test, probas)
            fpr, tpr, _ = roc_curve(y_test, probas)
            plt.figure(figsize=(4, 4))
            plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"roc_{name}.png", dpi=150)
            plt.close()

        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"cm_{name}.png", dpi=150)
        plt.close()

        # Store a summary of results (optional, as details are printed)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
        }

    # clustering evaluation (silhouette on train just for separation) optional
    cluster_path = MODELS_DIR / "kmeans_pca.joblib"
    if cluster_path.exists():
        cluster_model = joblib.load(cluster_path)
        # silhouette requires at least 2 labels present
        labels = cluster_model.predict(processed.X_test)
        if len(set(labels)) > 1:
            sil = silhouette_score(processed.X_test, labels)
            results["kmeans_pca"] = {"silhouette": float(sil)}
            print(f"\nKMeans PCA Silhouette Score: {sil:.3f}")

    return results


if __name__ == "__main__":
    evaluate()
