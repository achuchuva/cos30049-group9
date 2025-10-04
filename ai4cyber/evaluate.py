"""Evaluate saved models on held-out test data.
Produces metrics & confusion matrices; saves ROC curve for probabilistic models.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
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

# Function for evaluation (includes printing accuracy, precision, recall, R2 score, confusion matrix, and classification report)
def evaluate_model(y_test, y_pred, model_name):
    print(f"{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print('-'*50)

def evaluate(prefix: str = "spam"):
    processed = load_artifacts(prefix=prefix)
    X_test, y_test = processed.X_test, processed.y_test

    # iterate through expected model filenames
    for name in get_classification_models().keys():
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        preds = model.predict(X_test)

        # Use the new evaluation function to print detailed metrics
        evaluate_model(y_test, preds, name)

        # ROC AUC for probabilistic models (if applicable)
        probas = None
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_test)[:, 1]
            except Exception:
                probas = None

        # Compute Receiver Operating Characteristic (ROC) curve and AUC
        # ROC shows the trade-off between TPR and FPR at various threshold settings
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

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"cm_{name}.png", dpi=150)
        plt.close()

    # clustering evaluation
    cluster_path = MODELS_DIR / "kmeans_pca.joblib"
    if cluster_path.exists():
        cluster_model = joblib.load(cluster_path)
        
        # Ensure X_test is a dense array for PCA transformation and plotting
        X_test_dense = processed.X_test.toarray() if hasattr(processed.X_test, "toarray") else processed.X_test

        # Get cluster labels
        labels = cluster_model.predict(X_test_dense)

        # Project data to 2D using the PCA from the pipeline for visualization
        pca_transformer = cluster_model.named_steps['pca']
        X_test_pca = pca_transformer.transform(X_test_dense)

        # Predicted Clusters
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.legend(handles=scatter.legend_elements()[0], labels=set(labels), title="Clusters")
        plt.title('KMeans Cluster Separation (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(FIG_DIR / "kmeans_cluster_separation.png", dpi=150)
        plt.close()

        # Cluster of True Labels
        plt.figure(figsize=(8, 6))
        scatter_true = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', alpha=0.7)
        plt.legend(handles=scatter_true.legend_elements()[0], labels=['Not Spam', 'Spam'], title="True Labels")
        plt.title('True Label Distribution (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(FIG_DIR / "kmeans_true_labels.png", dpi=150)
        plt.close()

        # silhouette requires at least 2 labels present
        if len(set(labels)) > 1:
            sil = silhouette_score(X_test_dense, labels)
            print(f"\nKMeans PCA Silhouette Score: {sil:.3f}")


if __name__ == "__main__":
    evaluate()
