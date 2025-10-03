"""Model definitions for spam detection.
Provides multiple classifiers and a clustering approach for exploratory analysis.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


@dataclass
class ModelBundle:
    name: str
    model: Any


def get_classification_models() -> Dict[str, Any]:
    """Returns a dictionary of classification models with regularization parameters
    to mitigate overfitting.
    """
    return {
        "logreg": LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", random_state=42),
        "nb": MultinomialNB(alpha=0.5, fit_prior=False),
        "rf": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=30,
            min_samples_leaf=2,
            class_weight='balanced'
        ),
    }


def get_clustering_model(n_clusters: int = 2, pca_components: int = 50) -> Pipeline:
    return Pipeline(
        steps=[
            ("pca", PCA(n_components=pca_components, random_state=42)),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ]
    )
