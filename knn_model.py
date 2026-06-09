from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


RANDOM_STATE = 42
MALIGNANT_THRESHOLD = 0.5


def load_data(csv_path: Path):
    data = pd.read_csv(csv_path).dropna(axis=1, how="all")

    sample_ids = data["id"] if "id" in data.columns else None
    x = data.drop(columns=["diagnosis", "id"], errors="ignore")
    y = data["diagnosis"].map({"B": 0, "M": 1}).astype(int)

    return x, y, sample_ids


def create_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ]
    )


def tune_model(x_train, y_train) -> tuple[Pipeline, dict]:
    """Perform grid search to find best hyperparameters for KNN"""
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", KNeighborsClassifier(n_jobs=-1)),
        ]
    )

    param_grid = {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__metric": ["euclidean", "manhattan"],
        "model__weights": ["uniform", "distance"],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(x_train, y_train)

    best_params = {
        "n_neighbors": grid_search.best_params_["model__n_neighbors"],
        "metric": grid_search.best_params_["model__metric"],
        "weights": grid_search.best_params_["model__weights"],
    }

    return grid_search.best_estimator_, best_params


def save_predictions(sample_ids, test_index, y_test, y_pred, y_probability, output_dir: Path) -> None:
    predictions = pd.DataFrame(
        {
            "actual_label": y_test.map({0: "B", 1: "M"}).to_numpy(),
            "predicted_label": pd.Series(y_pred).map({0: "B", 1: "M"}).to_numpy(),
            "malignant_probability": y_probability,
        },
        index=test_index,
    )

    if sample_ids is not None:
        predictions.insert(0, "id", sample_ids.loc[test_index].to_numpy())

    predictions.to_csv(output_dir / "knn_test_predictions.csv", index=False)


def save_model_files(model, accuracy: float, auc_score: float, best_params: dict, output_dir: Path) -> None:
    joblib.dump(model, output_dir / "best_knn_model.joblib")

    metadata = {
        "method": "K-Nearest Neighbors",
        "best_variant": "RobustScaler + KNN",
        "best_params": best_params,
        "probability_threshold": MALIGNANT_THRESHOLD,
        "test_accuracy": accuracy,
        "test_roc_auc": auc_score,
    }
    (output_dir / "knn_model_metadata.json").write_text(json.dumps(metadata, indent=2))

    pd.DataFrame(
        [
            {
                "model": metadata["best_variant"],
                "test_accuracy": accuracy,
                "test_roc_auc": auc_score,
                "best_params": str(metadata["best_params"]),
            }
        ]
    ).to_csv(output_dir / "knn_model_comparison.csv", index=False)


def plot_class_distribution(y, output_dir: Path) -> None:
    labels = y.map({0: "Benign", 1: "Malignant"})

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=labels, hue=labels, palette=["#3a7d5f", "#b23a48"], legend=False)
    ax.set_title("Diagnosis Class Distribution")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Number of Samples")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(output_dir / "knn_class_distribution.png", dpi=160)
    plt.close()


def plot_correlation_heatmap(x, y, output_dir: Path) -> None:
    data = x.copy()
    data["diagnosis"] = y

    top_features = (
        data.corr(numeric_only=True)["diagnosis"]
        .abs()
        .sort_values(ascending=False)
        .drop("diagnosis")
        .head(15)
        .index
    )

    plt.figure(figsize=(12, 9))
    sns.heatmap(data[top_features].corr(numeric_only=True), cmap="vlag", center=0, square=True)
    plt.title("Correlation Heatmap of Top Features")
    plt.tight_layout()
    plt.savefig(output_dir / "knn_correlation_heatmap.png", dpi=160)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, output_dir: Path) -> None:
    matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Malignant"])

    _, ax = plt.subplots(figsize=(6, 5))
    display.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix - KNN")
    plt.tight_layout()
    plt.savefig(output_dir / "knn_confusion_matrix.png", dpi=160)
    plt.close()


def plot_roc_curve(y_test, y_probability, output_dir: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_probability)
    auc_score = roc_auc_score(y_test, y_probability)

    plt.figure(figsize=(7, 6))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.title("ROC Curve - KNN")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "knn_roc_curve.png", dpi=160)
    plt.close()


def plot_feature_importance(model, x_test, y_test, output_dir: Path) -> None:
    importance = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="accuracy",
        n_jobs=-1,
    )

    importance_data = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance": importance.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_data, x="importance", y="feature", hue="feature", palette="mako", legend=False)
    plt.title("Top Feature Importance - KNN")
    plt.xlabel("Mean Accuracy Decrease")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "knn_feature_importance.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a KNN breast cancer model.")
    parser.add_argument("--data", type=Path, default=Path("data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output.mkdir(parents=True, exist_ok=True)

    x, y, sample_ids = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    if args.tune:
        print("Performing hyperparameter tuning...")
        model, best_params = tune_model(x_train, y_train)
        print(f"Best parameters found: {best_params}")
    else:
        model = create_model()
        best_params = {
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform",
        }

    model.fit(x_train, y_train)

    # For KNN, get probabilities using predict_proba
    y_probability = model.predict_proba(x_test)[:, 1]
    y_pred = (y_probability >= MALIGNANT_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_probability)

    print("Model: RobustScaler + K-Nearest Neighbors")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc_score:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    save_model_files(model, accuracy, auc_score, best_params, args.output)
    save_predictions(sample_ids, x_test.index, y_test, y_pred, y_probability, args.output)

    plot_class_distribution(y, args.output)
    plot_correlation_heatmap(x, y, args.output)
    plot_confusion_matrix(y_test, y_pred, args.output)
    plot_roc_curve(y_test, y_probability, args.output)
    plot_feature_importance(model, x_test, y_test, args.output)

    print(f"Saved results in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
