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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
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
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    solver="liblinear",
                    max_iter=20000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


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

    predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def save_model_files(model, accuracy: float, auc_score: float, output_dir: Path) -> None:
    joblib.dump(model, output_dir / "best_breast_cancer_model.joblib")

    metadata = {
        "method": "Logistic Regression",
        "best_variant": "RobustScaler + LogisticRegression",
        "best_params": {"model__C": 0.1},
        "probability_threshold": MALIGNANT_THRESHOLD,
        "test_accuracy": accuracy,
        "test_roc_auc": auc_score,
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    pd.DataFrame(
        [
            {
                "model": metadata["best_variant"],
                "test_accuracy": accuracy,
                "test_roc_auc": auc_score,
                "best_params": metadata["best_params"],
            }
        ]
    ).to_csv(output_dir / "model_comparison.csv", index=False)


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
    plt.savefig(output_dir / "class_distribution.png", dpi=160)
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
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, output_dir: Path) -> None:
    matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Malignant"])

    _, ax = plt.subplots(figsize=(6, 5))
    display.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def plot_roc_curve(y_test, y_probability, output_dir: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_probability)
    auc_score = roc_auc_score(y_test, y_probability)

    plt.figure(figsize=(7, 6))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
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
    plt.title("Top Feature Importance")
    plt.xlabel("Mean Accuracy Decrease")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Logistic Regression breast cancer model.")
    parser.add_argument("--data", type=Path, default=Path("data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
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

    model = create_model()
    model.fit(x_train, y_train)

    y_probability = model.predict_proba(x_test)[:, 1]
    y_pred = (y_probability >= MALIGNANT_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_probability)

    print("Model: RobustScaler + LogisticRegression")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc_score:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    save_model_files(model, accuracy, auc_score, args.output)
    save_predictions(sample_ids, x_test.index, y_test, y_pred, y_probability, args.output)

    plot_class_distribution(y, args.output)
    plot_correlation_heatmap(x, y, args.output)
    plot_confusion_matrix(y_test, y_pred, args.output)
    plot_roc_curve(y_test, y_probability, args.output)
    plot_feature_importance(model, x_test, y_test, args.output)

    print(f"Saved results in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
