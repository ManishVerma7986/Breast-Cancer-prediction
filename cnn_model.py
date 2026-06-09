from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not installed. Install with: pip install tensorflow")


RANDOM_STATE = 42
MALIGNANT_THRESHOLD = 0.5


def load_data(csv_path: Path):
    data = pd.read_csv(csv_path).dropna(axis=1, how="all")

    sample_ids = data["id"] if "id" in data.columns else None
    x = data.drop(columns=["diagnosis", "id"], errors="ignore")
    y = data["diagnosis"].map({"B": 0, "M": 1}).astype(int)

    return x, y, sample_ids


def create_cnn_model(input_shape: int) -> keras.Sequential:
    """Create a CNN-inspired neural network for tabular data"""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_shape,)),
            # Reshape for 1D convolution
            layers.Reshape((input_shape, 1)),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC()],
    )

    return model


def train_cnn_model(x_train, y_train, x_val, y_val, epochs: int = 100) -> tuple[keras.Sequential, dict]:
    """Train the CNN model with early stopping"""
    model = create_cnn_model(x_train.shape[1])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    return model, history.history


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

    predictions.to_csv(output_dir / "cnn_test_predictions.csv", index=False)


def save_model_files(model, accuracy: float, auc_score: float, output_dir: Path) -> None:
    model.save(output_dir / "best_cnn_model.keras")

    metadata = {
        "method": "Convolutional Neural Network (1D CNN)",
        "architecture": "Conv1D + BatchNorm + Dropout + Dense layers",
        "probability_threshold": MALIGNANT_THRESHOLD,
        "test_accuracy": accuracy,
        "test_roc_auc": auc_score,
    }
    (output_dir / "cnn_model_metadata.json").write_text(json.dumps(metadata, indent=2))

    pd.DataFrame(
        [
            {
                "model": metadata["architecture"],
                "test_accuracy": accuracy,
                "test_roc_auc": auc_score,
            }
        ]
    ).to_csv(output_dir / "cnn_model_comparison.csv", index=False)


def plot_training_history(history: dict, output_dir: Path) -> None:
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history["val_loss"], label="Validation Loss", linewidth=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "cnn_training_history.png", dpi=160)
    plt.close()


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
    plt.savefig(output_dir / "cnn_class_distribution.png", dpi=160)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, output_dir: Path) -> None:
    matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Malignant"])

    _, ax = plt.subplots(figsize=(6, 5))
    display.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix - CNN")
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_confusion_matrix.png", dpi=160)
    plt.close()


def plot_roc_curve(y_test, y_probability, output_dir: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_probability)
    auc_score = roc_auc_score(y_test, y_probability)

    plt.figure(figsize=(7, 6))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.title("ROC Curve - CNN")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_roc_curve.png", dpi=160)
    plt.close()


def main() -> None:
    if not TENSORFLOW_AVAILABLE:
        print("ERROR: TensorFlow is required for the CNN model.")
        print("Install it with: pip install tensorflow")
        return

    parser = argparse.ArgumentParser(description="Train a CNN breast cancer model.")
    parser.add_argument("--data", type=Path, default=Path("data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output.mkdir(parents=True, exist_ok=True)

    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    x, y, sample_ids = load_data(args.data)

    # Normalize features
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)

    # Split data
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_scaled,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("Training CNN model...")
    model, history = train_cnn_model(x_train, y_train, x_val, y_val, epochs=args.epochs)

    y_probability = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_probability >= MALIGNANT_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_probability)

    print("\nModel: 1D Convolutional Neural Network")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc_score:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    save_model_files(model, accuracy, auc_score, args.output)
    save_predictions(sample_ids, range(len(y_test)), y_test, y_pred, y_probability, args.output)

    plot_training_history(history, args.output)
    plot_class_distribution(y, args.output)
    plot_confusion_matrix(y_test, y_pred, args.output)
    plot_roc_curve(y_test, y_probability, args.output)

    print(f"Saved results in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
