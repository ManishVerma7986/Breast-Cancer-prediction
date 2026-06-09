from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "outputs" / "best_breast_cancer_model.joblib"
METADATA_PATH = BASE_DIR / "outputs" / "model_metadata.json"
DEFAULT_METADATA = {
    "method": "Logistic Regression",
    "best_variant": "RobustScaler + LogisticRegression",
    "best_params": {"model__C": 0.1},
    "probability_threshold": 0.5,
    "test_accuracy": 0,
    "test_roc_auc": 0,
}

app = Flask(__name__)


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else DEFAULT_METADATA


def load_feature_data() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(DATA_PATH).dropna(axis=1, how="all")
    features = data.drop(columns=["diagnosis", "id"], errors="ignore")
    return data, list(features.columns)


def format_label(feature_name: str) -> str:
    label = feature_name.replace("_", " ")
    label = label.replace(" se", " SE")
    return label.title().replace(" Se", " SE")


def group_features(feature_names: list[str]) -> dict[str, list[str]]:
    groups = {
        "Mean Measurements": [],
        "Standard Error Measurements": [],
        "Worst Measurements": [],
    }
    for feature in feature_names:
        if feature.endswith("_mean"):
            groups["Mean Measurements"].append(feature)
        elif feature.endswith("_se"):
            groups["Standard Error Measurements"].append(feature)
        elif feature.endswith("_worst"):
            groups["Worst Measurements"].append(feature)
    return groups


def build_presets(data: pd.DataFrame, feature_names: list[str]) -> dict[str, dict[str, float]]:
    presets = {
        "overall": data[feature_names].median(numeric_only=True).to_dict(),
        "benign": data.loc[data["diagnosis"] == "B", feature_names].median(numeric_only=True).to_dict(),
        "malignant": data.loc[data["diagnosis"] == "M", feature_names].median(numeric_only=True).to_dict(),
    }
    return {
        preset_name: {feature: round(float(value), 6) for feature, value in values.items()}
        for preset_name, values in presets.items()
    }


def build_feature_ranges(data: pd.DataFrame, feature_names: list[str]) -> dict[str, dict[str, float]]:
    ranges = {}
    for feature in feature_names:
        ranges[feature] = {
            "min": round(float(data[feature].min()), 6),
            "max": round(float(data[feature].max()), 6),
        }
    return ranges


DATA, FEATURE_NAMES = load_feature_data()
MODEL = joblib.load(MODEL_PATH)
METADATA = load_metadata()
THRESHOLD = float(METADATA.get("probability_threshold", 0.5))
FEATURE_GROUPS = group_features(FEATURE_NAMES)
PRESETS = build_presets(DATA, FEATURE_NAMES)
FEATURE_RANGES = build_feature_ranges(DATA, FEATURE_NAMES)
LABELS = {feature: format_label(feature) for feature in FEATURE_NAMES}


@app.route("/", methods=["GET", "POST"])
def index():
    selected_preset = request.args.get("preset", "overall")
    form_values = PRESETS.get(selected_preset, PRESETS["overall"]).copy()
    result = None
    errors: list[str] = []

    if request.method == "POST":
        selected_preset = "custom"
        form_values = {}
        for feature in FEATURE_NAMES:
            raw_value = request.form.get(feature, "").strip()
            try:
                form_values[feature] = float(raw_value)
            except ValueError:
                errors.append(f"{LABELS[feature]} must be a number.")

        if not errors:
            input_frame = pd.DataFrame([form_values], columns=FEATURE_NAMES)
            malignant_probability = float(MODEL.predict_proba(input_frame)[0, 1])
            predicted_class = int(malignant_probability >= THRESHOLD)
            result = {
                "diagnosis": "Malignant" if predicted_class else "Benign",
                "label": "M" if predicted_class else "B",
                "malignant_probability": malignant_probability,
                "benign_probability": 1 - malignant_probability,
            }

    return render_template(
        "index.html",
        errors=errors,
        feature_groups=FEATURE_GROUPS,
        feature_ranges=FEATURE_RANGES,
        labels=LABELS,
        metadata=METADATA,
        result=result,
        selected_preset=selected_preset,
        values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
