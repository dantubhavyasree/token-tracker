"""
Quality Prediction Model Training - LLM Prompt Analyzer
Trains a regression model to predict prompt quality score from prompt features.

Target  : quality (1–8 score)
Problem : Regression
Models  : Random Forest, XGBoost (best one is saved)
Output  : models/quality_model.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_FILE  = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
MODEL_OUTPUT   = os.path.join(MODELS_DIR, "quality_model.pkl")

# Features used for training
FEATURE_COLS = [
    "prompt_length", "word_count", "sentence_count",
    "avg_word_length", "avg_sentence_length",
    "question_count", "exclamation_count",
    "code_block_count", "bullet_point_count",
    "has_role_instruction", "has_context", "has_constraints",
    "has_examples", "has_output_format", "has_step_by_step",
    "uppercase_ratio", "unique_word_ratio", "input_tokens"
]
TARGET_COL = "quality"


def load_data():
    """Load processed features CSV and return X, y."""
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"features.csv not found.\nRun: python data_collection/merge_datasets.py first")

    df = pd.read_csv(FEATURES_FILE)
    df = df.dropna(subset=[TARGET_COL])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(float)

    print(f"[INFO] Dataset size: {len(df)} rows")
    print(f"[INFO] Quality range: min={y.min()}, max={y.max()}, mean={y.mean():.2f}")

    # Show quality distribution
    print("\n[INFO] Quality distribution:")
    dist = y.value_counts().sort_index()
    for score, count in dist.items():
        bar = "█" * count
        print(f"  Score {int(score)}: {bar} ({count})")

    return X, y


def evaluate_model(name, model, X_test, y_test):
    """Print evaluation metrics."""
    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(np.round(y_pred), 1, 8)  # round to nearest score

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Within-1 accuracy: predictions within ±1 of actual score
    within_1 = np.mean(np.abs(y_pred_clipped - y_test) <= 1) * 100

    print(f"\n  [{name}]")
    print(f"    MAE  (avg error in score)  : {mae:.2f} points")
    print(f"    RMSE                       : {rmse:.2f} points")
    print(f"    R²   (accuracy, 1.0=best)  : {r2:.3f}")
    print(f"    Within ±1 score accuracy   : {within_1:.1f}%")

    return {
        "name": name, "model": model,
        "mae": mae, "rmse": rmse, "r2": r2,
        "within_1": within_1, "y_pred": y_pred
    }


def plot_feature_importance(model, feature_names, title):
    """Plot and save feature importance."""
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(fi_df["feature"], fi_df["importance"], color="coral")
    plt.title(f"Feature Importance — {title}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "quality_feature_importance.png")
    plt.savefig(plot_path)
    print(f"[INFO] Feature importance chart saved: {plot_path}")
    plt.close()


def plot_predictions(y_test, y_pred, title):
    """Plot actual vs predicted quality scores."""
    y_pred_rounded = np.clip(np.round(y_pred), 1, 8)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred_rounded, alpha=0.4, color="coral", edgecolors="white", linewidth=0.5)
    plt.plot([1, 8], [1, 8], "b--", lw=2, label="Perfect prediction")
    plt.xlabel("Actual Quality Score")
    plt.ylabel("Predicted Quality Score")
    plt.title(f"Actual vs Predicted — {title}")
    plt.xticks(range(1, 9))
    plt.yticks(range(1, 9))
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "quality_predictions.png")
    plt.savefig(plot_path)
    print(f"[INFO] Prediction chart saved: {plot_path}")
    plt.close()


def plot_quality_distribution(y_test, y_pred, title):
    """Compare actual vs predicted quality distribution."""
    y_pred_rounded = np.clip(np.round(y_pred), 1, 8).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_test, bins=range(1, 10), color="steelblue", edgecolor="white", align="left")
    axes[0].set_title("Actual Quality Distribution")
    axes[0].set_xlabel("Quality Score")
    axes[0].set_ylabel("Count")

    axes[1].hist(y_pred_rounded, bins=range(1, 10), color="coral", edgecolor="white", align="left")
    axes[1].set_title("Predicted Quality Distribution")
    axes[1].set_xlabel("Quality Score")

    plt.suptitle(title)
    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "quality_distribution.png")
    plt.savefig(plot_path)
    print(f"[INFO] Distribution chart saved: {plot_path}")
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n--- Step 1: Loading data ---")
    X, y = load_data()

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ── Train models ──────────────────────────────────────────────────────────
    print("\n--- Step 2: Training models ---")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    }

    results = []
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    # ── Pick best model (lowest MAE) ──────────────────────────────────────────
    print("\n--- Step 3: Selecting best model ---")
    best = min(results, key=lambda r: r["mae"])
    print(f"[INFO] Best model: {best['name']} (MAE: {best['mae']:.2f} points)")

    # ── Cross validation ──────────────────────────────────────────────────────
    print("\n--- Step 4: Cross Validation (5-fold) ---")
    cv_scores = cross_val_score(best["model"], X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()
    print(f"[INFO] Cross-val MAE: {cv_mae:.2f} points (±{cv_scores.std():.2f})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n--- Step 5: Generating charts ---")
    plot_feature_importance(best["model"], FEATURE_COLS, best["name"])
    plot_predictions(y_test, best["y_pred"], best["name"])
    plot_quality_distribution(y_test, best["y_pred"], best["name"])

    # ── Save model ────────────────────────────────────────────────────────────
    print("\n--- Step 6: Saving model ---")
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(best["model"], f)
    print(f"[INFO] Model saved: {MODEL_OUTPUT}")

    print("\n=== Quality Model Training Complete ===")
    print(f"  Best Model         : {best['name']}")
    print(f"  MAE                : {best['mae']:.2f} points")
    print(f"  R²                 : {best['r2']:.3f}")
    print(f"  Within ±1 accuracy : {best['within_1']:.1f}%")
    print(f"  Saved to           : {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
