"""
Token Prediction Model Training - LLM Prompt Analyzer
Trains a regression model to predict output token count from prompt features.

Target  : output_tokens
Problem : Regression
Models  : Random Forest, XGBoost (best one is saved)
Output  : models/token_model.pkl
"""

import os
import sys
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
MODEL_OUTPUT   = os.path.join(MODELS_DIR, "token_model.pkl")

# Features used for training (same as extractor output)
FEATURE_COLS = [
    "prompt_length", "word_count", "sentence_count",
    "avg_word_length", "avg_sentence_length",
    "question_count", "exclamation_count",
    "code_block_count", "bullet_point_count",
    "has_role_instruction", "has_context", "has_constraints",
    "has_examples", "has_output_format", "has_step_by_step",
    "uppercase_ratio", "unique_word_ratio", "input_tokens"
]
TARGET_COL = "output_tokens"


def load_data():
    """Load processed features CSV and return X, y."""
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"features.csv not found at {FEATURES_FILE}\nRun: python data_collection/merge_datasets.py first")

    df = pd.read_csv(FEATURES_FILE)

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0]  # remove zero token rows

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    print(f"[INFO] Dataset size: {len(df)} rows")
    print(f"[INFO] Target range: min={y.min()}, max={y.max()}, mean={y.mean():.1f}")
    return X, y


def evaluate_model(name, model, X_test, y_test):
    """Print evaluation metrics for a model."""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n  [{name}]")
    print(f"    MAE  (avg error in tokens) : {mae:.1f} tokens")
    print(f"    RMSE                       : {rmse:.1f} tokens")
    print(f"    R²   (accuracy, 1.0=best)  : {r2:.3f}")
    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def plot_feature_importance(model, feature_names, title):
    """Plot and save feature importance chart."""
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(fi_df["feature"], fi_df["importance"], color="steelblue")
    plt.title(f"Feature Importance — {title}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "token_feature_importance.png")
    plt.savefig(plot_path)
    print(f"[INFO] Feature importance chart saved: {plot_path}")
    plt.close()


def plot_predictions(y_test, y_pred, title):
    """Plot actual vs predicted tokens."""
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="white", linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
    plt.xlabel("Actual Output Tokens")
    plt.ylabel("Predicted Output Tokens")
    plt.title(f"Actual vs Predicted — {title}")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "token_predictions.png")
    plt.savefig(plot_path)
    print(f"[INFO] Prediction chart saved: {plot_path}")
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
    print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ── Train models ──────────────────────────────────────────────────────────
    print("\n--- Step 2: Training models ---")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
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
    print(f"[INFO] Best model: {best['name']} (MAE: {best['mae']:.1f} tokens)")

    # ── Cross validation ──────────────────────────────────────────────────────
    print("\n--- Step 4: Cross Validation (5-fold) ---")
    cv_scores = cross_val_score(best["model"], X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()
    print(f"[INFO] Cross-val MAE: {cv_mae:.1f} tokens (±{cv_scores.std():.1f})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n--- Step 5: Generating charts ---")
    plot_feature_importance(best["model"], FEATURE_COLS, best["name"])
    plot_predictions(y_test, best["y_pred"], best["name"])

    # ── Save model ────────────────────────────────────────────────────────────
    print("\n--- Step 6: Saving model ---")
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(best["model"], f)
    print(f"[INFO] Model saved: {MODEL_OUTPUT}")

    print("\n=== Token Model Training Complete ===")
    print(f"  Best Model : {best['name']}")
    print(f"  MAE        : {best['mae']:.1f} tokens")
    print(f"  R²         : {best['r2']:.3f}")
    print(f"  Saved to   : {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
