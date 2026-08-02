"""
Inference Module - LLM Prompt Analyzer
Takes a new prompt and predicts:
  - Expected output token count
  - Prompt quality score
  - Suggestions to improve the prompt
"""

import os
import sys
import pickle

# Allow importing from sibling folders
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from feature_engineering.extractor import extract_features

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_MODEL_PATH  = os.path.join(BASE_DIR, "models", "token_model.pkl")
QUALITY_MODEL_PATH = os.path.join(BASE_DIR, "models", "quality_model.pkl")


def load_models():
    """Load both trained models from disk."""
    if not os.path.exists(TOKEN_MODEL_PATH):
        raise FileNotFoundError(f"Token model not found: {TOKEN_MODEL_PATH}\nRun: python training/train_token_model.py")
    if not os.path.exists(QUALITY_MODEL_PATH):
        raise FileNotFoundError(f"Quality model not found: {QUALITY_MODEL_PATH}\nRun: python training/train_quality_model.py")

    with open(TOKEN_MODEL_PATH, "rb") as f:
        token_model = pickle.load(f)
    with open(QUALITY_MODEL_PATH, "rb") as f:
        quality_model = pickle.load(f)

    return token_model, quality_model


def get_quality_label(score: float) -> str:
    """Convert numeric quality score to a label."""
    if score >= 8:
        return "Excellent ⭐⭐⭐"
    elif score >= 6.5:
        return "Good ⭐⭐"
    elif score >= 5:
        return "Average ⭐"
    else:
        return "Poor"


def get_suggestions(features: dict) -> list:
    """Generate improvement suggestions based on missing features."""
    suggestions = []

    if features["has_role_instruction"] == 0:
        suggestions.append("Add a role → e.g. 'You are an expert in...'")
    if features["has_context"] == 0:
        suggestions.append("Add context → give background information")
    if features["has_examples"] == 0:
        suggestions.append("Add examples → e.g. 'For example...'")
    if features["has_constraints"] == 0:
        suggestions.append("Add constraints → e.g. 'Keep it under 200 words'")
    if features["has_output_format"] == 0:
        suggestions.append("Specify output format → e.g. 'Give as a bullet list'")
    if features["has_step_by_step"] == 0:
        suggestions.append("Ask for step-by-step → e.g. 'Explain step by step'")
    if features["word_count"] < 5:
        suggestions.append("Make prompt longer → add more details")

    return suggestions if suggestions else ["Prompt looks well structured!"]


def predict(prompt: str, input_tokens: int = None) -> dict:
    """
    Predict output tokens and quality score for a given prompt.

    Args:
        prompt (str): The user's prompt text.
        input_tokens (int): Optional. If not provided, estimated from word count.

    Returns:
        dict: Prediction results including tokens, quality, label, suggestions.
    """
    # Estimate input tokens if not provided (rough estimate: words * 1.3)
    if input_tokens is None:
        input_tokens = int(len(prompt.split()) * 1.3)

    # Step 1: Extract features
    features = extract_features(prompt, input_tokens)

    # Step 2: Load models
    token_model, quality_model = load_models()

    # Step 3: Prepare feature vector (must match training order)
    feature_order = [
        "prompt_length", "word_count", "sentence_count",
        "avg_word_length", "avg_sentence_length",
        "question_count", "exclamation_count",
        "code_block_count", "bullet_point_count",
        "has_role_instruction", "has_context", "has_constraints",
        "has_examples", "has_output_format", "has_step_by_step",
        "uppercase_ratio", "unique_word_ratio", "input_tokens"
    ]
    feature_vector = [[features[col] for col in feature_order]]

    # Step 4: Predict
    predicted_tokens = int(token_model.predict(feature_vector)[0])
    predicted_quality = round(float(quality_model.predict(feature_vector)[0]), 1)

    # Clip quality to valid range
    predicted_quality = max(1.0, min(10.0, predicted_quality))

    # Step 5: Get suggestions
    suggestions = get_suggestions(features)
    quality_label = get_quality_label(predicted_quality)

    return {
        "prompt": prompt,
        "input_tokens": input_tokens,
        "predicted_output_tokens": predicted_tokens,
        "predicted_quality": predicted_quality,
        "quality_label": quality_label,
        "suggestions": suggestions,
        "features": features
    }


def display_result(result: dict, show_features: bool = False):
    """Pretty print the prediction result."""
    print("\n" + "=" * 55)
    print("          LLM PROMPT ANALYZER — PREDICTION")
    print("=" * 55)
    print(f"\n  Prompt         : {result['prompt'][:70]}{'...' if len(result['prompt']) > 70 else ''}")
    print(f"\n  Input Tokens   : {result['input_tokens']}")
    print(f"  Output Tokens  : ~{result['predicted_output_tokens']} tokens (predicted)")
    print(f"\n  Quality Score  : {result['predicted_quality']} / 10")
    print(f"  Quality Label  : {result['quality_label']}")
    print(f"\n  Suggestions:")
    for s in result["suggestions"]:
        print(f"    → {s}")

    if show_features:
        print(f"\n  Extracted Features:")
        for k, v in result["features"].items():
            print(f"    {k:<25} : {v}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("       LLM PROMPT ANALYZER — INTERACTIVE MODE")
    print("=" * 55)
    print("  Type your prompt and press Enter to analyze it.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 55)

    while True:
        print()
        prompt = input("  Enter your prompt: ").strip()

        if not prompt:
            print("  [!] Please enter a prompt.")
            continue

        if prompt.lower() in ("quit", "exit"):
            print("\n  Goodbye!\n")
            break

        result = predict(prompt)
        display_result(result)
