"""
Feature Engineering Module - LLM Prompt Analyzer
Extracts numerical features from raw prompt text.

Features extracted:
    - prompt_length         : Total character count
    - word_count            : Number of words
    - sentence_count        : Number of sentences
    - avg_word_length       : Average characters per word
    - question_count        : Number of question marks
    - exclamation_count     : Number of exclamation marks
    - code_block_count      : Number of code blocks (``` or `)
    - bullet_point_count    : Number of bullet points (- or *)
    - has_role_instruction  : 1 if prompt defines a role (e.g. "you are", "act as")
    - has_context           : 1 if prompt provides background/context
    - has_constraints       : 1 if prompt sets constraints (e.g. "only", "do not", "must")
    - has_examples          : 1 if prompt includes examples
    - has_output_format     : 1 if prompt requests specific format (e.g. "list", "table", "json")
    - has_step_by_step      : 1 if prompt asks for step-by-step explanation
    - uppercase_ratio       : Ratio of uppercase letters to total letters
    - unique_word_ratio     : Ratio of unique words to total words
    - avg_sentence_length   : Average words per sentence
    - input_tokens          : Input token count (from dataset)
"""

import re
import pandas as pd


def extract_features(prompt: str, input_tokens: int = 0) -> dict:
    """
    Extract numerical features from a single prompt string.

    Args:
        prompt (str): The raw prompt text.
        input_tokens (int): Input token count from the dataset (default 0).

    Returns:
        dict: A dictionary of feature name -> value.
    """
    prompt = str(prompt)
    prompt_lower = prompt.lower()

    # --- Basic length features ---
    prompt_length = len(prompt)
    words = prompt.split()
    word_count = len(words)

    # Sentence count: split on . ! ?
    sentences = re.split(r'[.!?]+', prompt)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)  # avoid division by zero

    # Average word length
    avg_word_length = (
        sum(len(w.strip(".,!?;:\"'")) for w in words) / word_count
        if word_count > 0 else 0
    )

    # Average sentence length (words per sentence)
    avg_sentence_length = word_count / sentence_count

    # --- Punctuation features ---
    question_count = prompt.count('?')
    exclamation_count = prompt.count('!')

    # --- Code block count ---
    # Counts occurrences of triple backticks or inline backticks
    code_block_count = len(re.findall(r'```', prompt)) + len(re.findall(r'(?<!`)`(?!`)', prompt))

    # --- Bullet point count ---
    bullet_point_count = len(re.findall(r'^\s*[-*•]\s', prompt, re.MULTILINE))

    # --- Structural / intent features ---
    has_role_instruction = int(bool(re.search(
        r'\b(you are|act as|pretend|imagine you|assume you|your role)\b',
        prompt_lower
    )))

    has_context = int(bool(re.search(
        r'\b(context|background|given that|assuming|based on|here is|following)\b',
        prompt_lower
    )))

    has_constraints = int(bool(re.search(
        r'\b(only|do not|don\'t|must|should not|avoid|without|limit|restrict|no more than)\b',
        prompt_lower
    )))

    has_examples = int(bool(re.search(
        r'\b(for example|e\.g\.|such as|like|for instance|example|sample)\b',
        prompt_lower
    )))

    has_output_format = int(bool(re.search(
        r'\b(list|table|json|xml|bullet|numbered|format|structure|outline|summarize|summary)\b',
        prompt_lower
    )))

    has_step_by_step = int(bool(re.search(
        r'\b(step by step|step-by-step|explain how|walk me through|detailed|in detail)\b',
        prompt_lower
    )))

    # --- Ratio features ---
    letters = [c for c in prompt if c.isalpha()]
    uppercase_ratio = (
        sum(1 for c in letters if c.isupper()) / len(letters)
        if letters else 0
    )

    unique_word_ratio = (
        len(set(w.lower().strip(".,!?;:\"'") for w in words)) / word_count
        if word_count > 0 else 0
    )

    return {
        "prompt_length": prompt_length,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 3),
        "avg_sentence_length": round(avg_sentence_length, 3),
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "code_block_count": code_block_count,
        "bullet_point_count": bullet_point_count,
        "has_role_instruction": has_role_instruction,
        "has_context": has_context,
        "has_constraints": has_constraints,
        "has_examples": has_examples,
        "has_output_format": has_output_format,
        "has_step_by_step": has_step_by_step,
        "uppercase_ratio": round(uppercase_ratio, 3),
        "unique_word_ratio": round(unique_word_ratio, 3),
        "input_tokens": int(input_tokens) if str(input_tokens).strip().lstrip('-').isdigit() else 0,
    }


def extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from all rows in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'Prompt' and 'Input Tokens' columns.

    Returns:
        pd.DataFrame: Feature matrix with one row per prompt.
    """
    records = []
    for _, row in df.iterrows():
        features = extract_features(
            prompt=row.get("Prompt", ""),
            input_tokens=row.get("Input Tokens", 0)
        )
        # Keep target columns alongside features (safe conversion)
        def safe_int(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        features["output_tokens"] = safe_int(row.get("Output Tokens"))
        features["quality"] = safe_int(row.get("Quality"))
        records.append(features)

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Quick test on a single prompt
    sample = "Explain machine learning step by step with examples. Do not use jargon."
    result = extract_features(sample, input_tokens=12)
    print("=== Single Prompt Feature Extraction ===")
    for key, val in result.items():
        print(f"  {key:<25} : {val}")
