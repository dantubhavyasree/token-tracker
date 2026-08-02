"""
Dashboard - LLM Prompt Analyzer
Streamlit web app to analyze prompts and predict quality & token usage.
"""

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Allow importing from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from inference.predictor import predict

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Prompt Analyzer",
    page_icon="🔍",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 LLM Prompt Analyzer")
st.markdown("**Predict prompt quality and expected token usage — before sending to an LLM.**")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("✍️ Enter Your Prompt")
prompt = st.text_area(
    label="Type your prompt here:",
    placeholder="e.g. Explain machine learning step by step with examples...",
    height=150
)

analyze_btn = st.button("🚀 Analyze Prompt", use_container_width=True, type="primary")

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Analyzing your prompt..."):
            result = predict(prompt)

        st.divider()

        # ── Metrics Row ───────────────────────────────────────────────────────
        st.subheader("📊 Prediction Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="🔤 Input Tokens",
                value=result["input_tokens"]
            )
        with col2:
            st.metric(
                label="📈 Expected Output Tokens",
                value=f"~{result['predicted_output_tokens']}"
            )
        with col3:
            st.metric(
                label="⭐ Quality Score",
                value=f"{result['predicted_quality']} / 10"
            )

        # ── Quality Label ─────────────────────────────────────────────────────
        quality = result["predicted_quality"]
        if quality >= 8:
            st.success(f"Quality: {result['quality_label']}")
        elif quality >= 6.5:
            st.info(f"Quality: {result['quality_label']}")
        elif quality >= 5:
            st.warning(f"Quality: {result['quality_label']}")
        else:
            st.error(f"Quality: {result['quality_label']}")

        st.divider()

        # ── Quality Gauge Chart ───────────────────────────────────────────────
        st.subheader("📉 Quality Score Gauge")
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh(["Quality"], [10], color="#e0e0e0", height=0.5)
        color = "#2ecc71" if quality >= 7 else "#f39c12" if quality >= 5 else "#e74c3c"
        ax.barh(["Quality"], [quality], color=color, height=0.5)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Score (out of 10)")
        ax.axvline(x=quality, color="black", linewidth=1.5, linestyle="--")
        ax.text(quality + 0.1, 0, f"{quality}", va="center", fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

        # ── Suggestions ───────────────────────────────────────────────────────
        st.subheader("💡 Suggestions to Improve Your Prompt")
        for suggestion in result["suggestions"]:
            if "looks well structured" in suggestion:
                st.success(f"✅ {suggestion}")
            else:
                st.info(f"→ {suggestion}")

        st.divider()

        # ── Feature Breakdown ─────────────────────────────────────────────────
        with st.expander("🔬 View Extracted Features"):
            features = result["features"]

            # Structural features
            st.markdown("**Structural Checks**")
            checks = {
                "Role Instruction": features["has_role_instruction"],
                "Context Provided": features["has_context"],
                "Constraints Set": features["has_constraints"],
                "Examples Included": features["has_examples"],
                "Output Format Specified": features["has_output_format"],
                "Step-by-Step Requested": features["has_step_by_step"],
            }
            cols = st.columns(3)
            for i, (label, val) in enumerate(checks.items()):
                with cols[i % 3]:
                    if val == 1:
                        st.success(f"✅ {label}")
                    else:
                        st.error(f"❌ {label}")

            # Numeric features
            st.markdown("**Numeric Features**")
            numeric = {
                "Prompt Length (chars)": features["prompt_length"],
                "Word Count": features["word_count"],
                "Sentence Count": features["sentence_count"],
                "Avg Word Length": features["avg_word_length"],
                "Question Count": features["question_count"],
                "Unique Word Ratio": features["unique_word_ratio"],
            }
            st.dataframe(
                pd.DataFrame(numeric.items(), columns=["Feature", "Value"]),
                use_container_width=True,
                hide_index=True
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("LLM Prompt Analyzer — Built with Python, Scikit-learn, XGBoost & Streamlit")
