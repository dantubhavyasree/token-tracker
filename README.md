
# LLM Prompt Analyzer

> Predict the quality and expected token usage of a prompt **before** sending it to a Large Language Model.


## Overview

**LLM Prompt Analyzer** is a Machine Learning project that analyzes a user's prompt and predicts how an LLM is likely to respond **without actually sending the prompt to the model**.

The system provides:

- 📊 Expected input tokens
- 📈 Expected output tokens
- ⭐ Prompt quality score
- 💰 Estimated API cost *(Future Feature)*
- 💡 Prompt improvement suggestions

The goal is to help developers write better prompts, reduce API costs, and improve prompt engineering.

---

## Motivation

When interacting with LLMs like GPT, Claude, Gemini, or Kiro, users often don't know:

- How many output tokens the response will generate
- Whether the prompt is clear and complete
- How expensive the request may become
- Whether the prompt can be improved before submission

This project aims to solve these problems using Machine Learning.

---

## Problem Statement

Given a prompt, predict:

- Expected output token count
- Prompt quality score

before sending the prompt to an LLM.

---

# Project Workflow

```text
                        DATA COLLECTION
                              │
                              ▼
           Prompt + LLM Response Dataset Creation
                              │
                              ▼
                  Feature Engineering
                              │
                              ▼
                   Train ML Models
                              │
                              ▼
                    Save Trained Models
                              │
========================================================
                     USER USES LLM
========================================================
                              │
                              ▼
                  User types a prompt
                              │
                              ▼
              Feature Extraction Module
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
      Token Prediction Model        Quality Prediction Model
               │                             │
               └──────────────┬──────────────┘
                              ▼
                 Display Predictions
                              │
                              ▼
                  User edits prompt (optional)
                              │
                              ▼
                    Send Prompt to LLM
                              │
                              ▼
                     Receive Response
                              │
                              ▼
                  Store New Interaction
                              │
                              ▼
                    Retrain ML Models
````

---

# Dataset

Each interaction with the LLM becomes one training sample.

| Column         | Description             |
| -------------- | ----------------------- |
| Prompt         | User prompt             |
| Input Tokens   | Number of input tokens  |
| Output Tokens  | Number of output tokens |
| Prompt Quality | Quality score           |
| Model          | LLM used                |
| Response Time  | Time taken              |
| Timestamp      | Interaction time        |

Example:

| Prompt                   | Input Tokens | Output Tokens | Quality |
| ------------------------ | ------------ | ------------- | ------- |
| Explain AI               | 3            | 420           | 6       |
| Explain AI with examples | 7            | 730           | 9       |
| Write Python BFS         | 5            | 310           | 8       |

---

# Feature Engineering

The prompt is converted into numerical features.

Features include:

* Prompt length
* Number of words
* Number of sentences
* Input token count
* Role instruction
* Context availability
* Constraints
* Examples
* Output format requested
* Readability score
* Question count
* Code block count
* Average word length

These features are used to train the ML models.

---

# Machine Learning Models

## 1. Output Token Prediction

**Problem Type**

Regression

**Input**

Prompt Features

**Output**

Expected Output Tokens

Possible Algorithms

* Random Forest
* XGBoost
* LightGBM
* CatBoost

---

## 2. Prompt Quality Prediction

**Problem Type**

Regression / Classification

Output

* 8.5 / 10

or

* Poor
* Average
* Good
* Excellent

---

# Project Structure

```text
LLM-Prompt-Analyzer/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── data_collection/
│   ├── collect_data.py
│   ├── tokenizer.py
│
├── feature_engineering/
│   ├── extractor.py
│
├── training/
│   ├── train_token_model.py
│   ├── train_quality_model.py
│
├── models/
│   ├── token_model.pkl
│   ├── quality_model.pkl
│
├── inference/
│   ├── predictor.py
│
├── dashboard/
│   ├── app.py
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Streamlit / React
* FastAPI
* SQLite / PostgreSQL

---

# Pipeline

1. Collect prompts from LLM interactions.
2. Store prompt-response data.
3. Extract numerical features.
4. Train ML models.
5. Save trained models.
6. Analyze new prompts.
7. Predict expected output tokens.
8. Predict prompt quality.
9. Display suggestions before sending the prompt.

---

# Example Output

```text
Prompt Quality        : 8.8 / 10

Input Tokens          : 58

Expected Output Tokens: 742

Estimated Cost        : $0.014

Suggestions

✓ Add output format

✓ Add examples

✓ Specify constraints

✓ Mention target audience
```

---

# Future Enhancements

* Multi-model support
* Cost prediction
* Latency prediction
* Prompt optimization
* Prompt rewriting
* Explainable AI
* Browser extension
* VS Code extension
* Continuous learning
* Model comparison dashboard

---

# Current Progress

* [ ] Design dataset
* [ ] Collect data
* [ ] Build feature extractor
* [ ] Train token prediction model
* [ ] Train quality prediction model
* [ ] Build dashboard
* [ ] Integrate with LLM
* [ ] Continuous learning pipeline

---

# Vision

Build an intelligent assistant that evaluates prompts before they reach an LLM, helping developers write clearer, more efficient, and cost-effective prompts while predicting the expected response characteristics using Machine Learning.

```
```
