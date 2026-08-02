"""
FastAPI Server - LLM Prompt Analyzer
Exposes the prediction model as a REST API.

Endpoints:
    GET  /           → Health check
    POST /predict    → Analyze a prompt and return predictions
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow importing from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from inference.predictor import predict

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Prompt Analyzer API",
    description="Predict prompt quality and expected output tokens before sending to an LLM.",
    version="1.0.0"
)

# ── CORS — allows browser extensions and web apps to call this API ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all origins (browser extension, Streamlit, etc.)
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str
    input_tokens: int = None    # optional — estimated automatically if not provided

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain machine learning step by step with examples.",
                "input_tokens": 8
            }
        }


class PredictionResponse(BaseModel):
    prompt: str
    input_tokens: int
    predicted_output_tokens: int
    predicted_quality: float
    quality_label: str
    suggestions: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Check if the API is running."""
    return {
        "status": "running",
        "message": "LLM Prompt Analyzer API is live!",
        "endpoints": {
            "POST /predict": "Analyze a prompt",
            "GET /docs":     "Interactive API documentation"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def analyze_prompt(request: PromptRequest):
    """
    Analyze a prompt and return quality score + token prediction.

    - **prompt**: The prompt text you want to analyze
    - **input_tokens**: Optional. Auto-estimated if not provided.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if len(request.prompt) > 5000:
        raise HTTPException(status_code=400, detail="Prompt too long. Max 5000 characters.")

    try:
        result = predict(request.prompt, request.input_tokens)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        prompt=result["prompt"],
        input_tokens=result["input_tokens"],
        predicted_output_tokens=result["predicted_output_tokens"],
        predicted_quality=result["predicted_quality"],
        quality_label=result["quality_label"],
        suggestions=result["suggestions"]
    )


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
