// popup.js — LLM Prompt Analyzer Extension

const API_URL = "http://localhost:8000/predict";

const analyzeBtn     = document.getElementById("analyzeBtn");
const promptInput    = document.getElementById("promptInput");
const resultDiv      = document.getElementById("result");
const statusDiv      = document.getElementById("status");
const errorDiv       = document.getElementById("error");

// ── On load: try to grab text from the active LLM tab ────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        func: grabPromptFromPage,
      },
      (results) => {
        if (results && results[0] && results[0].result) {
          promptInput.value = results[0].result;
        }
      }
    );
  });
});

// ── Grab prompt text from the LLM page ───────────────────────────────────────
function grabPromptFromPage() {
  // Try common LLM input selectors
  const selectors = [
    // ChatGPT
    "#prompt-textarea",
    // Claude
    ".ProseMirror",
    // Gemini
    "rich-textarea .ql-editor",
    // Generic contenteditable
    "[contenteditable='true']",
    // Generic textarea
    "textarea",
  ];

  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el && el.innerText && el.innerText.trim().length > 0) {
      return el.innerText.trim();
    }
    if (el && el.value && el.value.trim().length > 0) {
      return el.value.trim();
    }
  }
  return null;
}

// ── Analyze button click ──────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();

  if (!prompt) {
    statusDiv.textContent = "Please enter a prompt.";
    return;
  }

  // Reset UI
  resultDiv.style.display = "none";
  errorDiv.style.display  = "none";
  statusDiv.textContent   = "Analyzing...";
  analyzeBtn.disabled     = true;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();
    displayResult(data);
    statusDiv.textContent = "";

  } catch (err) {
    errorDiv.style.display = "block";
    statusDiv.textContent  = "";
    console.error(err);
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ── Display results ───────────────────────────────────────────────────────────
function displayResult(data) {
  // Metrics
  document.getElementById("inputTokens").textContent  = data.input_tokens;
  document.getElementById("outputTokens").textContent = `~${data.predicted_output_tokens}`;
  document.getElementById("qualityScore").textContent = `${data.predicted_quality}/10`;

  // Quality bar
  const quality  = data.predicted_quality;
  const pct      = (quality / 10) * 100;
  const barFill  = document.getElementById("qualityBar");
  const barLabel = document.getElementById("qualityLabel");

  barFill.style.width = `${pct}%`;

  if (quality >= 7) {
    barFill.style.background = "#4ade80";   // green
    barLabel.style.color     = "#4ade80";
  } else if (quality >= 5) {
    barFill.style.background = "#fbbf24";   // yellow
    barLabel.style.color     = "#fbbf24";
  } else {
    barFill.style.background = "#f87171";   // red
    barLabel.style.color     = "#f87171";
  }

  barLabel.textContent = data.quality_label;

  // Suggestions
  const list = document.getElementById("suggestionsList");
  list.innerHTML = "";

  data.suggestions.forEach((s) => {
    const div = document.createElement("div");
    div.className = s.includes("looks well structured")
      ? "suggestion-item good"
      : "suggestion-item";
    div.textContent = s.includes("looks well structured") ? `✅ ${s}` : `→ ${s}`;
    list.appendChild(div);
  });

  resultDiv.style.display = "block";
}
