/**
 * MediTriageAI — Frontend Logic
 * ================================
 * Handles user input, API calls, and dynamic result rendering.
 */

const API_BASE = "/api";  // Vercel serverless functions live under /api/

// ─── DOM References ────────────────────────────────────────────────────────
const symptomInput = document.getElementById("symptomInput");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnText = document.getElementById("btnText");

const loadingSection = document.getElementById("loadingSection");
const loadingStep = document.getElementById("loadingStep");
const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");

const resultsSection = document.getElementById("resultsSection");
const errorSection = document.getElementById("errorSection");
const errorTitle = document.getElementById("errorTitle");
const errorMsg = document.getElementById("errorMsg");

// Sample chips
const SAMPLES = {
  chip1: "I have severe chest pain and shortness of breath. My left arm is numb and I feel very dizzy for the past 2 hours.",
  chip2: "I have a very high fever, severe headache, stiff neck and I am sensitive to light. I also feel nauseous.",
  chip3: "I have a runny nose, mild cough, and a slight sore throat. I feel a little tired but nothing too serious.",
  chip4: "I have stomach pain, nausea and vomiting since yesterday. I also have a mild fever and diarrhoea.",
};

// ─── Char Counter ──────────────────────────────────────────────────────────
symptomInput.addEventListener("input", () => {
  charCount.textContent = symptomInput.value.length;
});

// ─── Sample Chips ──────────────────────────────────────────────────────────
Object.entries(SAMPLES).forEach(([id, text]) => {
  const btn = document.getElementById(id);
  if (btn) {
    btn.addEventListener("click", () => {
      symptomInput.value = text;
      charCount.textContent = text.length;
      symptomInput.focus();
    });
  }
});

// ─── Analyze Button ────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", () => {
  const prompt = symptomInput.value.trim();
  if (!prompt) {
    symptomInput.focus();
    symptomInput.style.borderColor = "#ef4444";
    setTimeout(() => { symptomInput.style.borderColor = ""; }, 2000);
    return;
  }
  runAnalysis(prompt);
});

symptomInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    analyzeBtn.click();
  }
});

// ─── Try Again / Retry ────────────────────────────────────────────────────
document.getElementById("tryAgainBtn").addEventListener("click", resetUI);
document.getElementById("retryBtn").addEventListener("click", resetUI);

function resetUI() {
  resultsSection.style.display = "none";
  errorSection.style.display = "none";
  loadingSection.style.display = "none";
  document.querySelector(".input-section").style.display = "block";
  symptomInput.value = "";
  charCount.textContent = "0";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ─── Main Analysis Flow ───────────────────────────────────────────────────
async function runAnalysis(prompt) {
  // Hide input, show loading
  document.querySelector(".input-section").style.display = "none";
  resultsSection.style.display = "none";
  errorSection.style.display = "none";
  loadingSection.style.display = "block";
  window.scrollTo({ top: 0, behavior: "smooth" });

  // Animate loading steps
  const stepDelay = await animateLoadingSteps();

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `Server error: ${response.status}`);
    }

    const data = await response.json();
    clearTimeout(stepDelay);
    await saveToHistory(prompt, data);  // persist to Supabase
    showResults(data);

  } catch (err) {
    clearTimeout(stepDelay);
    showError(err.message || "Could not connect to the MediTriageAI server.");
  }
}

function animateLoadingSteps() {
  step1.className = "step active";
  step2.className = "step";
  step3.className = "step";
  loadingStep.textContent = "Extracting symptoms from your description...";

  const t1 = setTimeout(() => {
    step1.className = "step done";
    step2.className = "step active";
    loadingStep.textContent = "Running ML classification model...";
  }, 900);

  const t2 = setTimeout(() => {
    step2.className = "step done";
    step3.className = "step active";
    loadingStep.textContent = "Generating detailed report...";
  }, 1800);

  return t2;
}

// ─── Render Results ──────────────────────────────────────────────────────
function showResults(data) {
  loadingSection.style.display = "none";
  resultsSection.style.display = "block";
  window.scrollTo({ top: 0, behavior: "smooth" });

  // ── Emergency Banner ──
  const banner = document.getElementById("emergencyBanner");
  if (data.is_emergency) {
    banner.style.display = "block";
  } else {
    banner.style.display = "none";
  }

  // ── Risk Card ──
  const risk = data.risk_level || "Unknown";
  const riskKey = risk.toLowerCase();
  document.getElementById("riskLevel").textContent = risk;
  document.getElementById("riskLevel").className = `summary-value risk-${riskKey}`;
  document.getElementById("riskIcon").textContent = riskEmoji(riskKey);

  const riskBar = document.getElementById("riskBar");
  riskBar.className = `risk-bar risk-bar-${riskKey}`;

  // ── Disease Card ──
  document.getElementById("diseaseValue").textContent = data.predicted_disease || "Unknown";
  document.getElementById("diseaseMeta").textContent = `Severity Score: ${data.severity_score || "—"} / 7`;
  document.getElementById("diseaseIcon").textContent = "🔬";

  // ── Confidence Card ──
  const conf = Math.round((data.confidence || 0) * 100);
  const confFill = document.getElementById("confidenceFill");
  document.getElementById("confidenceValue").textContent = `${data.confidence_label || "—"} (${conf}%)`;
  setTimeout(() => { confFill.style.width = `${conf}%`; }, 100);

  // ── Emergency Status Card ──
  document.getElementById("emergencyIcon").textContent = data.is_emergency ? "🚨" : "✅";
  document.getElementById("emergencyStatus").textContent = data.is_emergency ? "EMERGENCY" : "Non-Emergency";
  document.getElementById("emergencyStatus").className = `summary-value ${data.is_emergency ? "risk-critical" : "risk-low"}`;
  document.getElementById("emergencyMeta").textContent = data.is_emergency ? "Seek immediate help" : "Monitor symptoms";

  // ── Symptoms Detected ──
  const tagsDiv = document.getElementById("symptomsTags");
  tagsDiv.innerHTML = "";
  const syms = data.symptoms_detected || [];
  if (syms.length === 0) {
    tagsDiv.innerHTML = `<span class="no-sym">No specific symptoms matched in our database. Please try more medical terms.</span>`;
  } else {
    syms.forEach((sym, i) => {
      const tag = document.createElement("span");
      tag.className = "sym-tag";
      tag.style.animationDelay = `${i * 60}ms`;
      tag.textContent = sym.replace(/_/g, " ");
      tagsDiv.appendChild(tag);
    });
  }

  // ── Top Predictions ──
  const predList = document.getElementById("predictionsList");
  predList.innerHTML = "";
  (data.top_predictions || []).forEach((pred, i) => {
    const pct = Math.round(pred.probability * 100);
    predList.innerHTML += `
      <div class="pred-item">
        <div class="pred-rank rank-${i + 1}">${i + 1}</div>
        <div class="pred-name">${pred.disease}</div>
        <div class="pred-bar-wrap">
          <div class="pred-bar-bg">
            <div class="pred-bar-fill" style="width:0%" data-width="${pct}%"></div>
          </div>
        </div>
        <div class="pred-prob">${pct}%</div>
      </div>
    `;
  });
  // Animate prediction bars
  setTimeout(() => {
    predList.querySelectorAll(".pred-bar-fill[data-width]").forEach(el => {
      el.style.width = el.dataset.width;
    });
  }, 120);

  // ── Precautions ──
  const precList = document.getElementById("precautionsList");
  precList.innerHTML = "";
  (data.precautions || []).forEach((p, i) => {
    precList.innerHTML += `
      <div class="prec-item" style="animation-delay:${i * 80}ms">
        <div class="prec-num">${i + 1}</div>
        <div class="prec-text">${capitalise(p)}</div>
      </div>
    `;
  });

  // ── Detailed Analysis ──
  const analysisDiv = document.getElementById("analysisContent");
  analysisDiv.innerHTML = formatAnalysis(data.detailed_analysis || "");
}

// ─── Error Display ─────────────────────────────────────────────────────────
function showError(msg) {
  loadingSection.style.display = "none";
  errorSection.style.display = "block";
  document.querySelector(".input-section").style.display = "none";

  if (msg.includes("fetch") || msg.includes("connect") || msg.includes("Failed")) {
    errorTitle.textContent = "Cannot Connect to Server";
    errorMsg.textContent = "Make sure the Flask backend is running: python backend/app.py";
  } else if (msg.includes("Model") || msg.includes("train")) {
    errorTitle.textContent = "Model Not Found";
    errorMsg.textContent = "Please train the model first: python model/train_model.py";
  } else {
    errorTitle.textContent = "Analysis Failed";
    errorMsg.textContent = msg;
  }
}

// ─── Helpers ───────────────────────────────────────────────────────────────
function riskEmoji(risk) {
  const map = { low: "🟢", medium: "🟡", high: "🟠", critical: "🔴" };
  return map[risk] || "⚪";
}

function capitalise(str) {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatAnalysis(text) {
  if (!text) return "";
  return text
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^(\d+\.) (.+)$/gm, '<div style="margin:4px 0; padding-left:8px;"><strong>$1</strong> $2</div>')
    .replace(/^---$/gm, '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:12px 0;">')
    .replace(/\n/g, '<br>');
}

// ─── Save to Supabase ─────────────────────────────────────────────────────
async function saveToHistory(prompt, result) {
  try {
    const { data: { session } } = await _supabase.auth.getSession();
    if (!session) return; // not logged in, skip silently
    await _supabase.from('analyses').insert({
      user_id: session.user.id,
      prompt: prompt,
      disease: result.predicted_disease,
      risk_level: result.risk_level,
      confidence: result.confidence,
      is_emergency: result.is_emergency,
      full_result: result,
    });
  } catch (err) {
    console.warn('Could not save history:', err);
  }
}
