#!/usr/bin/env python3
"""
MediTriageAI - Model Utilities (Vercel version)
Identical to backend/model_utils.py but with MODEL_DIR pointing
to the project root's model/ folder.
"""

import os
import re
import pickle
import numpy as np

# On Vercel, __file__ is inside api/, so go up one level to reach model/
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

_model        = None
_features     = None
_disease_info = None

def _load_artifacts():
    global _model, _features, _disease_info
    if _model is not None:
        return

    model_path        = os.path.join(MODEL_DIR, "model.pkl")
    features_path     = os.path.join(MODEL_DIR, "features.pkl")
    disease_info_path = os.path.join(MODEL_DIR, "disease_info.pkl")

    missing = [p for p in [model_path, features_path, disease_info_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Model files not found: {missing}\n"
            "Please run: python model/train_model.py"
        )

    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    with open(features_path, "rb") as f:
        _features = pickle.load(f)
    with open(disease_info_path, "rb") as f:
        _disease_info = pickle.load(f)


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\-]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

def extract_symptoms(prompt: str, all_symptoms: list) -> list:
    norm = _normalise(prompt)
    found = []
    for symptom in all_symptoms:
        sym_norm = _normalise(symptom)
        pattern = r'\b' + re.escape(sym_norm) + r'\b'
        if re.search(pattern, norm):
            found.append(symptom)
    return found

def _encode_prompt(prompt: str, all_symptoms: list) -> np.ndarray:
    found = set(extract_symptoms(prompt, all_symptoms))
    vec = np.array([1 if s in found else 0 for s in all_symptoms])
    return vec, list(found)

def _confidence_label(prob: float) -> str:
    if prob >= 0.80: return "Very High"
    elif prob >= 0.60: return "High"
    elif prob >= 0.40: return "Moderate"
    elif prob >= 0.20: return "Low"
    return "Very Low"

def _generate_analysis(disease, found_symptoms, info, confidence):
    precautions = info.get("precautions", [])
    description = info.get("description", "")
    risk        = info.get("risk_level", "Unknown")
    emergency   = info.get("is_emergency", False)
    score       = info.get("severity_score", 3.0)

    sym_str = ", ".join(found_symptoms[:6]) if found_symptoms else "various symptoms"
    lines = []
    lines.append("## Analysis Report")
    lines.append("")
    lines.append(f"**Symptoms Detected:** {sym_str}")
    lines.append(f"**Most Likely Condition:** {disease}")
    lines.append(f"**Confidence Level:** {_confidence_label(confidence)} ({confidence*100:.1f}%)")
    lines.append(f"**Severity Score:** {score:.1f} / 7")
    lines.append(f"**Risk Classification:** {risk}")
    lines.append("")
    if description:
        lines.append("### About This Condition")
        lines.append(description)
        lines.append("")
    if emergency:
        lines.append("### ⚠️ EMERGENCY NOTICE")
        lines.append(
            "Based on the identified symptoms and condition, this may be a **medical emergency**. "
            "Please call emergency services (112 / 108) immediately or proceed to the nearest "
            "emergency room without delay. Do NOT drive yourself."
        )
        lines.append("")
    lines.append("### Recommended Precautions")
    for i, p in enumerate(precautions, 1):
        lines.append(f"{i}. {p.capitalize()}")
    lines.append("")
    lines.append("### General Advice")
    if risk in ("Critical", "High"):
        lines.append("Given the high severity, seek professional medical attention immediately.")
    elif risk == "Medium":
        lines.append("Schedule an appointment with a healthcare provider soon.")
    else:
        lines.append("Rest, hydrate well, and monitor your condition.")
    lines.append("")
    lines.append("---")
    lines.append(
        "*⚠️ Disclaimer: This is AI-generated and for informational purposes only. "
        "Always consult a qualified healthcare professional.*"
    )
    return "\n".join(lines)


def predict(prompt: str) -> dict:
    _load_artifacts()
    all_symptoms = _features["symptoms"]
    le           = _features["label_encoder"]

    if not prompt or not prompt.strip():
        return {"error": "Please enter your symptoms."}

    vec, found_symptoms = _encode_prompt(prompt, all_symptoms)
    proba   = _model.predict_proba([vec])[0]
    top_idx = np.argsort(proba)[::-1][:3]

    best_idx   = top_idx[0]
    disease    = le.inverse_transform([best_idx])[0]
    confidence = float(proba[best_idx])

    top_predictions = [
        {"disease": le.inverse_transform([i])[0], "probability": round(float(proba[i]), 4)}
        for i in top_idx
    ]

    info = _disease_info.get(disease, {
        "description":    "A medical condition.",
        "precautions":    ["Consult a doctor", "Rest well", "Stay hydrated"],
        "severity_score": 3.0,
        "risk_level":     "Medium",
        "is_emergency":   False,
    })

    analysis = _generate_analysis(disease, found_symptoms, info, confidence)

    return {
        "predicted_disease":  disease,
        "confidence":         round(confidence, 4),
        "confidence_label":   _confidence_label(confidence),
        "risk_level":         info["risk_level"],
        "is_emergency":       info["is_emergency"],
        "severity_score":     round(info["severity_score"], 2),
        "symptoms_detected":  found_symptoms,
        "precautions":        info["precautions"],
        "detailed_analysis":  analysis,
        "top_predictions":    top_predictions,
    }
