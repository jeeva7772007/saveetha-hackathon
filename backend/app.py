#!/usr/bin/env python3
"""
MediTriageAI - Flask Backend
==============================
REST API that exposes the ML model for patient symptom analysis.

Endpoints:
  GET  /health           → health check
  POST /analyze          → analyze patient symptoms
  GET  /diseases         → list all known diseases
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_utils

app = Flask(__name__)
CORS(app, origins="*")

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "MediTriageAI",
        "version": "1.0.0"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze patient symptoms from free-text prompt.

    Request body (JSON):
        { "prompt": "I have chest pain and shortness of breath..." }

    Response (JSON):
        {
            "predicted_disease": "Heart attack",
            "confidence": 0.87,
            "confidence_label": "Very High",
            "risk_level": "Critical",
            "is_emergency": true,
            "severity_score": 6.5,
            "symptoms_detected": ["chest pain", "breathlessness"],
            "precautions": ["Call ambulance immediately", ...],
            "detailed_analysis": "## Analysis Report\n...",
            "top_predictions": [
                {"disease": "Heart attack", "probability": 0.87},
                ...
            ]
        }
    """
    data = request.get_json(silent=True)
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request body."}), 400

    prompt = str(data["prompt"]).strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    try:
        result = model_utils.predict(prompt)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result), 200
    except FileNotFoundError as e:
        return jsonify({
            "error": "Model not found. Please train the model first.",
            "details": str(e)
        }), 503
    except Exception as e:
        return jsonify({
            "error": "An internal error occurred during analysis.",
            "details": str(e)
        }), 500


@app.route("/diseases", methods=["GET"])
def list_diseases():
    """Return all diseases the model knows about."""
    try:
        import pickle
        base = os.path.join(os.path.dirname(__file__), "..", "model")
        with open(os.path.join(base, "disease_info.pkl"), "rb") as f:
            disease_info = pickle.load(f)
        diseases = [
            {
                "name": name,
                "risk_level": info["risk_level"],
                "is_emergency": info["is_emergency"]
            }
            for name, info in sorted(disease_info.items())
        ]
        return jsonify({"diseases": diseases, "count": len(diseases)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting MediTriageAI backend...")
    print("   API: http://localhost:5001")
    print("   Health: http://localhost:5001/health")
    print("   Analyze: POST http://localhost:5001/analyze")
    app.run(host="0.0.0.0", port=5001, debug=True)
