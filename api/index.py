"""
MediTriageAI — Vercel Python WSGI entrypoint
All /api/* routes are served from this single Flask app.
Vercel auto-discovers this file as the Flask entrypoint.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import pickle

# Make api/ importable (model_utils.py lives here)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_utils

app = Flask(__name__)
CORS(app, origins="*")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "MediTriageAI", "version": "1.0.0"})


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return "", 200
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
        return jsonify({"error": "Model not found.", "details": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "Internal error during analysis.", "details": str(e)}), 500


@app.route("/api/diseases", methods=["GET"])
def diseases():
    try:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
        with open(os.path.join(base, "disease_info.pkl"), "rb") as f:
            disease_info = pickle.load(f)
        diseases_list = [
            {"name": n, "risk_level": i["risk_level"], "is_emergency": i["is_emergency"]}
            for n, i in sorted(disease_info.items())
        ]
        return jsonify({"diseases": diseases_list, "count": len(diseases_list)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
