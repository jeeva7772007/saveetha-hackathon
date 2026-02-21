# MediTriageAI 🏥

**AI-powered Patient Symptom Risk Assessment System**

Enter your symptoms in plain English → get instant risk level, emergency flag, precautions, and detailed analysis.

---

## 📦 Dataset Setup

1. Download from Kaggle: [Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
2. Place these 4 files in the `data/` folder:
   - `dataset.csv`
   - `symptom_Description.csv`
   - `symptom_precaution.csv`
   - `symptom_severity.csv`

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2 — Train the model
```bash
python model/train_model.py
```
> ✅ Should print accuracy (~90%+) and save `model.pkl`, `features.pkl`, `disease_info.pkl`

### Step 3 — Start the Flask backend
```bash
python backend/app.py
```
> Backend runs at: http://localhost:5000

### Step 4 — Open the frontend
Open `frontend/index.html` in your browser.

> ⚡ Make sure the backend is running before using the frontend!

---

## 📡 API Reference

### `POST /analyze`
```json
Request:  { "prompt": "I have chest pain and shortness of breath" }

Response: {
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
    { "disease": "Heart attack", "probability": 0.87 }
  ]
}
```

### `GET /health` — Health check
### `GET /diseases` — List all 40+ known diseases

---

## 🗂️ Project Structure

```
saveetha_hackathon/
├── data/                      ← Kaggle CSVs go here
├── model/
│   └── train_model.py         ← Run this first!
├── backend/
│   ├── app.py                 ← Flask API
│   ├── model_utils.py         ← ML inference logic
│   └── requirements.txt
└── frontend/
    ├── index.html             ← Open this in browser
    ├── style.css
    └── app.js
```

---

## 🛠️ Tech Stack

| Layer    | Technology                  |
|----------|-----------------------------|
| Frontend | HTML + CSS + Vanilla JS     |
| Backend  | Python Flask + Flask-CORS   |
| ML Model | scikit-learn RandomForest   |
| Dataset  | Kaggle Disease-Symptom CSV  |

---

*⚠️ For educational/hackathon purposes only. Not a substitute for professional medical advice.*
