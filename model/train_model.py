#!/usr/bin/env python3
"""
MediTriageAI - Model Training Script
=====================================
Trains a RandomForestClassifier on the Disease-Symptom dataset from Kaggle.
Saves: model.pkl, features.pkl, disease_info.pkl

Dataset: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
Place the 4 CSVs in ../data/ before running this script.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = BASE_DIR

# ─── Load CSVs ────────────────────────────────────────────────────────────────
print("📂 Loading datasets...")

try:
    df_data        = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    df_desc        = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
    df_precaution  = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
    df_severity    = pd.read_csv(os.path.join(DATA_DIR, "symptom_severity.csv"))
except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease download the dataset from:")
    print("  https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
    print("And place the 4 CSV files in the 'data/' folder.\n")
    sys.exit(1)

print(f"✅ Loaded dataset.csv: {df_data.shape[0]} rows, {df_data.shape[1]} columns")

# ─── Clean & Prepare Dataset ──────────────────────────────────────────────────
# Strip whitespace from all string columns
df_data.columns = df_data.columns.str.strip()
df_data = df_data.map(lambda x: x.strip() if isinstance(x, str) else x)

# Collect all unique symptoms
symptom_cols = [c for c in df_data.columns if c.startswith("Symptom")]
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df_data[col].dropna().str.strip().str.lower().unique())
all_symptoms = sorted(all_symptoms)
print(f"✅ Found {len(all_symptoms)} unique symptoms")

# One-hot encode symptoms
def encode_symptoms(row, cols, symptoms):
    present = set()
    for col in cols:
        val = row.get(col, None)
        if pd.notna(val) and str(val).strip():
            present.add(str(val).strip().lower())
    return [1 if s in present else 0 for s in symptoms]

print("⚙️  Encoding symptom features...")
X = np.array(df_data.apply(encode_symptoms, axis=1, args=(symptom_cols, all_symptoms)).tolist())
y = df_data["Disease"].str.strip()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"✅ {len(le.classes_)} disease classes found")

# ─── Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ─── Train Model ──────────────────────────────────────────────────────────────
print("🤖 Training RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# ─── Build Disease Info Map ───────────────────────────────────────────────────
# Severity weights per disease (sum of symptom severities)
df_severity.columns = df_severity.columns.str.strip()
df_severity["Symptom"] = df_severity["Symptom"].str.strip().str.lower()
severity_map = dict(zip(df_severity["Symptom"], df_severity["weight"]))

# Clean descriptions
df_desc.columns = df_desc.columns.str.strip()
df_desc["Disease"]      = df_desc["Disease"].str.strip()
df_desc["Description"]  = df_desc["Description"].str.strip()
desc_map = dict(zip(df_desc["Disease"], df_desc["Description"]))

# Clean precautions
df_precaution.columns = df_precaution.columns.str.strip()
df_precaution["Disease"] = df_precaution["Disease"].str.strip()
precaution_cols = [c for c in df_precaution.columns if c.startswith("Precaution")]
precaution_map = {}
for _, row in df_precaution.iterrows():
    disease = row["Disease"]
    precs = [str(row[c]).strip() for c in precaution_cols if pd.notna(row[c]) and str(row[c]).strip()]
    precaution_map[disease] = precs

# ─── Risk Level Mapping ───────────────────────────────────────────────────────
# Emergency diseases (manually curated from medical knowledge)
EMERGENCY_DISEASES = {
    "Heart attack", "Paralysis (brain hemorrhage)", "Hypertension",
    "Diabetes", "Pneumonia", "Malaria", "Dengue", "Typhoid",
    "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
    "Jaundice", "Chronic cholestasis", "Alcoholic hepatitis",
    "Tuberculosis", "AIDS", "Cervical spondylosis", "Varicose veins"
}

def compute_disease_severity(disease_name, symptom_list):
    """Compute average severity weight for a disease's symptoms."""
    weights = [severity_map.get(s.strip().lower(), 3) for s in symptom_list]
    return np.mean(weights) if weights else 3

# Build disease severity scores from dataset
disease_severity_scores = {}
for disease in le.classes_:
    rows = df_data[df_data["Disease"] == disease]
    symptoms = []
    for _, row in rows.iterrows():
        for col in symptom_cols:
            val = row.get(col, None)
            if pd.notna(val) and str(val).strip():
                symptoms.append(str(val).strip().lower())
    disease_severity_scores[disease] = compute_disease_severity(disease, symptoms)

def get_risk_level(disease, score):
    if disease in EMERGENCY_DISEASES:
        return "Critical" if score >= 5 else "High"
    if score >= 6:
        return "High"
    elif score >= 4:
        return "Medium"
    else:
        return "Low"

def is_emergency(disease, score):
    if disease in EMERGENCY_DISEASES and score >= 5:
        return True
    if score >= 6.5:
        return True
    return False

disease_info = {}
for disease in le.classes_:
    score = disease_severity_scores.get(disease, 3)
    disease_info[disease] = {
        "description":  desc_map.get(disease, f"A medical condition: {disease}."),
        "precautions":  precaution_map.get(disease, ["Consult a doctor", "Rest well", "Stay hydrated"]),
        "severity_score": score,
        "risk_level":   get_risk_level(disease, score),
        "is_emergency": is_emergency(disease, score),
    }

# ─── Save Artifacts ───────────────────────────────────────────────────────────
print("💾 Saving model artifacts...")

with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(clf, f)

with open(os.path.join(MODEL_DIR, "features.pkl"), "wb") as f:
    pickle.dump({"symptoms": all_symptoms, "label_encoder": le}, f)

with open(os.path.join(MODEL_DIR, "disease_info.pkl"), "wb") as f:
    pickle.dump(disease_info, f)

print("\n🎉 Training complete!")
print(f"   ✅ model.pkl      → {os.path.join(MODEL_DIR, 'model.pkl')}")
print(f"   ✅ features.pkl   → {os.path.join(MODEL_DIR, 'features.pkl')}")
print(f"   ✅ disease_info.pkl → {os.path.join(MODEL_DIR, 'disease_info.pkl')}")
print(f"\n   🎯 Accuracy: {acc * 100:.2f}%")
print(f"   📊 Diseases: {len(le.classes_)}")
print(f"   💊 Symptoms: {len(all_symptoms)}")
