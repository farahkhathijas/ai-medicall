import os
import pickle
import numpy as np

# -----------------------------
# Load model and encoder safely relative to current file
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "triage_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# -----------------------------
# Risk prediction function
# -----------------------------
def predict_risk(patient):
    """
    patient: dict with keys
    Age, Gender, Systolic_BP, Diastolic_BP,
    Heart_Rate, Temperature, Symptoms
    """

    # ---- 1. Compute Mean Arterial Pressure (MAP) ----
    mean_bp = (patient["Systolic_BP"] + 2 * patient["Diastolic_BP"]) / 3

    # ---- 2. Prepare numerical features ----
    X_num = np.array([[
        patient["Age"],
        mean_bp,
        patient["Heart_Rate"],
        patient["Temperature"]
    ]])

    # ---- 3. Prepare categorical features ----
    X_cat = [[
        patient["Gender"],
        patient["Symptoms"]
    ]]

    X_cat_encoded = encoder.transform(X_cat)

    # ---- 4. Combine features ----
    X = np.hstack([X_num, X_cat_encoded])

    # ---- 5. Predict risk ----
    prediction = model.predict(X)[0]

    return prediction