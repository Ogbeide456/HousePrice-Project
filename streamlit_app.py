

from pathlib import Path
import joblib
import os
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

if not os.path.exists(MODEL_PATH):
    # Option: call an internal function to train, or import train_model and call it
    # from train_model import train_and_save; train_and_save()
    st.warning("Model not found — training now. This can take a while...")
    # ...call training code here...
model = joblib.load(MODEL_PATH)
import json
import pandas as pd
from flask import Flask, render_template, request, flash

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "house_price_model.pkl"
NEIGH_PATH = ROOT / "model" / "neighborhood_categories.json"

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-random-key"

# Load model (pipeline) at startup
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found. Train the model first: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Load neighborhood list if available
if NEIGH_PATH.exists():
    with open(NEIGH_PATH, "r", encoding="utf-8") as f:
        neighborhoods = json.load(f)
else:
    neighborhoods = []

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            # Get inputs from form and coerce to the same columns used in training
            data = {
                "OverallQual": float(request.form.get("OverallQual", 0)),
                "GrLivArea": float(request.form.get("GrLivArea", 0)),
                "TotalBsmtSF": float(request.form.get("TotalBsmtSF", 0)),
                "GarageCars": float(request.form.get("GarageCars", 0)),
                "YearBuilt": float(request.form.get("YearBuilt", 0)),
                "Neighborhood": request.form.get("Neighborhood", "")
            }
            X_input = pd.DataFrame([data])
            pred = model.predict(X_input)  # pipeline handles preprocessing
            price = float(pred[0])
            # Format as currency-like string
            prediction = f"${price:,.2f}"
        except Exception as e:
            error = str(e)
            flash(f"Error during prediction: {error}", "danger")
    return render_template("index.html", prediction=prediction, neighborhoods=neighborhoods, error=error)

if __name__ == "__main__":
    app.run(debug=True)

