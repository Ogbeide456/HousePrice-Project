# train_sklearn.py
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# CONFIG
INPUT_CSV = "train.csv"           # put train.csv in same folder
MODEL_OUT = "house_price_model.pkl"
NEIGHBOR_JSON = "neighborhood_categories.json"
RANDOM_STATE = 42

# FEATURES (change if you want different ones)
FEATURES = [
    "OverallQual", "GrLivArea", "TotalBsmtSF",
    "GarageCars", "FullBath", "YearBuilt", "Neighborhood"
]
TARGET = "SalePrice"

# Load dataset
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Put the dataset named {INPUT_CSV} in this folder and run again.")

df = pd.read_csv(INPUT_CSV)

# Basic drop of rows where target missing
df = df.dropna(subset=[TARGET])

# Keep only rows with required features present (or you can impute)
# We'll impute numeric/categorical separately, so just ensure columns exist
missing_cols = [c for c in FEATURES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

X = df[FEATURES]
y = df[TARGET]

# Split for evaluation (optional)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)

# Define preprocessing
numeric_features = [c for c in FEATURES if c != "Neighborhood"]
categorical_features = ["Neighborhood"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Create OneHotEncoder in a way that's compatible with different sklearn versions
try:
    # newer sklearn uses sparse_output
    ohe_instance = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    # older sklearn used 'sparse' param
    ohe_instance = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", ohe_instance)
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
])

# Train
print("Training scikit-learn RandomForest pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete. Evaluating on validation set...")

preds = pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"Validation RMSE: {rmse:.2f}")

# Save pipeline
joblib.dump(pipeline, MODEL_OUT)
print(f"Saved scikit-learn pipeline to {MODEL_OUT}")

# Save neighborhood categories for dropdowns (optional)
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["ohe"]
# after fit, categories_ is a list of arrays -> only one array for Neighborhood
neighborhood_categories = ohe.categories_[0].tolist()
with open(NEIGHBOR_JSON, "w") as f:
    json.dump(neighborhood_categories, f, indent=2)
print(f"Saved neighborhood categories to {NEIGHBOR_JSON}")
