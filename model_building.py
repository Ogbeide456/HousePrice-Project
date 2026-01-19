import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("train.csv")  # Kaggle House Prices dataset

# Select required features (6 chosen)
features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "Neighborhood"
]

target = "SalePrice"

df = df[features + [target]]

# Handle missing values
df["TotalBsmtSF"].fillna(df["TotalBsmtSF"].median(), inplace=True)
df["GarageCars"].fillna(df["GarageCars"].median(), inplace=True)

# Encode Neighborhood
neighborhood_mapping = {name: idx for idx, name in enumerate(df["Neighborhood"].unique())}
df["Neighborhood"] = df["Neighborhood"].map(neighborhood_mapping)

# Save neighborhood categories
with open("neighborhood_categories.json", "w") as f:
    json.dump(neighborhood_mapping, f, indent=4)

# Feature matrix & target
X = df[features]
y = df[target]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and files saved successfully.")
