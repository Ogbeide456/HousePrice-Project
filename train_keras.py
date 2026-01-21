# train_keras.py
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# CONFIG
INPUT_CSV = "train.csv"
MODEL_OUT = "model.h5"
CHECKPOINT_OUT = "best_model.h5"
PREPROCESSOR_OUT = "preprocessor.joblib"
NEIGHBOR_JSON = "neighborhood_categories.json"
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100

FEATURES = [
    "OverallQual", "GrLivArea", "TotalBsmtSF",
    "GarageCars", "FullBath", "YearBuilt", "Neighborhood"
]
TARGET = "SalePrice"

# Load
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Put the dataset named {INPUT_CSV} in this folder and run again.")
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=[TARGET])

missing_cols = [c for c in FEATURES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

X = df[FEATURES]
y = df[TARGET].values.reshape(-1, 1)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)

# Preprocessor (compatibility for OneHotEncoder arg name)
numeric_features = [c for c in FEATURES if c != "Neighborhood"]
categorical_features = ["Neighborhood"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# OneHotEncoder compatibility for different sklearn versions
try:
    ohe_instance = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe_instance = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", ohe_instance)
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Fit preprocessor on train, transform both
preprocessor = preprocessor.fit(X_train)
X_train_prep = preprocessor.transform(X_train)
X_val_prep = preprocessor.transform(X_val)

# Save preprocessor for later inference
joblib.dump(preprocessor, PREPROCESSOR_OUT)
print(f"Saved preprocessor to {PREPROCESSOR_OUT}")

# Save neighborhood categories for UI dropdowns
ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
neighborhood_categories = ohe.categories_[0].tolist()
with open(NEIGHBOR_JSON, "w") as f:
    json.dump(neighborhood_categories, f, indent=2)
print(f"Saved neighborhood categories to {NEIGHBOR_JSON}")

# Build Keras model
input_shape = X_train_prep.shape[1]
model = models.Sequential([
    layers.Input(shape=(input_shape,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)  # regression
])

model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Callbacks: early stopping + best model checkpoint
checkpoint_cb = callbacks.ModelCheckpoint(CHECKPOINT_OUT, save_best_only=True,
                                          monitor="val_root_mean_squared_error", mode="min")
early_cb = callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                   monitor="val_root_mean_squared_error", mode="min")

# Train
history = model.fit(
    X_train_prep, y_train,
    validation_data=(X_val_prep, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_cb, early_cb],
    verbose=1
)

# Save final model (HDF5)
model.save(MODEL_OUT)
print(f"Saved Keras model to {MODEL_OUT}")
print(f"Best checkpoint (if created) saved to {CHECKPOINT_OUT}")
