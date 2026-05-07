import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")

METRICS_DIR.mkdir(parents=True, exist_ok=True)

X_test = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

model = joblib.load(MODELS_DIR / "gbr_model.pkl")
predictions = model.predict(X_test)

pred_df = pd.DataFrame({
    "y_true": y_test,
    "prediction": predictions
})
pred_df.to_csv("data/predictions.csv", index=False)

scores = {
    "mse": mean_squared_error(y_test, predictions),
    "rmse": mean_squared_error(y_test, predictions) ** 0.5,
    "mae": mean_absolute_error(y_test, predictions),
    "r2": r2_score(y_test, predictions)
}

with open(METRICS_DIR / "scores.json", "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=4)

print("Évaluation terminée.")
print(scores)