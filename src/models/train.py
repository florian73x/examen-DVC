import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
import joblib

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]

best_params = joblib.load(MODELS_DIR / "best_params.pkl")

model = GradientBoostingRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

joblib.dump(model, MODELS_DIR / "gbr_model.pkl")

print("Entraînement terminé.")