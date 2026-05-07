import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(DATA_DIR / "X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(DATA_DIR / "X_test_scaled.csv", index=False)

joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

print("Normalisation terminée.")