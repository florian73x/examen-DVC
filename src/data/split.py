import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/raw.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# garder uniquement les colonnes numériques
df = df.select_dtypes(include=["number"])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_train.to_frame(name=df.columns[-1]).to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_frame(name=df.columns[-1]).to_csv(OUTPUT_DIR / "y_test.csv", index=False)

print("Split terminé.")
print("Colonnes utilisées :", list(df.columns))