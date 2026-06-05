import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Path dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Kolom yang tidak boleh bernilai 0
kolom = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

# Ubah 0 menjadi NaN
for col in kolom:
    df[col] = df[col].replace(0, np.nan)

# Handling missing value
for col in kolom:
    df.loc[
        (df['Outcome'] == 0) & (df[col].isna()),
        col
    ] = df[df['Outcome'] == 0][col].mean()

    df.loc[
        (df['Outcome'] == 1) & (df[col].isna()),
        col
    ] = df[df['Outcome'] == 1][col].mean()

# Feature dan target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Preprocessing selesai")