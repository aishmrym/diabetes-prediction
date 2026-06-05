import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load dataset
df = pd.read_csv('../data/diabetes.csv')

# kolom yang tidak boleh bernilai 0
kolom = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

# ubah 0 menjadi NaN
for col in kolom:
    df[col] = df[col].replace(0, np.nan)

# handling missing value
for col in kolom:
    df.loc[
        (df['Outcome'] == 0) & (df[col].isna()),
        col
    ] = df[df['Outcome'] == 0][col].mean()

    df.loc[
        (df['Outcome'] == 1) & (df[col].isna()),
        col
    ] = df[df['Outcome'] == 1][col].mean()

# feature dan target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing selesai")