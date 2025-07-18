import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load full feature set
df = pd.read_csv("../outputs/flattened_features_with_extras.csv").dropna()

# Use only a few rows
df = df.sample(n=20, random_state=42).copy()

# Keep a small subset of audio features
feature_cols = [
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3",
    "chroma_0", "chroma_1", "chroma_2", "chroma_3",
    "tempo", "centroid", "rms", "zcr"
]
# Try to convert all values safely
def safe_cast(val):
    try:
        if isinstance(val, str):
            parsed = eval(val) if "[" in val else float(val)
            return parsed[0] if isinstance(parsed, list) else float(parsed)
        return float(val)
    except:
        return np.nan

df[feature_cols] = df[feature_cols].applymap(safe_cast)
df = df.dropna(subset=feature_cols)

if df.shape[0] == 0:
    raise ValueError("❌ No valid rows left after cleaning!")

# Train and save the model
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=3)
knn.fit(X_scaled)

dump(knn, "../widgets/tunesense_knn_model_FIXED.joblib")
dump(scaler, "../widgets/tunesense_scaler.joblib")
print("✅ Dummy KNN model and scaler saved.")
