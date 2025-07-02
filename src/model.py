import os
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(42)

# 1) Read metadata
meta = pd.read_csv("data/processed/metadata/metadata.csv")
species_names = meta["species"].unique()
print(f"● Loaded metadata: {len(meta)} clips, {len(species_names)} species")

# 2) Load features & labels
X = np.vstack([np.load(fp) for fp in meta["feature_path"]])
y = meta["label"].values
print(f"● Feature matrix: {X.shape}, labels: {y.shape}")

# 3) Drop any species with only one sample (so stratify works)
counts = pd.Series(y).value_counts()
singles = counts[counts < 2].index.tolist()
if singles:
    mask = ~pd.Series(y).isin(singles)
    X = X[mask]
    y = y[mask]
    print(f"⚠️  Dropped {len(singles)} species with only one clip: {singles}")

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"● Train/test split: {X_train.shape}/{X_test.shape}")

# 5) Fit RF with verbose output
print("● Training RandomForestClassifier (this may take a while)…")
start = time.time()
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
    verbose=2
)
clf.fit(X_train, y_train)
print(f"✅ Trained in {time.time() - start:.1f}s")

# 6) Evaluate
y_pred = clf.predict(X_test)
print("\nTest accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=species_names))
cm = confusion_matrix(y_test, y_pred, normalize="true")
print("\nNormalized confusion matrix sample:\n", cm[:5, :5], "...")

# 7) Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/bird_detector_rf.pkl")
print("\nSaved model to models/bird_detector_rf.pkl")
