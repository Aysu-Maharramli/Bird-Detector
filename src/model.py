#!/usr/bin/env python3
import os, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# 3) Drop classes with only 1 sample (so stratify works)
counts = pd.Series(y).value_counts()
singles = counts[counts < 2].index.tolist()
if singles:
    mask = ~pd.Series(y).isin(singles)
    X, y = X[mask], y[mask]
    print(f"⚠️  Dropped {len(singles)} species with only one clip")

# 4) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"● Train/test split: {X_train.shape}/{X_test.shape}")

# 5) Fit a smaller, subsampled RF
print("● Training RandomForestClassifier…")
start = time.time()
clf = RandomForestClassifier(
    n_estimators=100,          # cut in half
    max_depth=20,              # cap tree depth
    max_samples=0.5,           # use only half the data per tree
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
clf.fit(X_train, y_train)
print(f"✅ Trained in {time.time() - start:.1f}s")

# 6) Evaluate
y_pred = clf.predict(X_test)
print("\nTest accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=species_names))
cm = confusion_matrix(y_test, y_pred, normalize="true")
print("Normalized CM sample:\n", cm[:5, :5], "...")

# 7) Save
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/bird_detector_rf.pkl")
print("\nSaved model to models/bird_detector_rf.pkl")
