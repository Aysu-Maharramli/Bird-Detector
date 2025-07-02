#!/usr/bin/env python3
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

# 0) Get original species list from config
import yaml
cfg = yaml.safe_load(open("src/config.yaml"))
all_species = cfg["species"]

# 1) Read metadata
meta = pd.read_csv("data/processed/metadata/metadata.csv")
print(f"● Loaded metadata: {len(meta)} clips, {meta['species'].nunique()} species")

# 2) Load features & labels
X = np.vstack([np.load(fp) for fp in meta["feature_path"]])
y = meta["label"].values
print(f"● Feature matrix: {X.shape}, labels: {y.shape}")

# 3) Drop classes with only 1 sample so stratify works
counts = pd.Series(y).value_counts()
singles = counts[counts < 2].index.tolist()
if singles:
    mask = ~pd.Series(y).isin(singles)
    X, y = X[mask.values], y[mask.values]
    print(f"⚠️  Dropped {len(singles)} species with only one clip: {[all_species[i] for i in singles]}")

# 4) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"● Train/test split: {X_train.shape}/{X_test.shape}")

# 5) Train a smaller, subsampled RF
print("● Training RandomForestClassifier…")
start = time.time()
clf = RandomForestClassifier(
    n_estimators=100,       # fewer trees
    max_depth=20,           # cap depth
    max_samples=0.5,        # subsample data per tree
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

# Build labels & names that actually appear in the test set
unique_labels = sorted(np.unique(y_test))
target_names = [all_species[i] for i in unique_labels]

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=unique_labels,
    target_names=target_names
))

cm = confusion_matrix(y_test, y_pred, labels=unique_labels, normalize="true")
print("Sample of normalized confusion matrix:\n", cm[:5, :5], "...")

# 7) Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/bird_detector_rf.pkl")
print("\nSaved model to models/bird_detector_rf.pkl")
