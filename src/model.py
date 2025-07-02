#!/usr/bin/env python3
# src/model.py

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# 1) Read metadata
meta = pd.read_csv("data/processed/metadata/metadata.csv")

# 2) Drop any species with fewer than 2 examples
counts = meta["label"].value_counts()
ok_labels = counts[counts >= 2].index
meta = meta[meta["label"].isin(ok_labels)].reset_index(drop=True)

species_names = meta["species"].unique()

# 3) Load features & labels
X = np.vstack([np.load(fp) for fp in meta["feature_path"]])
y = meta["label"].values

# 4) Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5) Fit RF (with balanced class weights)
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train, y_train)

# 6) Evaluate
y_pred = clf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=species_names))

cm = confusion_matrix(y_test, y_pred, normalize="true")
print("\nNormalized confusion matrix:\n", cm)

# 7) Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/bird_detector_rf.pkl")
print("\nSaved model to models/bird_detector_rf.pkl")
