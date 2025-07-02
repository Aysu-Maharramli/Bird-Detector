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

np.random.seed(42)

# 1) Read metadata
meta = pd.read_csv("data/processed/metadata/metadata.csv")
species_names = meta["species"].unique()

# 2) Load features & labels
X = np.vstack([np.load(fp) for fp in meta["feature_path"]])
y = meta["label"].values

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4) Fit RF
clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
clf.fit(X_train, y_train)

# 5) Evaluate
y_pred = clf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=species_names))

cm = confusion_matrix(y_test, y_pred, normalize="true")
print("\nNormalized confusion matrix:\n", cm)

# 6) Save model
joblib.dump(clf, "models/bird_detector_rf.pkl")
print("\nSaved model to models/bird_detector_rf.pkl")
