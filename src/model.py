import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(42)

# 1) Read metadata
meta = pd.read_csv("data/processed/metadata/metadata.csv")
species_names = meta["species"].unique()

# 2) Load features & labels
#    we stack MFCC + delta + delta-delta (precomputed in extract_features.py)
X = np.vstack([np.load(fp) for fp in meta["feature_path"]])
y = meta["label"].values

# 3) Split out a held-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4) Hyperparameter tuning via RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 400],
    'max_depth': [None, 20, 50],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

base_clf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rs = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_dist,
    n_iter=10,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("🔍 Running hyperparameter search...")
rs.fit(X_train, y_train)
print("✅ Best hyperparameters:", rs.best_params_)

# 5) Train final model on full train set
clf = rs.best_estimator_
clf.fit(X_train, y_train)

# 6) Evaluate on the held-out test set
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f}\n")

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=species_names))

cm = confusion_matrix(y_test, y_pred, normalize="true")
print("\nNormalized confusion matrix:\n", cm)

# 7) Persist model
out_path = "models/bird_detector_rf.pkl"
joblib.dump(clf, out_path)
print(f"\nSaved tuned model to {out_path}")
