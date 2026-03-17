import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

print("Loading dataset...")
df = pd.read_csv("hand_dataset.csv", header=None)

X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values   

print(f"  Samples : {len(y)}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes : {sorted(set(y))}")

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nEncoded {len(le.classes_)} classes: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete!")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(model, "gesture_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\nSaved: gesture_model.pkl")
print("Saved: label_encoder.pkl")
