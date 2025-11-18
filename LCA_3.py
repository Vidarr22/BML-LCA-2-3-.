!pip install wget joblib scikit-learn seaborn --quiet

import os, zipfile, warnings
warnings.filterwarnings("ignore")

import wget
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.utils import class_weight
import joblib

# 1) Download & Load Dataset
ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
ZIP_LOCAL = "bank-additional.zip"
CSV_PATH_IN_ZIP = "bank-additional/bank-additional-full.csv"

if not os.path.exists(ZIP_LOCAL):
    print("Downloading dataset...")
    wget.download(ZIP_URL, ZIP_LOCAL)
    print("\nDownload complete.")

with zipfile.ZipFile(ZIP_LOCAL, 'r') as zf:
    with zf.open(CSV_PATH_IN_ZIP) as f:
        df = pd.read_csv(f, sep=';')

print(" Data loaded:", df.shape, "rows and columns")

# 2) EDA
pd.set_option('display.max_columns', 50)
print("\n--- Target variable distribution ---")
print(df['y'].value_counts(), "\n")
print(df['y'].value_counts(normalize=True))

# Plot target
plt.figure(figsize=(4,3))
sns.countplot(x='y', data=df)
plt.title("Target Distribution")
plt.show()

# feature vs target
plt.figure(figsize=(6,4))
pout = df.groupby('poutcome')['y'].value_counts(normalize=True).unstack().fillna(0)
pout[['yes','no']].plot(kind='bar', stacked=False)
plt.title("poutcome vs y")
plt.ylabel("Proportion")
plt.show()

# 3) Feature Selection
selected_features = ['poutcome', 'campaign', 'age', 'job', 'euribor3m']
X = df[selected_features].copy()
y = (df['y'] == 'yes').astype(int)
print("Selected features:", selected_features)
print("Positive class ratio:", y.mean())

# 4) Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# 5) Preprocessing Pipeline
categorical_features = ['poutcome', 'job']
numeric_features = ['campaign', 'age', 'euribor3m']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])

# 6) Model Definition
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

pipeline = Pipeline([
    ('pre', preprocessor),
    ('rf', rf)
])

# 7) Train Model
print("Training Random Forest...")
pipeline.fit(X_train, y_train)
print(" Training complete")

# 8) Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
