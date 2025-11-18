import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

data = pd.read_csv(url, names=columns, na_values='?')
data.dropna(inplace=True)
data = data.apply(pd.to_numeric)
data['target'] = (data['target'] > 0).astype(int)

print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.describe())
print(data['target'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='target', data=data, palette='coolwarm')
plt.title("Heart Disease Distribution")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data['age'], kde=True, color='teal')
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='target', y='chol', data=data, palette='Set2')
plt.title("Cholesterol levels vs Heart Disease")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(data[['age','trestbps','chol','thalach','oldpeak','target']], hue='target', palette='husl')
plt.show()
