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
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import urllib.request

#Download Dataset Directly from UCI

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
urllib.request.urlretrieve(url, "heart.csv")

# Load and Preprocess Data
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

data = pd.read_csv("heart.csv", names=columns, na_values='?')
data = data.dropna()        # remove missing values
data['target'] = (data['target'] > 0).astype(int)  # convert 1-4 → 1, and 0 stays 0

X = data.drop('target', axis=1).values
y = data['target'].values
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper Functions
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def information_gain(y, left_idxs, right_idxs):
    parent_entropy = entropy(y)
    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    if n_l == 0 or n_r == 0:
        return 0
    e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
    child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
    return parent_entropy - child_entropy
# 4️⃣ Decision Tree Classes

class Node
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_gain = -1
        best_feat, best_thresh = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idxs = np.argwhere(X[:, feature] <= threshold).flatten()
                right_idxs = np.argwhere(X[:, feature] > threshold).flatten()
                gain = information_gain(y, left_idxs, right_idxs)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature
                    best_thresh = threshold

        if best_gain == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
        right_idxs = np.argwhere(X[:, best_feat] > best_thresh).flatten()
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Train & Evaluate

tree = DecisionTree(max_depth=6)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print(" Decision Tree Accuracy (Manual Implementation):", round(accuracy * 100, 2), "%")
