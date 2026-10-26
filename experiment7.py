
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Load and preprocess the dataset
df = pd.read_csv("Titanic-Dataset.csv")

data = df[["Survived", "Pclass", "Sex", "Age"]].copy()

print("Dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())

print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Fill missing ages with the median
data["Age"] = data["Age"].fillna(data["Age"].median())

# Encode gender: male = 0, female = 1
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nPreprocessed data:")
print(data.head())

# 2. Define features and target
X = data[["Sex", "Age", "Pclass"]]
y = data["Survived"]

# 3. Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

# 4. Logistic Regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n===== LOGISTIC REGRESSION =====")
print("Coefficients (Sex, Age, Pclass):", lr.coef_.round(4))
print("Intercept:", lr.intercept_.round(4))

print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_lr,
    target_names=["Not survived", "Survived"]
))

# 5. Predict survival probability for a new passenger
new = pd.DataFrame({
    "Sex": [1],
    "Age": [28],
    "Pclass": [1]
})

prob_lr = lr.predict_proba(new)

print("\n===== NEW PASSENGER: LOGISTIC REGRESSION =====")
print("Probabilities [Not survived, Survived]:", prob_lr.round(4))
print("Predicted class:", lr.predict(new)[0])
print("Survival probability: %.2f%%" % (prob_lr[0][1] * 100))

# 6. Decision Tree model
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n===== DECISION TREE =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_dt,
    target_names=["Not survived", "Survived"]
))

print("\nFeature Importances:")
for feature, importance in zip(X.columns, dt.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# 7. Decision Tree survival probability
prob_dt = dt.predict_proba(new)

print("\n===== NEW PASSENGER: DECISION TREE =====")
print("Probabilities [Not survived, Survived]:", prob_dt.round(4))
print("Survival probability: %.2f%%" % (prob_dt[0][1] * 100))

# 8. Plot the Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(
    dt,
    feature_names=list(X.columns),
    class_names=["Not survived", "Survived"],
    filled=True,
    fontsize=9
)
plt.title("Decision Tree (max_depth = 3)")
plt.tight_layout()
plt.show()

# 9. Compare confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, predictions, name in zip(
    axes,
    [y_pred_lr, y_pred_dt],
    ["Logistic Regression", "Decision Tree"]
):
    sns.heatmap(
        confusion_matrix(y_test, predictions),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Not survived", "Survived"],
        yticklabels=["Not survived", "Survived"]
    )
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 10. Compare model accuracy
print("\n===== MODEL COMPARISON =====")
print("Logistic Regression accuracy:",
      round(accuracy_score(y_test, y_pred_lr), 4))
print("Decision Tree accuracy:",
      round(accuracy_score(y_test, y_pred_dt), 4))
