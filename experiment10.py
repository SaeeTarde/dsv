
# Experiment 10: Decision Tree Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Step 1: Load the Iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Flower classes:", iris.target_names.tolist())
print("\nFirst five records:")
print(X.head())

print("\nClass counts:")
print(y.value_counts().sort_index().to_dict())

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 3: Create and train the Decision Tree
dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=0
)

dt.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = dt.predict(X_test)

# Step 5: Evaluate the model
print("\n--- Decision Tree Evaluation ---")
print("Training accuracy:", round(dt.score(X_train, y_train), 4))
print("Testing accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# Step 6: Visualize the Decision Tree
plt.figure(figsize=(14, 7))

plot_tree(
    dt,
    feature_names=X.columns,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree Classifier - Iris Dataset")
plt.tight_layout()
plt.show()

# Step 7: Print the decision rules
print("\n--- Decision Tree Rules ---")
print(export_text(dt, feature_names=list(X.columns)))

# Step 8: Compare Gini and Entropy
print("\n--- Gini vs Entropy ---")

for criterion in ["gini", "entropy"]:
    model = DecisionTreeClassifier(
        criterion=criterion,
        random_state=0
    )

    model.fit(X_train, y_train)

    print(
        criterion,
        "| Depth:", model.get_depth(),
        "| Leaves:", model.get_n_leaves(),
        "| Test accuracy:", round(model.score(X_test, y_test), 4)
    )

# Step 9: Check accuracy for different tree depths
depths = range(1, 11)
train_accuracies = []
test_accuracies = []

print("\n--- Accuracy vs Tree Depth ---")
print("Depth | Train Accuracy | Test Accuracy")

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=0
    )

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"{depth:5} | {train_acc:.4f}         | {test_acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(depths, train_accuracies, "o-", label="Training Accuracy")
plt.plot(depths, test_accuracies, "s-", label="Testing Accuracy")

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Maximum Tree Depth")
plt.xticks(list(depths))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Step 10: Feature importance
print("\n--- Feature Importance ---")

for name, importance in sorted(
    zip(X.columns, dt.feature_importances_),
    key=lambda item: -item[1]
):
    print(f"{name}: {importance:.4f}")

# Step 11: Display the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names,
    cmap="Blues"
)

plt.title("Confusion Matrix - Iris Dataset")
plt.tight_layout()
plt.show()

# Step 12: Predict the class of new flowers
new_flowers = pd.DataFrame(
    [
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.0, 5.2, 2.3]
    ],
    columns=X.columns
)

predictions = dt.predict(new_flowers)

print("\n--- New Flower Predictions ---")

for flower, prediction in zip(new_flowers.values, predictions):
    print(flower, "->", iris.target_names[prediction])

print("\nExperiment 10 completed successfully!")
