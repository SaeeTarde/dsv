
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.linear_model import LinearRegression

# ----------------------------------
# 1. Import and display CSV data
# ----------------------------------
df = pd.read_csv("Heart.csv")

print("FIRST 5 RECORDS")
print(df.head())

print("\nLAST 3 RECORDS")
print(df.tail(3))

print("\nCOLUMN NAMES")
print(df.columns.tolist())

# ----------------------------------
# 2. Shape of dataset
# ----------------------------------
print("\nSHAPE OF DATA:", df.shape)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# ----------------------------------
# 3. Missing values
# ----------------------------------
print("\nMISSING VALUES")
print(df.isnull().sum())
print("Total missing values:", df.isnull().sum().sum())

# ----------------------------------
# 4. Data types
# ----------------------------------
print("\nDATA TYPES")
print(df.dtypes)

print("\nDATASET INFORMATION")
df.info()

# ----------------------------------
# 5. Count zeros in each column
# ----------------------------------
print("\nZERO VALUES")
print((df == 0).sum())

# ----------------------------------
# 6. Mean age
# ----------------------------------
print("\nMEAN AGE:", round(df["Age"].mean(), 2))
print("MINIMUM AGE:", df["Age"].min())
print("MAXIMUM AGE:", df["Age"].max())

# ----------------------------------
# 7. Extract selected columns
# ----------------------------------
subset = df[["Age", "Sex", "ChestPain", "RestBP", "Chol"]]

print("\nSELECTED COLUMNS")
print(subset.head())
print("Subset shape:", subset.shape)

# ----------------------------------
# 8. Split data into training/testing
# ----------------------------------
train, test = train_test_split(
    subset,
    test_size=0.25,
    random_state=0
)

print("\nTRAINING SHAPE:", train.shape)
print("TESTING SHAPE:", test.shape)

# ----------------------------------
# 9. Export CSV files
# ----------------------------------
train.to_csv("heart_train.csv", index=False)
test.to_csv("heart_test.csv", index=False)

print("\nCSV FILES EXPORTED SUCCESSFULLY")
print("Reopened training file shape:",
      pd.read_csv("heart_train.csv").shape)

# ----------------------------------
# 10. Statistical summary
# ----------------------------------
print("\nSTATISTICAL SUMMARY")
print(df[["Age", "RestBP", "Chol", "MaxHR", "Oldpeak"]]
      .describe().round(2))

# ----------------------------------
# 11. Correlation
# ----------------------------------
print("\nCORRELATION MATRIX")
print(df[["Age", "RestBP", "Chol", "MaxHR", "Oldpeak"]]
      .corr().round(3))

# ----------------------------------
# 12. Confusion matrix: COVID test
# ----------------------------------
TP = 45
FP = 100 - TP
FN = 50 - TP
TN = 500 - TP - FP - FN

y_true = [1] * TP + [0] * FP + [1] * FN + [0] * TN
y_pred = [1] * TP + [1] * FP + [0] * FN + [0] * TN

cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

print("\nCONFUSION MATRIX")
print(pd.DataFrame(
    cm,
    index=["Actual Positive", "Actual Negative"],
    columns=["Predicted Positive", "Predicted Negative"]
))

print("Accuracy:", round(accuracy_score(y_true, y_pred), 2))
print("Precision:", round(precision_score(y_true, y_pred), 2))
print("Recall:", round(recall_score(y_true, y_pred), 2))
print("F1-score:", round(f1_score(y_true, y_pred), 2))

# ----------------------------------
# 13. Linear Regression: Age vs MaxHR
# ----------------------------------
X = df[["Age"]]
y = df["MaxHR"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLINEAR REGRESSION RESULTS")
print("Intercept:", round(model.intercept_, 3))
print("Coefficient:", round(model.coef_[0], 3))
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("R-squared:", round(r2_score(y_test, y_pred), 3))

# ----------------------------------
# 14. Plot regression line
# ----------------------------------
X_sorted = X_test.sort_values("Age")
y_sorted_pred = model.predict(X_sorted)

plt.scatter(X_test["Age"], y_test, label="Actual values")
plt.plot(
    X_sorted["Age"],
    y_sorted_pred,
    color="red",
    label="Regression line"
)

plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.title("Linear Regression: Age vs MaxHR")
plt.legend()
plt.tight_layout()
plt.show()
