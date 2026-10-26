
# Experiment 8: Simple and Multiple Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load dataset
df = pd.read_csv("Housing.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df[["price", "area", "bedrooms", "bathrooms",
          "stories", "parking"]].head())

print("\nTotal missing values:", df.isnull().sum().sum())

print("\nCorrelation with price:")
print(df[["price", "area", "bedrooms", "bathrooms",
          "stories", "parking"]].corr()["price"].round(3))

# 2. Simple Linear Regression
# Independent variable: area
# Dependent variable: price

X = df[["area"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

slr = LinearRegression()
slr.fit(X_train, y_train)

y_pred = slr.predict(X_test)

print("\n===== SIMPLE LINEAR REGRESSION =====")
print("Intercept:", round(slr.intercept_, 2))
print("Slope:", round(slr.coef_[0], 2))

print("Equation: price = %.2f + %.2f * area"
      % (slr.intercept_, slr.coef_[0]))

print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

# 3. Plot actual prices and regression line
plt.figure(figsize=(7, 5))

plt.scatter(X_test["area"], y_test, label="Actual prices")

area_values = pd.DataFrame({
    "area": np.linspace(df["area"].min(), df["area"].max(), 100)
})

plt.plot(
    area_values["area"],
    slr.predict(area_values),
    color="red",
    label="Regression line"
)

plt.xlabel("Area (square feet)")
plt.ylabel("Price (Rs.)")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.tight_layout()
plt.show()

# Predict price for a 5000 sq. ft. house
new_house = pd.DataFrame({"area": [5000]})
simple_price = slr.predict(new_house)[0]

print("\nPredicted price for 5000 sq. ft. house: Rs. %.0f"
      % simple_price)

# 4. Multiple Linear Regression
features = ["area", "bedrooms", "bathrooms", "stories", "parking"]

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred_mlr = mlr.predict(X_test)

print("\n===== MULTIPLE LINEAR REGRESSION =====")
print("Intercept:", round(mlr.intercept_, 2))

print("\nRegression coefficients:")
for feature, coefficient in zip(features, mlr.coef_):
    print(f"{feature}: Rs. {coefficient:.2f}")

print("\nEvaluation metrics:")
print("MAE:", round(mean_absolute_error(y_test, y_pred_mlr), 2))
print("MSE:", round(mean_squared_error(y_test, y_pred_mlr), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_mlr)), 2))
print("R2 Score:", round(r2_score(y_test, y_pred_mlr), 4))

# 5. Actual vs predicted prices plot
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred_mlr)

limits = [
    min(y_test.min(), y_pred_mlr.min()),
    max(y_test.max(), y_pred_mlr.max())
]

plt.plot(limits, limits, "r--", label="Perfect prediction")

plt.xlabel("Actual price (Rs.)")
plt.ylabel("Predicted price (Rs.)")
plt.title("Multiple Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()

# 6. Predict price for a new house
new_house = pd.DataFrame({
    "area": [6000],
    "bedrooms": [3],
    "bathrooms": [2],
    "stories": [2],
    "parking": [1]
})

multiple_price = mlr.predict(new_house)[0]

print("\nPredicted price for new house: Rs. %.0f"
      % multiple_price)

# 7. Compare both models
print("\n===== MODEL COMPARISON =====")
print("Simple Regression R2:",
      round(r2_score(y_test, slr.predict(X_test[["area"]]))
            if False else r2_score(
                y_test,
                slr.predict(X_test[["area"]])
            ), 4)
      if list(X_test.columns) == ["area"] else
      "See note below")

# Evaluate both models on the same test split for a fair comparison
X_simple = df[["area"]]
X_multiple = df[features]
y_all = df["price"]

X_simple_train, X_simple_test, y_train2, y_test2 = train_test_split(
    X_simple, y_all, test_size=0.2, random_state=42
)

X_multiple_train, X_multiple_test, _, _ = train_test_split(
    X_multiple, y_all, test_size=0.2, random_state=42
)

simple_model = LinearRegression().fit(X_simple_train, y_train2)
multiple_model = LinearRegression().fit(X_multiple_train, y_train2)

print("Simple Regression R2:",
      round(r2_score(y_test2, simple_model.predict(X_simple_test)), 4))
print("Multiple Regression R2:",
      round(r2_score(y_test2, multiple_model.predict(X_multiple_test)), 4))
