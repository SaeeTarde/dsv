
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Import datasets
# -----------------------------------
house = pd.read_csv("Housing.csv")
tesla = pd.read_csv("tesla_data.csv")

tesla["Date"] = pd.to_datetime(tesla["Date"])

print("Housing dataset shape:", house.shape)
print(house.head())

print("\nTesla dataset shape:", tesla.shape)
print(tesla[["Date", "Open", "High", "Low", "Close", "Volume"]].head())

# -----------------------------------
# 2. Line plot: Tesla closing price
# -----------------------------------
plt.figure(figsize=(8, 4))
plt.plot(tesla["Date"], tesla["Close"], label="Closing Price")
plt.title("Tesla Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------
# 3. Bar plot: Average house price
# -----------------------------------
avg = house.groupby("furnishingstatus")["price"].mean()
print("\nAverage house price by furnishing status:")
print(avg.round(2))

plt.figure(figsize=(7, 4))
plt.bar(avg.index, avg.values, edgecolor="black")
plt.title("Average House Price by Furnishing Status")
plt.xlabel("Furnishing Status")
plt.ylabel("Average Price (Rs.)")
plt.tight_layout()
plt.show()

# -----------------------------------
# 4. Pie chart: Houses by stories
# -----------------------------------
counts = house["stories"].value_counts().sort_index()
print("\nNumber of houses by stories:")
print(counts)

plt.figure(figsize=(6, 5))
plt.pie(
    counts.values,
    labels=[str(s) + " stories" for s in counts.index],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Houses by Number of Stories")
plt.show()

# -----------------------------------
# 5. Histogram: House price distribution
# -----------------------------------
plt.figure(figsize=(7, 4))
plt.hist(house["price"], bins=10, edgecolor="black")
plt.title("Distribution of House Prices")
plt.xlabel("Price (Rs.)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\nMean house price:", round(house["price"].mean(), 2))
print("Median house price:", round(house["price"].median(), 2))

# -----------------------------------
# 6. Scatter plot: Area vs Price
# -----------------------------------
plt.figure(figsize=(7, 4))
plt.scatter(house["area"], house["price"], marker="*", s=30)
plt.title("House Area vs Price")
plt.xlabel("Area (sq. ft.)")
plt.ylabel("Price (Rs.)")
plt.tight_layout()
plt.show()

print("Correlation between area and price:",
      round(house["area"].corr(house["price"]), 3))

# -----------------------------------
# 7. Box plot and subplots
# -----------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))

# Line plot of Tesla trading volume
ax[0, 0].plot(tesla["Date"], tesla["Volume"])
ax[0, 0].set_title("Tesla Trading Volume")
ax[0, 0].tick_params(axis="x", rotation=30)

# Bar plot: Number of houses by bedrooms
bed = house["bedrooms"].value_counts().sort_index()
ax[0, 1].bar(bed.index, bed.values, edgecolor="black")
ax[0, 1].set_title("Houses by Number of Bedrooms")
ax[0, 1].set_xlabel("Bedrooms")
ax[0, 1].set_ylabel("Number of Houses")

# Histogram: House area
ax[1, 0].hist(house["area"], bins=12, edgecolor="black")
ax[1, 0].set_title("Distribution of House Area")
ax[1, 0].set_xlabel("Area (sq. ft.)")
ax[1, 0].set_ylabel("Frequency")

# Box plot: Price by furnishing status
house.boxplot(
    column="price",
    by="furnishingstatus",
    ax=ax[1, 1]
)
ax[1, 1].set_title("Price by Furnishing Status")
ax[1, 1].set_xlabel("Furnishing Status")
ax[1, 1].tick_params(axis="x", rotation=20)

plt.suptitle("Various Plots Using Matplotlib")
plt.tight_layout()
plt.savefig("subplots.png", dpi=150)
plt.show()

print("\nFigure saved as subplots.png")
