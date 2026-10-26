
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Import the dataset
df = pd.read_csv("tesla_data.csv")

# 2. Preprocess the data
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Dataset Shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe().round(2))

# Remove Adj Close if it is identical to Close
if "Adj Close" in df.columns:
    if df["Close"].equals(df["Adj Close"]):
        df = df.drop(columns=["Adj Close"])

# 3. Calculate daily percentage returns
df["Daily_Return_%"] = df["Close"].pct_change() * 100

# 4. Calculate moving averages
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()

# 5. Find highest and lowest closing prices
highest = df.loc[df["Close"].idxmax()]
lowest = df.loc[df["Close"].idxmin()]

print("\nHighest Closing Price:")
print("Date:", highest["Date"].date())
print("Price: $", round(highest["Close"], 2))

print("\nLowest Closing Price:")
print("Date:", lowest["Date"].date())
print("Price: $", round(lowest["Close"], 2))

# 6. Display the top 10 days with highest gains
top10 = df.nlargest(10, "Daily_Return_%")

print("\nTop 10 Days with Highest Daily Gains:")
print(top10[["Date", "Close", "Daily_Return_%"]].to_string(index=False))

# 7. Calculate average return and volatility
print("\nMean Daily Return: {:.4f}%".format(
    df["Daily_Return_%"].mean()
))

print("Standard Deviation of Returns: {:.4f}%".format(
    df["Daily_Return_%"].std()
))

# 8. Plot closing price and moving averages
plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["Close"], label="Closing Price")
plt.plot(df["Date"], df["MA20"], label="20-Day Moving Average")
plt.plot(df["Date"], df["MA50"], label="50-Day Moving Average")

plt.title("Tesla Stock Closing Price Analysis")
plt.xlabel("Date")
plt.ylabel("Price in USD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Plot daily returns histogram
plt.figure(figsize=(10, 5))

plt.hist(
    df["Daily_Return_%"].dropna(),
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Tesla Daily Returns")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 10. Plot daily trading volume
plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["Volume"])
plt.title("Tesla Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nStock market analysis completed successfully!")
