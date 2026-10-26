
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# --------------------------------------------------
# PART 1: TESLA STOCK ANALYSIS
# --------------------------------------------------

print("=" * 55)
print("PART 1: TESLA STOCK ANALYSIS")
print("=" * 55)

df = pd.read_csv("tesla_data.csv")

# Convert Date to datetime and sort chronologically
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("\nDataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe().round(2))

# Drop Adj Close if it is identical to Close
if "Adj Close" in df.columns:
    if df["Close"].equals(df["Adj Close"]):
        df = df.drop(columns=["Adj Close"])
        print("\nDropped redundant Adj Close column.")

# Calculate daily percentage return
df["Daily_Return_%"] = df["Close"].pct_change() * 100

# Calculate moving averages
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()

# Find highest and lowest closing prices
highest = df.loc[df["Close"].idxmax()]
lowest = df.loc[df["Close"].idxmin()]

print("\nHighest closing price:")
print("Date:", highest["Date"].date())
print("Price: $", round(highest["Close"], 2))

print("\nLowest closing price:")
print("Date:", lowest["Date"].date())
print("Price: $", round(lowest["Close"], 2))

# Display top 10 daily gains
print("\nTop 10 days with highest daily gains:")
top10 = df.nlargest(10, "Daily_Return_%")
print(top10[["Date", "Close", "Daily_Return_%"]].to_string(index=False))

print("\nMean daily return: {:.4f}%".format(
    df["Daily_Return_%"].mean()
))
print("Daily return standard deviation: {:.4f}%".format(
    df["Daily_Return_%"].std()
))

# Cumulative return
df["Cumulative_Return_%"] = (
    (1 + df["Daily_Return_%"] / 100).cumprod() - 1
) * 100

# Plot closing price and moving averages
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Close"], label="Closing Price")
plt.plot(df["Date"], df["MA20"], label="20-Day Moving Average")
plt.plot(df["Date"], df["MA50"], label="50-Day Moving Average")
plt.title("Tesla Stock Closing Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot daily returns
plt.figure(figsize=(10, 4))
plt.hist(df["Daily_Return_%"].dropna(), bins=20, edgecolor="black")
plt.title("Distribution of Tesla Daily Returns")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# PART 2: MULTI-STOCK ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 55)
print("PART 2: MULTI-STOCK ANALYSIS")
print("=" * 55)

big = pd.read_csv("stock_market_data_large.csv")
big["Date"] = pd.to_datetime(big["Date"])

# Select the same period as Tesla
start_date = df["Date"].min()
end_date = df["Date"].max()

# Prepare Tesla records
tesla = pd.read_csv("tesla_data.csv", parse_dates=["Date"])
tesla = tesla[
    (tesla["Date"] >= start_date) &
    (tesla["Date"] <= end_date)
][["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

tesla["Symbol"] = "TSLA"

# Prepare second stock records
second = big[
    (big["Date"] >= start_date) &
    (big["Date"] <= end_date)
][["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

second["Symbol"] = "STOCK2"

# Combine both datasets
stocks = pd.concat([tesla, second], ignore_index=True)
stocks = stocks.sort_values(["Symbol", "Date"]).reset_index(drop=True)

print("\nNumber of records per stock:")
print(stocks.groupby("Symbol")["Close"].count())

# Calculate returns separately for each stock
stocks["Return"] = stocks.groupby("Symbol")["Close"].pct_change()

# Mean and standard deviation
summary = stocks.groupby("Symbol")["Return"].agg(
    ["mean", "std"]
)

summary.columns = ["Mean_Return", "Return_Std_Dev"]

print("\nMean return and volatility:")
print(summary.round(5))

# Rank by average daily return
summary["Rank"] = summary["Mean_Return"].rank(
    ascending=False, method="min"
).astype(int)

print("\nStock ranking by average return:")
print(summary.sort_values("Rank").round(5))

# Identify the most volatile stock
print("\nMost volatile stock:",
      summary["Return_Std_Dev"].idxmax())

# Calculate 20-day moving average separately
stocks["MA20"] = stocks.groupby("Symbol")["Close"].transform(
    lambda values: values.rolling(window=20).mean()
)

# Correlation between daily returns
returns_pivot = stocks.pivot(
    index="Date", columns="Symbol", values="Return"
)

print("\nCorrelation matrix:")
print(returns_pivot.corr().round(4))

# Compare closing prices
closing_pivot = stocks.pivot(
    index="Date", columns="Symbol", values="Close"
)

closing_pivot.plot(figsize=(12, 5))
plt.title("Comparison of Stock Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# PART 3: COMPANY-WISE STATISTICS AND SORTING
# --------------------------------------------------

print("\n" + "=" * 55)
print("PART 3: COMPANY-WISE STATISTICS")
print("=" * 55)

print("\nMaximum closing price per stock:")
print(stocks.groupby("Symbol")["Close"].max().round(2))

print("\nAverage trading volume per stock:")
print(stocks.groupby("Symbol")["Volume"].mean().round(0))

average_close = stocks.groupby("Symbol")["Close"].mean()

print("\nAverage closing price per stock:")
print(average_close.round(2))

print("\nStock with highest average closing price:",
      average_close.idxmax())

print("\nRecords sorted by closing price (descending):")
sorted_stocks = stocks.sort_values(
    "Close", ascending=False
)

print(
    sorted_stocks[["Date", "Symbol", "Close"]]
    .head(10)
    .to_string(index=False)
)

print("\nAnalysis completed successfully!")
