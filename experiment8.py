# 8. Housing data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load first 50 houses showing required columns
df = pd.read_csv('Housing.csv')
print(df[['price', 'area', 'bedrooms']].head())

# 1. Mean, Median, Variance, Standard Deviation of price
print("Mean    :", df['price'].mean())
print("Median  :", df['price'].median())
print("Variance:", df['price'].var())
print("Std Dev :", df['price'].std())

# 2. Correlation between area and price
print("Correlation:", df['area'].corr(df['price']))

# 3. Scatter Plot
plt.scatter(df['area'], df['price'])
plt.xlabel('Carpet Area')
plt.ylabel('Price')
plt.title('Carpet Area vs Price')
plt.show()

# 4. Histogram with Normal Distribution Curve
sns.histplot(df['price'], kde=True)
plt.title('House Price Distribution with Normal Curve')
plt.show()

# 5. Interpretation
print("Interpretation: Positive correlation confirms larger carpet area leads to higher house prices. The histogram with KDE curve shows whether prices are approximately normally distributed.")