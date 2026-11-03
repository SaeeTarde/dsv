# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Basic Analysis
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())

# 1. Age Distribution
sns.histplot(df['Age'].dropna(), kde=True)
plt.title("Age Distribution")
plt.show()

# 2. Survival Count
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.show()

# 3. Survival by Gender
sns.barplot(data=df, x='Sex', y='Survived')
plt.title("Survival by Gender")
plt.show()

# 4. Correlation Heatmap
sns.heatmap(df.select_dtypes('number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()