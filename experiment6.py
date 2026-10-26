
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 1. Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("Shape of dataset:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

# 2. Statistical summary
print("\nStatistical summary:")
print(df[["Survived", "Pclass", "Age",
          "SibSp", "Parch", "Fare"]].describe().round(2))

# 3. Missing values and duplicate records
print("\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nPercentage of missing values:")
print((missing[missing > 0] / len(df) * 100).round(2))

print("\nDuplicate rows:", df.duplicated().sum())

# 4. Countplots for survival, gender and passenger class
fig, ax = plt.subplots(1, 3, figsize=(13, 4))

sns.countplot(data=df, x="Survived", ax=ax[0])
ax[0].set_title("Survival (0 = No, 1 = Yes)")

sns.countplot(data=df, x="Sex", ax=ax[1])
ax[1].set_title("Passenger Gender")

sns.countplot(data=df, x="Pclass", ax=ax[2])
ax[2].set_title("Passenger Class")

plt.tight_layout()
plt.show()

print("\nSurvival counts:")
print(df["Survived"].value_counts())

print("\nOverall survival rate: {:.2f}%".format(
    df["Survived"].mean() * 100
))

# 5. Distribution of age and fare
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["Age"].dropna(), bins=25, kde=True, ax=ax[0])
ax[0].set_title("Distribution of Passenger Age")

sns.histplot(df["Fare"], bins=30, kde=True, ax=ax[1])
ax[1].set_title("Distribution of Passenger Fare")

plt.tight_layout()
plt.show()

print("\nAge skewness:", round(df["Age"].skew(), 3))
print("Fare skewness:", round(df["Fare"].skew(), 3))

# 6. Survival by gender, class and port
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

sns.countplot(data=df, x="Sex", hue="Survived", ax=ax[0])
ax[0].set_title("Survival by Gender")

sns.countplot(data=df, x="Pclass", hue="Survived", ax=ax[1])
ax[1].set_title("Survival by Passenger Class")

sns.countplot(data=df, x="Embarked", hue="Survived", ax=ax[2])
ax[2].set_title("Survival by Embarkation Port")

plt.tight_layout()
plt.show()

print("\nSurvival rate by gender (%):")
print((df.groupby("Sex")["Survived"].mean() * 100).round(2))

print("\nSurvival rate by class (%):")
print((df.groupby("Pclass")["Survived"].mean() * 100).round(2))

# 7. Survival by age and fare
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

sns.histplot(data=df, x="Age", hue="Survived",
             kde=True, bins=25, ax=ax[0])
ax[0].set_title("Age Distribution by Survival")

sns.boxplot(data=df, x="Survived", y="Age", ax=ax[1])
ax[1].set_title("Age by Survival Status")

sns.boxplot(data=df, x="Pclass", y="Fare", ax=ax[2])
ax[2].set_title("Fare by Passenger Class")

plt.tight_layout()
plt.show()

print("\nAverage age of survivors:",
      round(df[df["Survived"] == 1]["Age"].mean(), 2))

print("Average age of non-survivors:",
      round(df[df["Survived"] == 0]["Age"].mean(), 2))

print("\nAverage fare by class:")
print(df.groupby("Pclass")["Fare"].mean().round(2))

# 8. Detect outliers using the IQR method
print("\nOutlier detection using IQR:")

for col in ["Age", "Fare"]:
    values = df[col].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((values < lower) | (values > upper)).sum()

    print(
        f"{col}: Lower limit={lower:.2f}, "
        f"Upper limit={upper:.2f}, "
        f"Outliers={outliers}"
    )

plt.figure(figsize=(9, 4))
sns.boxplot(data=df[["Age", "Fare"]], orient="h")
plt.title("Boxplots of Age and Fare")
plt.tight_layout()
plt.show()

# 9. Visualize missing values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values in Titanic Dataset")
plt.tight_layout()
plt.show()

# 10. Correlation heatmap
d2 = df.copy()
d2["Sex_code"] = d2["Sex"].map({"male": 0, "female": 1})

columns = [
    "Survived", "Pclass", "Sex_code",
    "Age", "SibSp", "Parch", "Fare"
]

corr = d2[columns].corr()

print("\nCorrelation matrix:")
print(corr.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Titanic Correlation Heatmap")
plt.tight_layout()
plt.show()

# 11. Five significant observations
female_rate = df[df["Sex"] == "female"]["Survived"].mean() * 100
male_rate = df[df["Sex"] == "male"]["Survived"].mean() * 100

class_rates = df.groupby("Pclass")["Survived"].mean() * 100

fare_q1 = df["Fare"].quantile(0.25)
fare_q3 = df["Fare"].quantile(0.75)
fare_iqr = fare_q3 - fare_q1
fare_outliers = (
    (df["Fare"] < fare_q1 - 1.5 * fare_iqr) |
    (df["Fare"] > fare_q3 + 1.5 * fare_iqr)
).sum()

print("\nFIVE SIGNIFICANT OBSERVATIONS")

print(
    f"1. Gender: {female_rate:.1f}% of females survived, "
    f"compared with {male_rate:.1f}% of males."
)

print(
    f"2. Class: survival rates were {class_rates.loc[1]:.1f}% "
    f"in first class, {class_rates.loc[2]:.1f}% in second class "
    f"and {class_rates.loc[3]:.1f}% in third class."
)

print(
    f"3. Age: {df['Age'].isnull().mean() * 100:.1f}% "
    f"of age values are missing. Median age is "
    f"{df['Age'].median():.0f} years."
)

print(
    f"4. Fare: the distribution is right-skewed "
    f"(skewness = {df['Fare'].skew():.2f}) "
    f"with {fare_outliers} IQR outliers."
)

print(
    f"5. Correlation: Survived-Pclass = "
    f"{corr.loc['Survived', 'Pclass']:.2f}, "
    f"Survived-Fare = {corr.loc['Survived', 'Fare']:.2f}, "
    f"Survived-Sex = {corr.loc['Survived', 'Sex_code']:.2f}."
)

print(
    f"Cabin data is missing for "
    f"{df['Cabin'].isnull().mean() * 100:.1f}% of passengers."
)

print("\nEDA completed successfully!")
