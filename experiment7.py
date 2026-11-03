# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# 1. Data Loading
df = pd.read_csv("StudentPerformanceFactors.csv")

# Select first 30 students
data = df[['Hours_Studied', 'Attendance', 'Exam_Score']].head(30)
print(data)

# 2. Statistical Analysis
print("\nStatistical Analysis:")
print("Mean:", data['Exam_Score'].mean())
print("Median:", data['Exam_Score'].median())
print("Variance:", data['Exam_Score'].var())
print("Standard Deviation:", data['Exam_Score'].std())

# Correlation
corr = data['Hours_Studied'].corr(data['Exam_Score'])
print("\nCorrelation:", corr)

# 3. Scatter Plot
sns.scatterplot(data=df, x='Hours_Studied', y='Exam_Score')
sns.scatterplot(data=df, x='Attendance', y='Exam_Score')
# 4. Histogram with Normal Curve
sns.histplot(df['Exam_Score'], kde=True)


