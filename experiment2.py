
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --------------------------------------------------
# Function to calculate statistics and draw graphs
# --------------------------------------------------
def analyse(data, xcol, ycol, xlabel, ylabel, title):
    y = data[ycol]

    print("\nStatistics of", ylabel)
    print("Mean:", round(y.mean(), 2))
    print("Median:", round(y.median(), 2))
    print("Variance (sample):", round(y.var(), 2))
    print("Standard deviation:", round(y.std(), 2))

    r = data[xcol].corr(y)
    print("Correlation:", round(r, 4))

    # Scatter plot with best-fit line
    plt.figure(figsize=(7, 4))
    plt.scatter(data[xcol], y, edgecolor="black")

    m, c = np.polyfit(data[xcol], y, 1)
    plt.plot(data[xcol], m * data[xcol] + c)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Scatter Plot: " + title)
    plt.tight_layout()
    plt.show()

    # Histogram with normal distribution curve
    plt.figure(figsize=(7, 4))
    plt.hist(y, bins=8, density=True, edgecolor="black")

    xs = np.linspace(y.min(), y.max(), 200)
    plt.plot(xs, stats.norm.pdf(xs, y.mean(), y.std()))

    plt.xlabel(ylabel)
    plt.ylabel("Density")
    plt.title("Histogram with Normal Curve")
    plt.tight_layout()
    plt.show()

    # Interpretation
    if abs(r) >= 0.7:
        strength = "Strong"
    elif abs(r) >= 0.4:
        strength = "Moderate"
    else:
        strength = "Weak"

    direction = "Positive" if r > 0 else "Negative"

    print("Interpretation:", strength, direction, "correlation")
    print("Skewness:", round(stats.skew(y), 2))
    print("Mean:", round(y.mean(), 2),
          "| Median:", round(y.median(), 2))

    return r


# --------------------------------------------------
# Problem 1: Academic performance of 30 students
# --------------------------------------------------
np.random.seed(7)

hours = np.round(np.random.uniform(1, 9, 30), 1)

attendance = np.clip(
    np.round(55 + 4.5 * hours +
             np.random.normal(0, 6, 30)),
    40, 100
)

marks = np.clip(
    np.round(28 + 5.5 * hours +
             0.25 * (attendance - 75) +
             np.random.normal(0, 6, 30)),
    0, 100
)

students = pd.DataFrame({
    "StudyHours": hours,
    "Attendance": attendance,
    "Marks": marks
})

print("STUDENT DATA")
print(students.head(10))
print("Shape:", students.shape)

r1 = analyse(
    students, "StudyHours", "Marks",
    "Study Hours", "Marks",
    "Student Performance"
)

print("\nStudent Correlation Matrix:")
print(students.corr().round(3))

sns.heatmap(
    students.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Student Correlation Heatmap")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# Problem 2: Employee salary and experience
# --------------------------------------------------
np.random.seed(11)

exp = np.round(np.random.uniform(0.5, 20, 30), 1)

salary = np.round(
    22000 + 3300 * exp +
    np.random.normal(0, 9000, 30),
    -2
)

emp = pd.DataFrame({
    "Experience": exp,
    "Salary": salary
})

print("\nEMPLOYEE DATA")
print(emp.head(10))
print("Shape:", emp.shape)

r2 = analyse(
    emp, "Experience", "Salary",
    "Experience (Years)", "Monthly Salary (Rs.)",
    "Employee Salary and Experience"
)

print(
    "Experience has a strong relationship with salary:",
    abs(r2) >= 0.7
)

# --------------------------------------------------
# Normal distribution: 68-95-99.7 rule
# --------------------------------------------------
p = emp["Salary"]
mu = p.mean()
sd = p.std()

print("\nEmpirical Rule for Employee Salaries:")

for k in (1, 2, 3):
    percentage = (
        ((p > mu - k * sd) & (p < mu + k * sd)).mean()
        * 100
    )
    print(
        f"Within {k} standard deviation(s): "
        f"{percentage:.1f}%"
    )
