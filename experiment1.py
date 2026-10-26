
import pandas as pd
import numpy as np

# Display library versions
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

# 1. Create a Pandas Series
s = pd.Series([10, 20, 30, 40],
              index=["a", "b", "c", "d"])

print("\nPandas Series:")
print(s)
print("Value at index b:", s["b"])

# 2. Create a DataFrame
df = pd.DataFrame({
    "Name": ["Amit", "Priya", "Rahul", "Sneha"],
    "Age": [20, 21, 20, 22],
    "Marks": [85, 78, 92, 70]
})

print("\nStudent DataFrame:")
print(df)
print("Shape:", df.shape)

# Select rows and columns
print("\nUsing loc:")
print(df.loc[1, "Name"])

print("\nUsing iloc:")
print(df.iloc[0:2, 0:2])

# Filter students with marks greater than 75
print("\nStudents with Marks > 75:")
print(df[df["Marks"] > 75])

# 3. Product DataFrame: Stock < 10
products = pd.DataFrame({
    "Name": ["Laptop", "Mouse", "Keyboard",
             "Monitor", "Printer", "Webcam"],
    "Category": ["Computers", "Accessories",
                 "Accessories", "Computers",
                 "Office", "Accessories"],
    "Price": [55000, 500, 1200, 12000, 8500, 2500],
    "Stock": [5, 40, 25, 8, 12, 3]
})

print("\nProducts with Stock < 10:")
print(products[products["Stock"] < 10])

# 4. Find employee with maximum salary
emp = pd.DataFrame({
    "EmpID": [101, 102, 103, 104, 105, 106],
    "Name": ["Ravi", "Anita", "Karan",
             "Meera", "Suresh", "Neha"],
    "Age": [24, 31, 45, 29, 38, 52],
    "Dept": ["IT", "HR", "IT",
             "Finance", "Sales", "Finance"],
    "Salary": [40000, 52000, 95000,
               61000, 58000, 120000]
})

idx = emp["Salary"].idxmax()

print("\nEmployee with Maximum Salary:")
print(emp.loc[idx])

# 5. Add Experience_Level using np.where
emp["Experience_Level"] = np.where(
    emp["Age"] > 35, "Senior", "Junior"
)

print("\nExperience Levels:")
print(emp[["Name", "Age", "Experience_Level"]])

# 6. Merge student names and marks
students = pd.DataFrame({
    "RollNo": [1, 2, 3, 4, 5],
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Vikas"]
})

marks = pd.DataFrame({
    "RollNo": [2, 3, 4, 5, 6],
    "Marks": [78, 92, 70, 66, 88]
})

result = pd.merge(students, marks,
                  on="RollNo", how="inner")

print("\nInner Merge Result:")
print(result)

# 7. Group sales by region
sales = pd.DataFrame({
    "Region": ["East", "West", "East", "North",
               "West", "North", "East", "South"],
    "Sales": [250, 400, 150, 300, 200, 500, 350, 275]
})

summary = sales.groupby("Region")["Sales"].agg(
    ["sum", "count"]
)

print("\nRegion-wise Sales Summary:")
print(summary)

# 8. Sort employees by salary
print("\nEmployees Sorted by Salary:")
print(emp.sort_values(by="Salary", ascending=False))

# 9. Add, rename, and drop a column
emp["Bonus"] = emp["Salary"] * 0.10
emp = emp.rename(columns={"Salary": "Pay"})

print("\nEmployee Pay and Bonus:")
print(emp[["Name", "Pay", "Bonus"]])

emp = emp.drop(columns=["Bonus"])
print("\nColumns after dropping Bonus:")
print(emp.columns.tolist())

# 10. Find and remove duplicate rows
d = pd.DataFrame({
    "A": [1, 2, 2, 3],
    "B": ["x", "y", "y", "z"]
})

print("\nNumber of duplicate rows:", d.duplicated().sum())

print("\nAfter removing duplicates:")
print(d.drop_duplicates())

# 11. Concatenate DataFrames
print("\nConcatenation Result:")
print(pd.concat([d.head(2), d.tail(2)],
                ignore_index=True))
