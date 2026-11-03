import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("temperatures.csv")
print(df.head())

X = df[['YEAR']]
months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

year = int(input("Enter year: "))

for month in months:
    y = df[month]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\n", month)
    print("MSE:", mean_squared_error(y_test, pred))
    print("MAE:", mean_absolute_error(y_test, pred))

    result = model.predict(pd.DataFrame({'YEAR': [year]}))
    print("Predicted Temperature:", round(result[0], 2), "°C")

    if month == 'JAN':
        sns.scatterplot(x=df['YEAR'], y=y, label="Actual")
        sns.lineplot(x=df['YEAR'], y=model.predict(X), color='red', label="Prediction")
        plt.title("January Temperature Prediction")
        plt.show()