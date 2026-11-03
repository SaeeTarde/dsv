import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('tesla_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

high = df.loc[df['Close'].idxmax()]
low = df.loc[df['Close'].idxmin()]

print("Highest Closing Price:")
print(high['Date'], high['Close'])

print("\nLowest Closing Price:")
print(low['Date'], low['Close'])

df['Change %'] = df['Close'].pct_change() * 100

print(df[['Date', 'Close', 'Change %']])

top10 = df.nlargest(10, 'Change %')

print(top10[['Date', 'Close', 'Change %']])

sns.lineplot(data=df, x='Date', y='Close')