import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


df=pd.read_csv("Mall_Customers.csv")
df.head()

sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)'
)

sns.scatterplot(
    data=df,
    x='Age',
    y='Spending Score (1-100)'
)

#Feature Selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
    
sns.lineplot(x=range(1,11), y=wcss)

#Optimal K=5

kmeans=KMeans(n_clusters=5)
kmeans.fit(X)

#y_pred will have the cluster label 
y_pred=kmeans.predict(X)
y_pred


sns.scatterplot(
    data=X,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue=y_pred,
    palette='rainbow',
)
