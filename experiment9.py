
# Experiment 9: K-Means Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# 2. Exploratory Data Analysis
print("\nStatistical summary:")
print(df.describe().round(2))

print("\nGender counts:")
print(df["Genre"].value_counts())

# Plot distributions
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, column in zip(
    axes,
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
):
    sns.histplot(df[column], bins=15, kde=True, ax=ax)
    ax.set_title(column)

plt.tight_layout()
plt.show()

# 3. Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

# 4. Elbow Method and Silhouette Score
wcss = []
silhouette_scores = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X)
    wcss.append(model.inertia_)

    if k > 1:
        silhouette_scores.append(silhouette_score(X, labels))

print("\nK    WCSS       Silhouette Score")
for k in range(1, 11):
    score = "-" if k == 1 else f"{silhouette_scores[k - 2]:.3f}"
    print(f"{k:<4} {wcss[k - 1]:<10.2f} {score}")

# Plot elbow curve and silhouette scores
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(range(1, 11), wcss, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (K)")
axes[0].set_ylabel("WCSS")

axes[1].plot(range(2, 11), silhouette_scores, marker="s")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (K)")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

# 5. Apply K-Means with K = 5
km = KMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    random_state=42
)

df["Cluster"] = km.fit_predict(X)

print("\nCluster centroids (Income, Spending Score):")
print(np.round(km.cluster_centers_, 2))

print("\nInertia:", round(km.inertia_, 2))
print("Iterations:", km.n_iter_)
print("Silhouette Score:", round(silhouette_score(X, df["Cluster"]), 3))

print("\nCustomers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# 6. Visualize clusters and centroids
plt.figure(figsize=(8, 5))

for cluster in range(5):
    points = df[df["Cluster"] == cluster]

    plt.scatter(
        points["Annual Income (k$)"],
        points["Spending Score (1-100)"],
        s=45,
        label=f"Cluster {cluster}"
    )

plt.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    marker="*",
    s=300,
    color="black",
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segmentation (K = 5)")
plt.legend()
plt.tight_layout()
plt.show()

# 7. Profile each cluster
summary = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean")
).round(2)

print("\nCluster profiles:")
print(summary)

# 8. Standardization comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km_scaled = KMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    random_state=42
)

scaled_labels = km_scaled.fit_predict(X_scaled)

print("\nScaled-data inertia:", round(km_scaled.inertia_, 2))

# 9. Predict cluster for new customers
print("\nNew customer predictions:")

new_customers = np.array([
    [70, 80],
    [20, 10]
])

predicted_clusters = km.predict(new_customers)

for customer, cluster in zip(new_customers, predicted_clusters):
    print(
        f"Income: ${customer[0]}k, Spending Score: {customer[1]}"
        f" -> Cluster {cluster}"
    )
