
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")

print(df.head())


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Elbow Graph
plt.plot(range(1,11), wcss)
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)

y_kmeans = kmeans.fit_predict(X)

# Cluster Visualization
plt.figure(figsize=(8,6))

plt.scatter(X.iloc[:,0],
            X.iloc[:,1],
            c=y_kmeans,
            cmap='rainbow')

# Centroids
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=200,
            c='black',
            marker='X')

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.title("Customer Segmentation using KMeans")

plt.show()
