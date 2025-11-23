import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from sklearn.utils import resample

# Set up Chinese display for plots
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. Data Loading & Preprocessing

# Load the dataset (replace with your file path)
df = pd.read_csv('F:\桌面\Python Coding\data sources\EastWestAirlinesCluster.csv')

# Check dataset structure and missing values
print("Dataset shape (rows, columns):", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())

# Select features for clustering (exclude ID#, use numeric features only)
features = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
            'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
            'Days_since_enroll', 'Award?']
X = df[features].copy()

# Feature normalization (critical to eliminate scale bias)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)


# 2. Hierarchical Clustering (Euclidean Distance + Ward's Method)

# Plot dendrogram to determine optimal cluster number
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(
    sch.linkage(X_scaled, method='ward', metric='euclidean')
)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Ward Distance')
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Define number of clusters (adjust based on dendrogram)
n_clusters = 4
hierarchical_clusterer = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward',
    metric='euclidean'
)
hierarchical_labels = hierarchical_clusterer.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Hierarchical_Cluster'] = hierarchical_labels

# 3. Analyze Cluster Centroids

# Calculate centroids by averaging samples in each cluster
hierarchical_centroids = []
for cluster_id in range(n_clusters):
    cluster_samples = X_scaled[hierarchical_labels == cluster_id]
    centroid_scaled = cluster_samples.mean(axis=0)
    centroid_original = scaler.inverse_transform([centroid_scaled])[0]
    hierarchical_centroids.append(centroid_original)

hierarchical_centroids = pd.DataFrame(
    hierarchical_centroids,
    columns=features,
    index=[f'Cluster {i+1}' for i in range(n_clusters)]
)
print("\nHierarchical Clustering Centroids (Original Scale):")
print(hierarchical_centroids.round(2))

# 4. Cluster Stability Test (Remove 5% Data)

X_scaled_95 = resample(
    X_scaled,
    replace=False,
    n_samples=int(len(X_scaled)*0.95),
    random_state=42
)

hierarchical_clusterer_95 = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward',
    metric='euclidean'
)
hierarchical_labels_95 = hierarchical_clusterer_95.fit_predict(X_scaled_95)

# Evaluate with Silhouette Score
silhouette_full = silhouette_score(X_scaled, hierarchical_labels)
silhouette_95 = silhouette_score(X_scaled_95, hierarchical_labels_95)
print(f"\nSilhouette Score (Full Data): {silhouette_full:.3f}")
print(f"Silhouette Score (95% Data): {silhouette_95:.3f}")

# 5. K-Means Clustering

kmeans_clusterer = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)
kmeans_labels = kmeans_clusterer.fit_predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels

# Analyze K-means centroids
kmeans_centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans_clusterer.cluster_centers_),
    columns=features,
    index=[f'Cluster {i+1}' for i in range(n_clusters)]
)
print("\nK-Means Clustering Centroids (Original Scale):")
print(kmeans_centroids.round(2))

# Evaluate K-means
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
print(f"\nSilhouette Score (K-Means): {silhouette_kmeans:.3f}")

# 6. Cluster Visualization
plt.figure(figsize=(12, 5))

# Hierarchical Clustering Plot
plt.subplot(1, 2, 1)
for cluster_id in range(n_clusters):
    cluster_samples = df[df['Hierarchical_Cluster'] == cluster_id]
    plt.scatter(
        cluster_samples['Flight_miles_12mo'],
        cluster_samples['Balance'],
        label=f'Cluster {cluster_id+1}',
        alpha=0.6
    )
plt.title('Hierarchical Clustering: Flight Miles vs Balance')
plt.xlabel('Flight Miles (Last 12 Months)')
plt.ylabel('Frequent Flier Balance')
plt.legend()

# K-Means Clustering Plot
plt.subplot(1, 2, 2)
for cluster_id in range(n_clusters):
    cluster_samples = df[df['KMeans_Cluster'] == cluster_id]
    plt.scatter(
        cluster_samples['Flight_miles_12mo'],
        cluster_samples['Balance'],
        label=f'Cluster {cluster_id+1}',
        alpha=0.6
    )
plt.title('K-Means Clustering: Flight Miles vs Balance')
plt.xlabel('Flight Miles (Last 12 Months)')
plt.ylabel('Frequent Flier Balance')
plt.legend()

plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Cluster Size and Labeling
print("\nHierarchical Clustering: Sample Distribution")
hier_cluster_counts = df['Hierarchical_Cluster'].value_counts().sort_index()
for cluster_id, count in hier_cluster_counts.items():
    percentage = (count / len(df)) * 100
    print(f"Cluster {cluster_id+1}: {count} samples ({percentage:.1f}% of total)")

print("\nK-Means Clustering: Sample Distribution")
kmeans_cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()
for cluster_id, count in kmeans_cluster_counts.items():
    percentage = (count / len(df)) * 100
    print(f"Cluster {cluster_id+1}: {count} samples ({percentage:.1f}% of total)")

# Example Cluster Labeling (adjust based on your results)
cluster_labels = {
    0: "High-Engagement Loyalists",
    1: "Credit Card Miles Accumulators",
    2: "Inactive Dormant Users",
    3: "Short-Term Frequent Travelers"
}
print("\nExample Cluster Labels:")
for cluster_id, label in cluster_labels.items():
    print(f"Cluster {cluster_id+1}: {label}")