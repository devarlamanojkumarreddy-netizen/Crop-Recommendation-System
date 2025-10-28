import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

# Select numerical features for clustering (excluding 'label')
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Generate linkage matrix for dendrogram (using Ward linkage)
linkage_matrix = linkage(features, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram (Ward Linkage)")
dendrogram(linkage_matrix)
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Perform Agglomerative Clustering (Ward linkage)
agg_clustering = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
clusters = agg_clustering.fit_predict(features)

# Add cluster labels to the dataset
data['Cluster'] = clusters

# Save clustered data to a new CSV file
data.to_csv("Clustered_Crop_Data.csv", index=False)

print("Agglomerative clustering completed. Clustered data saved to 'Clustered_Crop_Data.csv'.")
