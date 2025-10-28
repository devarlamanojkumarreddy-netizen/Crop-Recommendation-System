import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Encode crop labels for performance evaluation
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Drop label for clustering
X = df.drop(columns=['label', 'label_encoded'])

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Estimate bandwidth
bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500)

# Apply Mean Shift clustering(used to discover clusters in a dataset by iteratively shifting data points towards the mean of the points within a given region)
meanshift = MeanShift(bandwidth=bandwidth)
df["cluster"] = meanshift.fit_predict(X_scaled)

# Compute Silhouette Score(measure of how well each data point fits into its assigned cluster compared to other clusters)
if len(np.unique(df["cluster"])) > 1:
    silhouette_avg = silhouette_score(X_scaled, df["cluster"])
    print(f"Silhouette Score: {silhouette_avg:.3f}")
else:
    print("Silhouette Score cannot be computed (only one cluster found).")

# Compute Adjusted Rand Index (ARI) for performance evaluation
ari_score = adjusted_rand_score(df["label_encoded"], df["cluster"])
print(f"Adjusted Rand Index (ARI): {ari_score:.3f}")

# Map clusters to crop names
cluster_crops = df.groupby("cluster")["label"].unique()
cluster_mapping = {cluster: ", ".join(crops[:3]) + ("..." if len(crops) > 3 else "") 
                   for cluster, crops in cluster_crops.items()}  # Show first 3 crop names per cluster

df["cluster_label"] = df["cluster"].map(cluster_mapping)

# 📊 **Plot Temperature vs Rainfall with Clusters**
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(data=df, x="P", y="rainfall", hue="cluster_label", palette="tab10", s=50)

# Adding a legend with original crop names
plt.legend(title="Clusters (Sample Crops)", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.xlabel("P")
plt.ylabel("Rainfall (mm)")
plt.title("Mean Shift Clustering: Temperature vs Rainfall")
plt.grid(True)
plt.tight_layout()
plt.show()
