import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Encode crop labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Drop label for clustering
X = df.drop(columns=['label', 'label_encoded'])

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================
# 1. ADAPTIVE MEAN SHIFT (Variable Bandwidth)
# ================================================
# Estimate individual bandwidths using k-nearest neighbors
knn = NearestNeighbors(n_neighbors=10)
knn.fit(X_scaled)
distances, _ = knn.kneighbors(X_scaled)

# Take average distance to neighbors as bandwidth per point
local_bandwidths = distances.mean(axis=1)

# Use median as representative bandwidth (adaptive heuristic)
adaptive_bandwidth = np.median(local_bandwidths)

# Apply Adaptive Mean Shift
adaptive_meanshift = MeanShift(bandwidth=adaptive_bandwidth, bin_seeding=True)
df["cluster_adaptive"] = adaptive_meanshift.fit_predict(X_scaled)

# ================================================
# 2. ACCELERATED MEAN SHIFT (default uses KD-Tree)
# ================================================
# Use global bandwidth with bin_seeding for speed
bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500)
fast_meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
df["cluster_fast"] = fast_meanshift.fit_predict(X_scaled)

# ================================================
# Evaluation
# ================================================
def evaluate_clustering(true_labels, pred_labels, label=""):
    if len(np.unique(pred_labels)) > 1:
        silhouette = silhouette_score(X_scaled, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        print(f"\n\U0001F50D {label} Clustering:")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  Adjusted Rand Index (ARI): {ari:.3f}")
        return silhouette, ari
    else:
        print(f"\n❗ {label}: Only one cluster found. Metrics not computed.")
        return None, None

sil_adaptive, ari_adaptive = evaluate_clustering(df["label_encoded"], df["cluster_adaptive"], "Adaptive Mean Shift")
sil_fast, ari_fast = evaluate_clustering(df["label_encoded"], df["cluster_fast"], "Accelerated Mean Shift")

# ================================================
# Visualization
# ================================================
def plot_clusters(column, title):
    cluster_crops = df.groupby(column)["label"].unique()
    cluster_mapping = {
        cluster: ", ".join(crops[:3]) + ("..." if len(crops) > 3 else "")
        for cluster, crops in cluster_crops.items()
    }
    df["cluster_label_plot"] = df[column].map(cluster_mapping)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="P", y="rainfall", hue="cluster_label_plot", palette="tab10", s=50)
    plt.legend(title="Clusters (Sample Crops)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("P")
    plt.ylabel("Rainfall (mm)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_clusters("cluster_adaptive", "Adaptive Mean Shift: P vs Rainfall")
plot_clusters("cluster_fast", "Accelerated Mean Shift: P vs Rainfall")

# ================================================
# Performance Bar Plot
# ================================================
if sil_adaptive is not None and sil_fast is not None:
    perf_df = pd.DataFrame({
        "Method": ["Adaptive Mean Shift", "Accelerated Mean Shift"],
        "Silhouette Score": [sil_adaptive, sil_fast],
        "ARI": [ari_adaptive, ari_fast]
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=perf_df, x="Method", y="Silhouette Score", hue="Method", palette="pastel", ax=axes[0], legend=False)
    axes[0].set_title("Silhouette Score Comparison")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=15)

    sns.barplot(data=perf_df, x="Method", y="ARI", hue="Method", palette="coolwarm", ax=axes[1], legend=False)
    axes[1].set_title("Adjusted Rand Index Comparison")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.show()
