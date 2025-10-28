import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth, SpectralClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import silhouette_score, confusion_matrix, roc_auc_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


# Load and clean dataset
df = pd.read_csv("Crop_recommendation.csv")
df.fillna(df.mode().iloc[0], inplace=True)

# Label encode target
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# Feature scaling
features = df.drop(columns=["label", "label_encoded"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# === Mean Shift Variants ===
# 1️⃣ Robust Mean Shift
bandwidth_robust = estimate_bandwidth(scaled_features, quantile=0.2, n_samples=500)
robust_meanshift = MeanShift(bandwidth=bandwidth_robust, bin_seeding=True)
df["cluster_robust"] = robust_meanshift.fit_predict(scaled_features)

# 2️⃣ Spectral Clustering
spectral = SpectralClustering(
    n_clusters=5,
    affinity="nearest_neighbors",
    assign_labels="kmeans",
    random_state=42,
    n_neighbors=10
)
df["cluster_spectral"] = spectral.fit_predict(scaled_features)

# 3️⃣ Adaptive Mean Shift
knn = NearestNeighbors(n_neighbors=10)
knn.fit(scaled_features)
distances, _ = knn.kneighbors(scaled_features)
local_bandwidths = distances.mean(axis=1)
adaptive_bandwidth = np.median(local_bandwidths)
adaptive_meanshift = MeanShift(bandwidth=adaptive_bandwidth, bin_seeding=True)
df["cluster_adaptive"] = adaptive_meanshift.fit_predict(scaled_features)

# 4️⃣ Accelerated Mean Shift
bandwidth_fast = estimate_bandwidth(scaled_features, quantile=0.2, n_samples=500)
fast_meanshift = MeanShift(bandwidth=bandwidth_fast, bin_seeding=True)
df["cluster_fast"] = fast_meanshift.fit_predict(scaled_features)

# === Evaluation Function ===
def evaluate_clustering(true_labels, pred_labels):
    sil = silhouette_score(scaled_features, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return sil, ari

# Evaluate
sil_robust, ari_robust = evaluate_clustering(df["label_encoded"], df["cluster_robust"])
sil_spectral, ari_spectral = evaluate_clustering(df["label_encoded"], df["cluster_spectral"])
sil_adaptive, ari_adaptive = evaluate_clustering(df["label_encoded"], df["cluster_adaptive"])
sil_fast, ari_fast = evaluate_clustering(df["label_encoded"], df["cluster_fast"])

# ROC-AUC
true_bin = label_binarize(df["label_encoded"], classes=np.unique(df["label_encoded"]))

def compute_auc(pred_column):
    pred_bin = label_binarize(df[pred_column], classes=np.unique(df[pred_column]))
    min_classes = min(true_bin.shape[1], pred_bin.shape[1])
    return roc_auc_score(true_bin[:, :min_classes], pred_bin[:, :min_classes], average="macro", multi_class="ovr")

auc_robust = compute_auc("cluster_robust")
auc_spectral = compute_auc("cluster_spectral")
auc_adaptive = compute_auc("cluster_adaptive")
auc_fast = compute_auc("cluster_fast")

# === Performance Plots ===
def plot_scores(title, scores, palette):
    methods = list(scores.keys())
    values = list(scores.values())
    sns.barplot(x=methods, y=values, hue=methods, palette=palette, legend=False)
    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()
    plt.close()

plot_scores("Silhouette Score Comparison", {
    "Robust": sil_robust,
    "Spectral": sil_spectral,
    "Adaptive": sil_adaptive,
    "Accelerated": sil_fast
}, palette="pastel")

plot_scores("Adjusted Rand Index (ARI)", {
    "Robust": ari_robust,
    "Spectral": ari_spectral,
    "Adaptive": ari_adaptive,
    "Accelerated": ari_fast
}, palette="coolwarm")

plot_scores("ROC-AUC Score Comparison", {
    "Robust": auc_robust,
    "Spectral": auc_spectral,
    "Adaptive": auc_adaptive,
    "Accelerated": auc_fast
}, palette="Blues_d")

# Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, name, column in zip(axes.flat, ["Robust", "Spectral", "Adaptive", "Accelerated"],
                            ["cluster_robust", "cluster_spectral", "cluster_adaptive", "cluster_fast"]):
    cm = confusion_matrix(df["label_encoded"], df[column])
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("True Label")
plt.tight_layout()
plt.show()
plt.close()

# Cluster Scatter Plots
for name, column in zip(["Robust", "Spectral", "Adaptive", "Accelerated"],
                        ["cluster_robust", "cluster_spectral", "cluster_adaptive", "cluster_fast"]):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="temperature", y="rainfall", hue=column, palette="tab10")
    plt.title(f"{name} Clustering (Temp vs Rainfall)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()
    plt.close()

# Summary
print("\n📈 Performance Summary")
print(f"1. Robust Mean Shift\n   - Silhouette Score : {sil_robust:.4f}\n   - ARI: {ari_robust:.4f}\n   - ROC-AUC : {auc_robust:.4f}")
print(f"\n2. Spectral Clustering\n   - Silhouette Score : {sil_spectral:.4f}\n   - ARI: {ari_spectral:.4f}\n   - ROC-AUC : {auc_spectral:.4f}")
print(f"\n3. Adaptive Mean Shift\n   - Silhouette Score : {sil_adaptive:.4f}\n   - ARI: {ari_adaptive:.4f}\n   - ROC-AUC : {auc_adaptive:.4f}")
print(f"\n4. Accelerated Mean Shift\n   - Silhouette Score : {sil_fast:.4f}\n   - ARI: {ari_fast:.4f}\n   - ROC-AUC : {auc_fast:.4f}")
