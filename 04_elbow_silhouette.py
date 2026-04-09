"""Elbow method and Silhouette Score for choosing k in K-Means clustering."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('PH_Housing_Cleaned.csv')
coords = df[['Latitude', 'Longitude']].values

K_range = range(3, 13)
wcss, sil_scores = [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(coords, labels))
    print(f"k={k:2d}: WCSS={km.inertia_:9.2f}, Silhouette={sil_scores[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_range), wcss, 'bo-', markersize=8, linewidth=2)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS')
axes[0].set_title('Elbow Method for Optimal k', fontweight='bold')
axes[0].set_xticks(list(K_range))

axes[1].plot(list(K_range), sil_scores, 'rs-', markersize=8, linewidth=2)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by Cluster Count', fontweight='bold')
axes[1].set_xticks(list(K_range))

plt.tight_layout()
plt.savefig('fig_elbow_silhouette.png', dpi=200, bbox_inches='tight')
print("\nSaved: fig_elbow_silhouette.png")
