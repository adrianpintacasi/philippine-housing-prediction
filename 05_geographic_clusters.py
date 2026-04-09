"""Apply K-Means (k=5) to property coordinates and profile each cluster."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('PH_Housing_Cleaned.csv')
coords = df[['Latitude', 'Longitude']].values

k = 5
km = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = km.fit_predict(coords)
print(f"Silhouette Score (k={k}): {silhouette_score(coords, df['Cluster']):.4f}")

print(f"\n{'Cluster':<8} {'n':>5} {'Avg Price':>12} {'Median':>12} {'Floor':>9} {'Land':>9}")
print("-" * 60)
for c in range(k):
    sub = df[df['Cluster'] == c]
    print(f"Cluster {c+1:<2d} {len(sub):>5d} "
          f"P{sub['Price'].mean()/1e6:>8.1f}M "
          f"P{sub['Price'].median()/1e6:>8.1f}M "
          f"{sub['Floor Area'].mean():>7.0f}sqm "
          f"{sub['Land Area'].mean():>7.0f}sqm")

colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
fig, ax = plt.subplots(figsize=(10, 12))
for c in range(k):
    mask = df['Cluster'] == c
    ax.scatter(df.loc[mask, 'Longitude'], df.loc[mask, 'Latitude'],
               c=colors[c], s=20, alpha=0.6, label=f'Cluster {c+1}')
    centroid = km.cluster_centers_[c]
    ax.scatter(centroid[1], centroid[0], c=colors[c], s=200,
               marker='X', edgecolors='black', linewidths=1.5, zorder=5)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'Geographic Clusters of Philippine Properties (k={k})', fontweight='bold')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('fig_geographic_clusters.png', dpi=200, bbox_inches='tight')
print("\nSaved: fig_geographic_clusters.png")
