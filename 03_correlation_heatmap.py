"""Pearson correlation heatmap for numeric features."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('PH_Housing_Cleaned.csv')

cols = ['Price', 'Bedrooms', 'Bathrooms', 'Floor Area', 'Land Area', 'Latitude', 'Longitude']
corr = df[cols].corr()

print("Correlations with Price:")
for col in cols[1:]:
    print(f"  {col:15s}  r = {corr.loc['Price', col]:.3f}")

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Heatmap of Numeric Features', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_correlation_heatmap.png', dpi=200, bbox_inches='tight')
print("\nSaved: fig_correlation_heatmap.png")
