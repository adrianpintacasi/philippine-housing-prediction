"""Plot raw and log-transformed price distributions."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('PH_Housing_Cleaned.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Price'] / 1e6, bins=50, color='#2196F3', edgecolor='white')
axes[0].axvline(df['Price'].median() / 1e6, color='red', linestyle='--',
                label=f"Median: P{df['Price'].median()/1e6:.1f}M")
axes[0].axvline(df['Price'].mean() / 1e6, color='orange', linestyle='--',
                label=f"Mean: P{df['Price'].mean()/1e6:.1f}M")
axes[0].set_xlabel('Price (Millions P)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Property Prices', fontweight='bold')
axes[0].legend()

axes[1].hist(np.log(df['Price']), bins=50, color='#4CAF50', edgecolor='white')
axes[1].set_xlabel('Log(Price)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Log-Transformed Price Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_price_distribution.png', dpi=200, bbox_inches='tight')
print("Saved: fig_price_distribution.png")
