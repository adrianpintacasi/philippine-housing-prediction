"""Data cleaning, feature engineering, and descriptive statistics.

Produces PH_Housing_Cleaned.csv, which is the single input file used by every
downstream script in the pipeline.
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('PH_Housing.csv')
print(f"Loaded {len(df)} raw records")

# Drop rows without price, impute the rest with medians
df = df.dropna(subset=['Price'])
for col in ['Bedrooms', 'Bathrooms', 'Floor Area', 'Land Area']:
    df[col] = df[col].fillna(df[col].median())
df['Location'] = df['Location'].fillna('Unknown')

# Derived features
df['Price_per_sqm'] = (df['Price'] / df['Floor Area']).replace([np.inf, -np.inf], np.nan)
df['Price_per_sqm'] = df['Price_per_sqm'].fillna(df['Price_per_sqm'].median())

def infer_type(desc):
    d = str(desc).lower()
    if 'condo' in d: return 'Condominium'
    if 'townhouse' in d: return 'Townhouse'
    if 'lot' in d and 'house' not in d: return 'Lot'
    if 'house' in d or 'single detached' in d: return 'House'
    if 'duplex' in d: return 'Duplex'
    return 'Other'

df['Property_Type'] = df['Description'].apply(infer_type)
df.to_csv('PH_Housing_Cleaned.csv', index=False)
print(f"Saved PH_Housing_Cleaned.csv ({len(df)} records)")

print(f"\nPrice: mean=P{df['Price'].mean():,.0f}, median=P{df['Price'].median():,.0f}")
print(f"Skewness: {stats.skew(df['Price']):.2f}")
