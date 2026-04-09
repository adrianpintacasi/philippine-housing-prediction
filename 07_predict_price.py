"""Interactive price prediction using the best-performing model (XGBoost).

Trains on all 1,289 records after IQR filtering, then prompts for property
features and returns a predicted price in Philippine Pesos.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
    MODEL_NAME = "XGBoost"
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    MODEL_NAME = "Gradient Boosting (XGBoost fallback)"

# ---- Prepare data (same preprocessing as 06_regression.py) ----
df = pd.read_csv('PH_Housing_Cleaned.csv')
Q1, Q3 = df['Price'].quantile(0.25), df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price'] >= Q1 - 1.5 * IQR) & (df['Price'] <= Q3 + 1.5 * IQR)].copy()
for col in ['Floor Area', 'Land Area']:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))
df['Log_Price'] = np.log(df['Price'])

features = ['Bedrooms', 'Bathrooms', 'Floor Area', 'Land Area', 'Latitude', 'Longitude']
X = df[features].values
y_log = df['Log_Price'].values

# ---- Train on all data ----
if MODEL_NAME == "XGBoost":
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.01, reg_lambda=1.0, random_state=42, verbosity=0
    )
else:
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=3, random_state=42
    )

model.fit(X, y_log)
print(f"{MODEL_NAME} trained on {len(df)} properties")
print(f"Training price range: P{df['Price'].min():,.0f} to P{df['Price'].max():,.0f}\n")


def predict_price(bedrooms, bathrooms, floor_area, land_area, latitude, longitude):
    """Predict property price in Philippine Pesos from six features."""
    x = np.array([[bedrooms, bathrooms, floor_area, land_area, latitude, longitude]])
    return float(np.exp(model.predict(x)[0]))


# ---- Sample predictions ----
print("Sample predictions from the dataset:")
print(f"{'Description':<45s} {'Actual':>12s} {'Predicted':>12s} {'Error':>8s}")
print("-" * 80)
samples = df.sort_values('Price').iloc[::len(df)//10].head(10)
for _, row in samples.iterrows():
    pred = predict_price(row['Bedrooms'], row['Bathrooms'], row['Floor Area'],
                         row['Land Area'], row['Latitude'], row['Longitude'])
    err = abs(row['Price'] - pred) / row['Price'] * 100
    desc = str(row['Description'])[:43]
    print(f"{desc:<45s} P{row['Price']/1e6:>8.2f}M  P{pred/1e6:>8.2f}M  {err:>5.1f}%")

# ---- Interactive loop ----
print("\nInteractive price prediction. Type 'quit' to exit.\n")
print("Reference ranges:")
print(f"  Bedrooms:   {df['Bedrooms'].min():.0f}-{df['Bedrooms'].max():.0f} (median {df['Bedrooms'].median():.0f})")
print(f"  Bathrooms:  {df['Bathrooms'].min():.0f}-{df['Bathrooms'].max():.0f} (median {df['Bathrooms'].median():.0f})")
print(f"  Floor Area: {df['Floor Area'].min():.0f}-{df['Floor Area'].max():.0f} sqm")
print(f"  Land Area:  {df['Land Area'].min():.0f}-{df['Land Area'].max():.0f} sqm")
print(f"  Latitude:   {df['Latitude'].min():.2f}-{df['Latitude'].max():.2f} (Metro Manila ~14.5)")
print(f"  Longitude:  {df['Longitude'].min():.2f}-{df['Longitude'].max():.2f} (Metro Manila ~121.0)\n")

while True:
    try:
        raw = input("Bedrooms (or 'quit'): ").strip()
        if raw.lower() in ('quit', 'q', 'exit'):
            break
        bedrooms = float(raw)
        bathrooms = float(input("Bathrooms: "))
        floor_area = float(input("Floor Area (sqm): "))
        land_area = float(input("Land Area (sqm): "))
        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))

        price = predict_price(bedrooms, bathrooms, floor_area, land_area, latitude, longitude)
        print(f"\n  Predicted Price: P{price:,.0f}  (P{price/1e6:.2f}M)")

        similar = df[(df['Bedrooms'] == bedrooms) & (abs(df['Floor Area'] - floor_area) < 50)]
        if len(similar) > 0:
            print(f"  Context: {len(similar)} similar properties, median P{similar['Price'].median():,.0f}")
        print()

    except ValueError:
        print("Please enter valid numbers.\n")
    except (EOFError, KeyboardInterrupt):
        break

print("Exited.")
