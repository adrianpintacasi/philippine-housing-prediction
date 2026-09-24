# Mapping Value and Predicting Prices

**A machine learning and spatial analysis of the Philippine housing market**

![Python](https://img.shields.io/badge/Python-3.9--3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189FDD)
![Folium](https://img.shields.io/badge/Folium-maps-77B829)

This project predicts residential property prices in the Philippines using only six basic listing attributes. It also maps how prices differ across regions. The data is 1,500 real listings from Lamudi. The project compares four regression models of increasing complexity under 5-fold cross-validation, groups the listings into five regional submarkets with K-Means clustering, and produces interactive maps of where prices are high and where housing is affordable.

**Headline result:** XGBoost explains **81% of the variance in listed prices** (R² = 0.81, MAPE = 24.7%) from just bedrooms, bathrooms, floor area, land area and coordinates.

---

## Why this problem

Pricing a home in the Philippines is hard to do fairly. The Bureau of Internal Revenue (BIR) and local government units each publish their own property values, and those values often differ from market prices. Residential prices are also rising faster than household incomes: the BSP reported +7.6% nationwide and +13.9% in Metro Manila year-on-year in Q1 2025. A buyer looking at a listing has little independent evidence of whether the asking price is reasonable.

The project asks two questions:

1. **How much of a listing's price can be explained by its basic, publicly available attributes?**
2. **How does pricing differ across regions of the country?**

## Results

### Price prediction: 5-fold cross-validation on 1,289 listings

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Linear Regression | 0.208 ± 0.100 | ₱5.24M | ₱9.80M | 47.7% |
| Decision Tree | 0.617 ± 0.049 | ₱3.58M | ₱6.81M | 34.1% |
| Random Forest | 0.782 ± 0.020 | ₱2.70M | ₱5.13M | 27.1% |
| **XGBoost** | **0.810 ± 0.026** | **₱2.51M** | **₱4.79M** | **24.7%** |

Metrics are the mean ± standard deviation across folds. They are computed in pesos, after converting the log-scale predictions back.

- **Accuracy rises with model complexity.** Linear regression is the baseline. A single tree captures non-linear effects. Bagging (Random Forest) makes the predictions more stable. Boosting (XGBoost) reduces the error further.
- **Location matters, but not in a linear way.** Latitude and longitude have almost no linear correlation with price (r ≈ 0.03). Tree models still use them for 10–15% of their feature importance, because being in Metro Manila matters but a higher latitude number does not. This explains why linear regression performs poorly.
- **Floor area is the main driver of price.** It accounts for 60–80% of feature importance in all three tree models. This is consistent with hedonic pricing theory, which treats a home's price as the sum of the values of its features.
- **The result is competitive with published studies.** Ramolete et al. (2023) reached R² = 0.84 on Lamudi data using 39+ features, including government and neighbourhood indicators. This project reaches 0.81 with six features.

### Regional submarkets: K-Means on coordinates, k = 5, silhouette = 0.648

| Zone | Listings | Median price | Avg. floor area |
|---|---|---|---|
| Metro Manila & Southern Luzon | 881 (60.7%) | ₱15.0M | 260 sqm |
| Northern Luzon (Pangasinan, Baguio) | 93 | ₱8.2M | 115 sqm |
| Mindanao (Davao, Zamboanga) | 53 | ₱7.6M | 139 sqm |
| Visayas (Cebu, Iloilo) | 100 | ₱7.0M | 212 sqm |
| Central Luzon (Tarlac, Bulacan, Subic) | 324 | ₱5.9M | 109 sqm |

The median price in Central Luzon is about **61% lower** than in Metro Manila. On the price-tier map, luxury listings (over ₱20M) sit almost entirely in Metro Manila, while affordable listings (under ₱5M) are spread across the provinces. The data behaves like several separate regional markets rather than one national market.

k = 5 was chosen from the elbow in the within-cluster error. k = 3 had a higher silhouette score (0.85), but it grouped Metro Manila together with northern Luzon provinces, which hides the price differences this analysis is trying to show.

## Approach

```
Raw listings (1,500)
  └─ 01 Clean ──────────── drop missing prices, median-impute, derive price/sqm and property type   → 1,451 rows
      ├─ 02–03 Explore ─── price skewness 13.06 → log transform (skewness −0.03); correlation analysis
      ├─ 04–05 Cluster ─── K-Means on lat/long, k chosen by elbow + silhouette
      ├─ 06 Model ──────── IQR outlier filter → 1,289 rows; 4 models × 5-fold CV on log(price)
      ├─ 07 Predict ────── interactive CLI price estimator (XGBoost)
      └─ 08 Map ────────── interactive Folium maps by cluster and price tier
```

Key modelling decisions:

- **Log-transformed target.** Raw prices range from ₱300K to ₱2.5B, and the mean is 3.5× the median. The models are trained on `ln(price)`, and every metric is reported after converting back to pesos so the errors are real peso amounts.
- **Outliers are removed for modelling only.** The IQR filter and the 99th-percentile caps on floor and land area are applied to the regression dataset only. The exploratory analysis and the clustering keep all 1,451 listings, because luxury properties are real data points.
- **Every model is evaluated with cross-validation.** Each model is scored with 5-fold CV rather than a single train/test split, so the tables show how consistent each model is as well as how accurate it is.

## Limitations

- **Asking prices, not sale prices.** The model predicts what a property is likely to be *listed* for.
- **Six features only.** The data has no property age, condition, developer, amenities or distance to the CBD. Luxury properties above about ₱30M are hard to predict for this reason.
- **Platform bias.** All listings come from one site (Lamudi), so Luzon and developers that list heavily there (e.g. Camella) are over-represented. Mindanao has only 53 listings.
- **Approximate coordinates.** The coordinates were geocoded from location text, not recorded as exact property positions.
- **Single snapshot.** The data was scraped in March 2024 and has no time dimension, so it cannot show trends.

## What I'd do next

- Add location features from OpenStreetMap (nearby amenities) and PSA or DTI socio-economic indicators
- Tune hyperparameters with `RandomizedSearchCV` (current settings are sensible but untuned)
- Train a separate model for each regional cluster
- Serve the XGBoost model in a small Streamlit app for buyers

## Running it

Requires Python 3.9–3.12.

```bash
pip install -r requirements.txt

python 01_data_cleaning.py        # must run first: writes PH_Housing_Cleaned.csv
python 02_price_distribution.py
python 03_correlation_heatmap.py
python 04_elbow_silhouette.py
python 05_geographic_clusters.py
python 06_regression.py           # model comparison, figures, results JSON
python 07_predict_price.py        # interactive: enter a property, get a price
python 08_folium_maps.py          # open the generated map_*.html in a browser
```

Figures are saved as `fig_*.png`, the regression metrics as `results_regression_summary.json`, and the maps as `map_clusters.html`, `map_prices.html` and `map_combined.html`.

| File | Purpose |
|---|---|
| `01_data_cleaning.py` | Missing-value handling, median imputation, feature engineering |
| `02_price_distribution.py` | Raw vs. log-transformed price histograms |
| `03_correlation_heatmap.py` | Pearson correlations between numeric features |
| `04_elbow_silhouette.py` | Elbow and silhouette analysis for k = 3–12 |
| `05_geographic_clusters.py` | K-Means (k = 5) and per-cluster profiles |
| `06_regression.py` | Four-model comparison with 5-fold CV, plus diagnostic plots |
| `07_predict_price.py` | Interactive price prediction with XGBoost |
| `08_folium_maps.py` | Interactive cluster, price-tier and combined maps |

## Data

`PH_Housing.csv` contains 1,500 residential listings scraped from [Lamudi](https://www.lamudi.com.ph/) in March 2024, with coordinates geocoded using GeoPy. Source: Dungaran, K. (2024). *Philippines Housing Market* [Data set]. Kaggle. <https://www.kaggle.com/datasets/klekzee/phillipines-housing-market>

## About

**Adrian E. Pintacasi**, BS Business Administration (Business Analytics), Silliman University

This was my capstone project for *BA AN 32P – Fundamentals of Predictive Analytics* (April 2026).
