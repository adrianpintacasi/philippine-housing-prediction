# Philippine Housing Market Price Prediction

A machine learning analysis of Philippine residential property listings, comparing four regression models for price prediction and using K-Means clustering to identify geographic submarkets.

This repository contains the code and analysis pipeline for the capstone project *Mapping Value and Predicting Prices: A Data-Driven Analysis of the Philippine Housing Market* by Adrian E. Pintacasi.

## What this project does

* Trains and compares four regression models (Linear Regression, Decision Tree, Random Forest, XGBoost) on 1,289 Philippine property listings
* Applies K-Means clustering to identify five geographic housing zones across the country
* Generates interactive Folium maps visualizing cluster assignments and price tiers
* Provides an interactive script for predicting property prices from six basic features

## Key results

|Model|R²|MAE|MAPE|
|-|-|-|-|
|Linear Regression|0.2076|₱5.24M|47.71%|
|Decision Tree|0.6167|₱3.58M|34.09%|
|Random Forest|0.7822|₱2.70M|27.11%|
|**XGBoost**|**0.8096**|**₱2.51M**|**24.72%**|

XGBoost achieved the best performance, explaining approximately 81% of price variance using only six features (Bedrooms, Bathrooms, Floor Area, Land Area, Latitude, Longitude).

## Dataset

The dataset contains 1,500 residential property listings scraped from Lamudi, obtained from Kaggle:

https://www.kaggle.com/datasets/klekzee/phillipines-housing-market

Download `PH\_Housing.csv` from the link above and place it in the same folder as the scripts before running anything.

## Requirements

Python 3.9 or later. All required libraries can be installed with:

```
pip install -r requirements.txt
```

## Usage

Run the scripts in order. Each script depends on the output of Script 01, so Script 01 must be run first.

```
python 01\_data\_cleaning.py
python 02\_price\_distribution.py
python 03\_correlation\_heatmap.py
python 04\_elbow\_silhouette.py
python 05\_geographic\_clusters.py
python 06\_regression.py
python 07\_predict\_price.py
python 08\_folium\_maps.py
```

Figures are saved as PNG files, regression results are saved to `results\_regression\_summary.json`, and the interactive maps are saved as HTML files that can be opened in any web browser.

## Files

|File|Purpose|
|-|-|
|`01\_data\_cleaning.py`|Cleans raw data, imputes missing values, engineers features|
|`02\_price\_distribution.py`|Histogram of raw and log-transformed prices|
|`03\_correlation\_heatmap.py`|Pearson correlation heatmap of numeric features|
|`04\_elbow\_silhouette.py`|Determines optimal k for K-Means clustering|
|`05\_geographic\_clusters.py`|Applies K-Means (k=5) and profiles each cluster|
|`06\_regression.py`|Trains and evaluates all four regression models|
|`07\_predict\_price.py`|Interactive price prediction using XGBoost|
|`08\_folium\_maps.py`|Generates three interactive HTML maps|
|`requirements.txt`|Python library dependencies with pinned versions|



## Author

**Adrian E. Pintacasi**
BSBA Business Analytics III

